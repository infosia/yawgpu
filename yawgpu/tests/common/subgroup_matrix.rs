//! Shared body of the `chromium-experimental-subgroup-matrix` e2e (Block 170), included
//! by `e2e_{metal,vulkan}_subgroup_matrix.rs` through `#[path]`. The scenarios follow
//! Dawn's `SubgroupMatrixTests.cpp`.
//!
//! Proves on hardware:
//! - the adapter advertises the feature, the `WGPUAdapterPropertiesSubgroupMatrixConfigs`
//!   chain returns a non-empty list of well-formed configurations through both the
//!   adapter and the device query, and a device requesting only this feature also
//!   reports `Subgroups`;
//! - for every advertised configuration, `subgroupMatrixMultiply` and
//!   `subgroupMatrixMultiplyAccumulate` match a CPU reference bit for bit, with
//!   column-major and row-major load / store;
//! - `subgroupMatrixScalarAdd` / `Subtract` / `Multiply` match a CPU reference;
//! - the pipeline rules fire as validation errors on a real device: a matrix shape no
//!   configuration has, and a workgroup x-size that is not a multiple of the maximum
//!   subgroup size.
//!
//! On Vulkan, run under `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`; the expected
//! validation output is empty.

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};

use yawgpu::native;
use yawgpu::{YaWGPUInstanceBackendSelect, YAWGPU_STYPE_INSTANCE_BACKEND_SELECT};
use yawgpu_test::wait;
use yawgpu_test::{real_backend_skip_reason, RealBackend};

type Config = native::WGPUSubgroupMatrixConfig;
type Component = native::WGPUSubgroupMatrixComponentType;

const F32: Component = native::WGPUSubgroupMatrixComponentType_F32;
const F16: Component = native::WGPUSubgroupMatrixComponentType_F16;
const U32: Component = native::WGPUSubgroupMatrixComponentType_U32;
const I32: Component = native::WGPUSubgroupMatrixComponentType_I32;
const U8: Component = native::WGPUSubgroupMatrixComponentType_U8;
const I8: Component = native::WGPUSubgroupMatrixComponentType_I8;

/// `required`: the adapter must advertise the feature; otherwise a missing feature skips.
pub fn adapter_advertises_subgroup_matrix(backend: RealBackend, select: u32, required: bool) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let instance = create_backend_instance(select);
        let adapter = request_adapter(instance);
        if !adapter_has_matrix(adapter) {
            assert!(!required, "the adapter must advertise subgroup matrix");
            eprintln!("skip: adapter does not expose subgroup matrix");
            release_adapter_instance(adapter, instance);
            return;
        }
        assert!(
            yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_Subgroups) != 0,
            "subgroup matrix implies an adapter with subgroups"
        );

        let (configs, min, max) = query_configs(|info| yawgpu::wgpuAdapterGetInfo(adapter, info));
        assert!(
            !configs.is_empty(),
            "an advertised feature has configurations"
        );
        for config in &configs {
            eprintln!("config: {}", describe(config));
            assert!(config.M > 0 && config.N > 0 && config.K > 0);
            assert!(config.minSubgroupSize.is_power_of_two());
            assert!(config.maxSubgroupSize.is_power_of_two());
            assert!(config.minSubgroupSize >= 4 && config.maxSubgroupSize <= 128);
            assert!(config.minSubgroupSize <= config.maxSubgroupSize);
            assert!(config.minSubgroupSize >= min && config.maxSubgroupSize <= max);
            assert!(wgsl_type(config.componentType).is_some());
            assert!(wgsl_type(config.resultComponentType).is_some());
        }

        // Requesting only subgroup matrix enables subgroups too, and the device reports
        // the same configurations without having to enable anything else.
        let device = request_device_with(
            instance,
            adapter,
            &[native::WGPUFeatureName_ChromiumExperimentalSubgroupMatrix],
        );
        assert!(
            yawgpu::wgpuDeviceHasFeature(
                device,
                native::WGPUFeatureName_ChromiumExperimentalSubgroupMatrix
            ) != 0
        );
        assert!(
            yawgpu::wgpuDeviceHasFeature(device, native::WGPUFeatureName_Subgroups) != 0,
            "subgroup matrix must implicitly enable subgroups"
        );
        let (device_configs, _, _) =
            query_configs(|info| yawgpu::wgpuDeviceGetAdapterInfo(device, info));
        assert_eq!(device_configs.len(), configs.len());
        for (a, b) in device_configs.iter().zip(&configs) {
            assert_eq!(describe(a), describe(b));
        }

        yawgpu::wgpuDeviceRelease(device);
        release_adapter_instance(adapter, instance);
    }
}

pub fn matrix_multiply_matches_reference(backend: RealBackend, select: u32) {
    with_matrix_device(backend, select, |ctx| unsafe {
        let mut ran = 0;
        for config in &ctx.configs {
            for accumulate in [false, true] {
                for column_major in [true, false] {
                    run_multiply_case(ctx, config, accumulate, column_major);
                    ran += 1;
                }
            }
        }
        assert!(ran > 0, "no runnable configuration");
    });
}

pub fn matrix_scalar_arithmetic_matches_reference(backend: RealBackend, select: u32) {
    with_matrix_device(backend, select, |ctx| unsafe {
        let mut ran = 0;
        for config in &ctx.configs {
            // Dawn's scalar shader does not cover the packed 8-bit element types.
            if is_8bit(config.componentType) {
                continue;
            }
            for op in [ScalarOp::Add, ScalarOp::Subtract, ScalarOp::Multiply] {
                for column_major in [true, false] {
                    run_scalar_case(ctx, config, op, column_major);
                    ran += 1;
                }
            }
        }
        assert!(ran > 0, "no runnable configuration");
    });
}

pub fn subgroup_matrix_validation_errors(backend: RealBackend, select: u32) {
    with_matrix_device(backend, select, |ctx| unsafe {
        let mut unsupported_m = 1;
        while ctx.configs.iter().any(|c| c.M == unsupported_m) {
            unsupported_m += 1;
        }
        for config in &ctx.configs {
            let header = shader_header(config);
            let body = "@compute @workgroup_size(SubgroupMaxSize)\n\
                 fn main() {\n\
                     let lhs = subgroup_matrix_left<ComponentType, K, M>();\n\
                     let rhs = subgroup_matrix_right<ComponentType, N, K>();\n\
                     _ = subgroupMatrixMultiply<ResultComponentType>(lhs, rhs);\n\
                 }\n";

            // The advertised shape compiles.
            let supported = format!(
                "{header}const M = {};\nconst N = {};\nconst K = {};\n\
                 const SubgroupMaxSize = {};\n{body}",
                config.M, config.N, config.K, ctx.max_subgroup_size
            );
            ctx.errors.lock().expect("error lock").clear();
            let module = create_wgsl_module(ctx.device, &supported);
            let pipeline = create_compute_pipeline(ctx.device, module);
            assert!(
                ctx.errors.lock().expect("error lock").is_empty(),
                "{}: advertised shape must compile: {:?}",
                describe(config),
                ctx.errors.lock().expect("error lock")
            );
            yawgpu::wgpuComputePipelineRelease(pipeline);
            yawgpu::wgpuShaderModuleRelease(module);

            // The same shape with an M no configuration has is rejected.
            let unsupported = format!(
                "{header}const M = {unsupported_m};\nconst N = {};\nconst K = {};\n\
                 const SubgroupMaxSize = {};\n{body}",
                config.N, config.K, ctx.max_subgroup_size
            );
            expect_pipeline_error(ctx, &unsupported, "unsupported subgroup matrix");

            // Workgroup x must be a multiple of the maximum subgroup size.
            let half = format!(
                "{header}const M = {};\nconst N = {};\n\
                 @compute @workgroup_size({}, 2)\n\
                 fn main() {{\n\
                     _ = subgroup_matrix_result<ResultComponentType, N, M>();\n\
                 }}\n",
                config.M,
                config.N,
                ctx.max_subgroup_size / 2
            );
            expect_pipeline_error(ctx, &half, "maximum subgroup size");
        }
    });
}

// ---- cases ----

unsafe fn run_multiply_case(ctx: &Context, config: &Config, accumulate: bool, column_major: bool) {
    let trace = format!(
        "{} accumulate={accumulate} column_major={column_major}",
        describe(config)
    );
    let input_div = if is_8bit(config.componentType) {
        "/4"
    } else {
        ""
    };
    let result_div = if is_8bit(config.resultComponentType) {
        "/4"
    } else {
        ""
    };
    let major = if column_major {
        "col_major"
    } else {
        "row_major"
    };
    let (left_stride, right_stride, result_stride) = if column_major {
        ("M", "K", "M")
    } else {
        ("K", "N", "N")
    };
    let operation = if accumulate {
        format!(
            "var result = subgroupMatrixLoad<ResultType, {major}>(&output, 0, kResultStride);\n\
             result = subgroupMatrixMultiplyAccumulate(lhs, rhs, result);\n"
        )
    } else {
        "let result = subgroupMatrixMultiply<ResultComponentType>(lhs, rhs);\n".to_owned()
    };
    let shader = format!(
        "{header}\
         alias InputArrayType = {input_array};\n\
         alias ResultArrayType = {result_array};\n\
         alias LeftType = subgroup_matrix_left<ComponentType, K, M>;\n\
         alias RightType = subgroup_matrix_right<ComponentType, N, K>;\n\
         alias ResultType = subgroup_matrix_result<ResultComponentType, N, M>;\n\
         const M = {m};\nconst N = {n};\nconst K = {k};\n\
         const kInputArraySize = (K*M + N*K){input_div};\n\
         const kResultArraySize = (M*N){result_div};\n\
         const kLoadOffset = K * M{input_div};\n\
         const kLeftStride = {left_stride}{input_div};\n\
         const kRightStride = {right_stride}{input_div};\n\
         const kResultStride = {result_stride}{result_div};\n\
         const SubgroupMaxSize = {max};\n\
         @group(0) @binding(0) var<storage, read> inputs : array<InputArrayType, kInputArraySize>;\n\
         @group(0) @binding(1) var<storage, read_write> output : array<ResultArrayType, kResultArraySize>;\n\
         @compute @workgroup_size(SubgroupMaxSize)\n\
         fn main(@builtin(subgroup_id) sgid: u32) {{\n\
             if sgid != 0 {{\n  return;\n}}\n\
             let lhs = subgroupMatrixLoad<LeftType, {major}>(&inputs, 0, kLeftStride);\n\
             let rhs = subgroupMatrixLoad<RightType, {major}>(&inputs, kLoadOffset, kRightStride);\n\
             {operation}\
             subgroupMatrixStore<{major}>(&output, 0, result, kResultStride);\n\
         }}\n",
        header = shader_header(config),
        input_array = scalar_shader_type(config.componentType),
        result_array = scalar_shader_type(config.resultComponentType),
        m = config.M,
        n = config.N,
        k = config.K,
        max = ctx.max_subgroup_size,
    );

    let mut lhs = Matrix::new(config.K, config.M, config.componentType, column_major);
    let mut rhs = Matrix::new(config.N, config.K, config.componentType, column_major);
    let mut acc = Matrix::new(config.N, config.M, config.resultComponentType, column_major);
    lhs.fill(0);
    rhs.fill(1);
    if accumulate {
        acc.fill(3);
    }
    let mut input_bytes = lhs.data.clone();
    input_bytes.extend_from_slice(&rhs.data);

    let mut expected = Matrix::new(config.N, config.M, config.resultComponentType, column_major);
    reference_multiply(&mut expected, &lhs, &rhs, &acc);

    let actual = run_two_buffer_pipeline(ctx, &shader, &input_bytes, &acc.data, &trace);
    assert_eq!(
        actual, expected.data,
        "{trace}: matrix multiply result differs from the CPU reference"
    );
}

#[derive(Clone, Copy, Debug)]
enum ScalarOp {
    Add,
    Subtract,
    Multiply,
}

unsafe fn run_scalar_case(ctx: &Context, config: &Config, op: ScalarOp, column_major: bool) {
    let trace = format!("{} {op:?} column_major={column_major}", describe(config));
    let major = if column_major {
        "col_major"
    } else {
        "row_major"
    };
    let builtin = match op {
        ScalarOp::Add => "subgroupMatrixScalarAdd",
        ScalarOp::Subtract => "subgroupMatrixScalarSubtract",
        ScalarOp::Multiply => "subgroupMatrixScalarMultiply",
    };
    let shader = format!(
        "{header}\
         alias ScalarShaderType = {scalar};\n\
         const M = {m};\nconst K = {k};\n\
         alias LeftType = subgroup_matrix_left<ComponentType, K, M>;\n\
         const kMatrixDataSize = (K * M);\n\
         const kStride = {stride};\n\
         const SubgroupMaxSize = {max};\n\
         @group(0) @binding(0) var<storage, read> inputs : array<ScalarShaderType, kMatrixDataSize>;\n\
         @group(0) @binding(1) var<storage, read_write> output : array<ScalarShaderType, kMatrixDataSize>;\n\
         @compute @workgroup_size(SubgroupMaxSize)\n\
         fn main(@builtin(subgroup_id) sgid: u32) {{\n\
             if sgid != 0 {{\n  return;\n}}\n\
             let lhs = subgroupMatrixLoad<LeftType, {major}>(&inputs, 0, kStride);\n\
             let result = {builtin}(lhs, 5);\n\
             subgroupMatrixStore<{major}>(&output, 0, result, kStride);\n\
         }}\n",
        header = shader_header(config),
        scalar = scalar_shader_type(config.componentType),
        m = config.M,
        k = config.K,
        stride = if column_major { "M" } else { "K" },
        max = ctx.max_subgroup_size,
    );

    let mut lhs = Matrix::new(config.K, config.M, config.componentType, column_major);
    lhs.fill(0);
    let mut expected = Matrix::new(config.K, config.M, config.componentType, column_major);
    for r in 0..expected.rows {
        for c in 0..expected.cols {
            if is_float(config.componentType) {
                let v = lhs.get_float(c, r);
                expected.set_float(
                    match op {
                        ScalarOp::Add => v + 5.0,
                        ScalarOp::Subtract => v - 5.0,
                        ScalarOp::Multiply => v * 5.0,
                    },
                    c,
                    r,
                );
            } else {
                let v = lhs.get_int(c, r);
                expected.set_int(
                    match op {
                        ScalarOp::Add => v + 5,
                        ScalarOp::Subtract => v - 5,
                        ScalarOp::Multiply => v * 5,
                    },
                    c,
                    r,
                );
            }
        }
    }

    let zeros = vec![0u8; lhs.data.len()];
    let actual = run_two_buffer_pipeline(ctx, &shader, &lhs.data, &zeros, &trace);
    assert_eq!(
        actual, expected.data,
        "{trace}: matrix scalar result differs from the CPU reference"
    );
}

/// Runs `shader` once with `input` at binding 0 (read-only storage) and `output_init` at
/// binding 1 (read-write storage), and returns the first `output_init.len()` output bytes.
unsafe fn run_two_buffer_pipeline(
    ctx: &Context,
    shader: &str,
    input: &[u8],
    output_init: &[u8],
    trace: &str,
) -> Vec<u8> {
    ctx.errors.lock().expect("error lock").clear();
    let input_size = input.len().next_multiple_of(4) as u64;
    let output_size = output_init.len().next_multiple_of(4) as u64;
    let storage_dst = native::WGPUBufferUsage_Storage | native::WGPUBufferUsage_CopyDst;
    let input_buffer = create_buffer_sized(ctx.device, input_size, storage_dst);
    let output_buffer = create_buffer_sized(
        ctx.device,
        output_size,
        storage_dst | native::WGPUBufferUsage_CopySrc,
    );
    let readback = create_buffer_sized(
        ctx.device,
        output_size,
        native::WGPUBufferUsage_CopyDst | native::WGPUBufferUsage_MapRead,
    );
    let mut padded_input = input.to_vec();
    padded_input.resize(input_size as usize, 0);
    let mut padded_output = output_init.to_vec();
    padded_output.resize(output_size as usize, 0);
    yawgpu::wgpuQueueWriteBuffer(
        ctx.queue,
        input_buffer,
        0,
        padded_input.as_ptr().cast(),
        padded_input.len(),
    );
    yawgpu::wgpuQueueWriteBuffer(
        ctx.queue,
        output_buffer,
        0,
        padded_output.as_ptr().cast(),
        padded_output.len(),
    );

    let module = create_wgsl_module(ctx.device, shader);
    let pipeline = create_compute_pipeline(ctx.device, module);
    assert!(
        ctx.errors.lock().expect("error lock").is_empty(),
        "{trace}: pipeline creation raised {:?}\n{shader}",
        ctx.errors.lock().expect("error lock")
    );
    assert!(!pipeline.is_null());
    let layout = yawgpu::wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    let entries = [
        buffer_entry(0, input_buffer, input_size),
        buffer_entry(1, output_buffer, output_size),
    ];
    let bind_group_descriptor = native::WGPUBindGroupDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout,
        entryCount: entries.len(),
        entries: entries.as_ptr(),
    };
    let bind_group = yawgpu::wgpuDeviceCreateBindGroup(ctx.device, &bind_group_descriptor);
    assert!(!bind_group.is_null());

    let encoder = yawgpu::wgpuDeviceCreateCommandEncoder(ctx.device, std::ptr::null());
    let pass = yawgpu::wgpuCommandEncoderBeginComputePass(encoder, std::ptr::null());
    yawgpu::wgpuComputePassEncoderSetPipeline(pass, pipeline);
    yawgpu::wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, std::ptr::null());
    yawgpu::wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    yawgpu::wgpuComputePassEncoderEnd(pass);
    yawgpu::wgpuCommandEncoderCopyBufferToBuffer(
        encoder,
        output_buffer,
        0,
        readback,
        0,
        output_size,
    );
    let command_buffer = yawgpu::wgpuCommandEncoderFinish(encoder, std::ptr::null());
    yawgpu::wgpuQueueSubmit(ctx.queue, 1, &command_buffer);
    yawgpu::wgpuCommandBufferRelease(command_buffer);
    yawgpu::wgpuComputePassEncoderRelease(pass);
    yawgpu::wgpuCommandEncoderRelease(encoder);

    let mut bytes = read_buffer(ctx.instance, readback, output_size as usize);
    bytes.truncate(output_init.len());
    assert!(
        ctx.errors.lock().expect("error lock").is_empty(),
        "{trace}: execution raised {:?}",
        ctx.errors.lock().expect("error lock")
    );

    yawgpu::wgpuBindGroupRelease(bind_group);
    yawgpu::wgpuBindGroupLayoutRelease(layout);
    yawgpu::wgpuComputePipelineRelease(pipeline);
    yawgpu::wgpuShaderModuleRelease(module);
    yawgpu::wgpuBufferRelease(readback);
    yawgpu::wgpuBufferRelease(output_buffer);
    yawgpu::wgpuBufferRelease(input_buffer);
    bytes
}

unsafe fn expect_pipeline_error(ctx: &Context, shader: &str, needle: &str) {
    ctx.errors.lock().expect("error lock").clear();
    let module = create_wgsl_module(ctx.device, shader);
    assert!(
        ctx.errors.lock().expect("error lock").is_empty(),
        "shader module itself must be valid: {shader}\n{:?}",
        ctx.errors.lock().expect("error lock")
    );
    let pipeline = create_compute_pipeline(ctx.device, module);
    let captured = ctx.errors.lock().expect("error lock");
    assert_eq!(
        captured.len(),
        1,
        "expected exactly one pipeline validation error for:\n{shader}\ngot {captured:?}"
    );
    let message = format!("{:?}", captured[0]);
    assert!(
        message.contains(needle),
        "error for:\n{shader}\nmust mention {needle:?}, got {message}"
    );
    drop(captured);
    if !pipeline.is_null() {
        yawgpu::wgpuComputePipelineRelease(pipeline);
    }
    yawgpu::wgpuShaderModuleRelease(module);
}

// ---- CPU reference ----

/// One matrix in its buffer layout, mirroring the `Matrix` helper of Dawn's test.
struct Matrix {
    cols: u32,
    rows: u32,
    component: Component,
    column_major: bool,
    data: Vec<u8>,
}

impl Matrix {
    fn new(cols: u32, rows: u32, component: Component, column_major: bool) -> Self {
        let len = (cols * rows) as usize * byte_size(component);
        Self {
            cols,
            rows,
            component,
            column_major,
            data: vec![0; len],
        }
    }

    /// Fills with values that stay exact through a small multiply, cycling through an
    /// odd number of them so a majorness or stride bug changes the result.
    fn fill(&mut self, value_offset: u32) {
        const FLOATS: [f32; 9] = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
        const SINTS: [i64; 9] = [-43, -32, -21, -10, 0, 10, 21, 32, 43];
        const UINTS: [i64; 9] = [0, 1, 2, 3, 11, 23, 37, 71, 101];
        for r in 0..self.rows {
            for c in 0..self.cols {
                let index = ((value_offset + c + r * self.cols) % 9) as usize;
                match self.component {
                    F16 | F32 => self.set_float(FLOATS[index], c, r),
                    I32 | I8 => self.set_int(SINTS[index], c, r),
                    _ => self.set_int(UINTS[index], c, r),
                }
            }
        }
    }

    fn offset(&self, c: u32, r: u32) -> usize {
        let index = if self.column_major {
            c * self.rows + r
        } else {
            c + r * self.cols
        };
        index as usize * byte_size(self.component)
    }

    fn get_float(&self, c: u32, r: u32) -> f32 {
        let at = self.offset(c, r);
        match self.component {
            F32 => f32::from_ne_bytes(self.data[at..at + 4].try_into().expect("4 bytes")),
            F16 => f16_bits_to_f32(u16::from_ne_bytes(
                self.data[at..at + 2].try_into().expect("2 bytes"),
            )),
            _ => panic!("not a float matrix"),
        }
    }

    fn set_float(&mut self, value: f32, c: u32, r: u32) {
        let at = self.offset(c, r);
        match self.component {
            F32 => self.data[at..at + 4].copy_from_slice(&value.to_ne_bytes()),
            F16 => self.data[at..at + 2].copy_from_slice(&f32_to_f16_bits(value).to_ne_bytes()),
            _ => panic!("not a float matrix"),
        }
    }

    fn get_int(&self, c: u32, r: u32) -> i64 {
        let at = self.offset(c, r);
        match self.component {
            U32 => i64::from(u32::from_ne_bytes(
                self.data[at..at + 4].try_into().expect("4 bytes"),
            )),
            I32 => i64::from(i32::from_ne_bytes(
                self.data[at..at + 4].try_into().expect("4 bytes"),
            )),
            U8 => i64::from(self.data[at]),
            I8 => i64::from(self.data[at] as i8),
            _ => panic!("not an integer matrix"),
        }
    }

    fn set_int(&mut self, value: i64, c: u32, r: u32) {
        let at = self.offset(c, r);
        match self.component {
            U32 => self.data[at..at + 4].copy_from_slice(&(value as u32).to_ne_bytes()),
            I32 => self.data[at..at + 4].copy_from_slice(&(value as i32).to_ne_bytes()),
            U8 | I8 => self.data[at] = value as u8,
            _ => panic!("not an integer matrix"),
        }
    }
}

/// `expected = acc + lhs * rhs`, rounding through the result type like the GPU does.
fn reference_multiply(expected: &mut Matrix, lhs: &Matrix, rhs: &Matrix, acc: &Matrix) {
    for r in 0..expected.rows {
        for c in 0..expected.cols {
            if is_float(expected.component) {
                let mut product = 0.0f32;
                for k in 0..lhs.cols {
                    product += lhs.get_float(k, r) * rhs.get_float(c, k);
                }
                let mut value = acc.get_float(c, r) + product;
                if expected.component == F16 {
                    value = f16_bits_to_f32(f32_to_f16_bits(value));
                }
                expected.set_float(value, c, r);
            } else {
                let mut value = acc.get_int(c, r);
                for k in 0..lhs.cols {
                    value += lhs.get_int(k, r) * rhs.get_int(c, k);
                }
                expected.set_int(value, c, r);
            }
        }
    }
}

/// Converts a finite `f32` whose value is exactly representable as a normal `f16` (or
/// zero), which is all the reference data needs.
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if value == 0.0 {
        return sign;
    }
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = bits & 0x007F_FFFF;
    assert!(
        (1..31).contains(&exponent) && mantissa & 0x1FFF == 0,
        "{value} is not exactly representable as a normal f16"
    );
    sign | ((exponent as u16) << 10) | (mantissa >> 13) as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1F);
    let mantissa = f32::from(bits & 0x3FF);
    if exponent == 0 {
        assert_eq!(mantissa, 0.0, "subnormal f16 is outside the reference data");
        return sign * 0.0;
    }
    assert!(
        exponent != 31,
        "non-finite f16 is outside the reference data"
    );
    sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
}

// ---- component helpers ----

fn wgsl_type(component: Component) -> Option<&'static str> {
    Some(match component {
        F32 => "f32",
        F16 => "f16",
        U32 => "u32",
        I32 => "i32",
        U8 => "u8",
        I8 => "i8",
        _ => return None,
    })
}

/// The WGSL type of the storage array a matrix of `component` is loaded from: 8-bit
/// elements are packed four to a 32-bit word.
fn scalar_shader_type(component: Component) -> &'static str {
    match component {
        F32 => "f32",
        F16 => "f16",
        U32 | U8 => "u32",
        _ => "i32",
    }
}

fn byte_size(component: Component) -> usize {
    match component {
        F16 => 2,
        U8 | I8 => 1,
        _ => 4,
    }
}

fn is_8bit(component: Component) -> bool {
    component == U8 || component == I8
}

fn is_float(component: Component) -> bool {
    component == F32 || component == F16
}

fn uses_f16(config: &Config) -> bool {
    config.componentType == F16 || config.resultComponentType == F16
}

fn describe(config: &Config) -> String {
    format!(
        "{}x{}x{} {} -> {} [{}, {}]",
        config.M,
        config.N,
        config.K,
        wgsl_type(config.componentType).unwrap_or("?"),
        wgsl_type(config.resultComponentType).unwrap_or("?"),
        config.minSubgroupSize,
        config.maxSubgroupSize
    )
}

/// `enable` directives plus the `ComponentType` / `ResultComponentType` aliases.
fn shader_header(config: &Config) -> String {
    format!(
        "enable chromium_experimental_subgroup_matrix;\nenable subgroups;\n{}\
         alias ComponentType = {};\nalias ResultComponentType = {};\n",
        if uses_f16(config) {
            "enable f16;\n"
        } else {
            ""
        },
        wgsl_type(config.componentType).expect("known component type"),
        wgsl_type(config.resultComponentType).expect("known component type"),
    )
}

// ---- device setup ----

struct Context {
    instance: native::WGPUInstance,
    device: native::WGPUDevice,
    queue: native::WGPUQueue,
    errors: Arc<Mutex<Vec<yawgpu_core::DeviceError>>>,
    /// The adapter's configurations this device can compile (f16 ones only with
    /// `shader-f16`).
    configs: Vec<Config>,
    max_subgroup_size: u32,
}

fn with_matrix_device(backend: RealBackend, select: u32, body: impl FnOnce(&Context)) {
    if real_backend_skip_reason(backend).is_some() {
        return;
    }
    unsafe {
        let instance = create_backend_instance(select);
        let adapter = request_adapter(instance);
        if !adapter_has_matrix(adapter) {
            eprintln!("skip: adapter does not expose subgroup matrix");
            release_adapter_instance(adapter, instance);
            return;
        }
        let (mut configs, _, max_subgroup_size) =
            query_configs(|info| yawgpu::wgpuAdapterGetInfo(adapter, info));
        let mut features = vec![native::WGPUFeatureName_ChromiumExperimentalSubgroupMatrix];
        if yawgpu::wgpuAdapterHasFeature(adapter, native::WGPUFeatureName_ShaderF16) != 0 {
            features.push(native::WGPUFeatureName_ShaderF16);
        } else {
            configs.retain(|config| !uses_f16(config));
        }
        let device = request_device_with(instance, adapter, &features);
        let errors = install_error_capture(device);
        let queue = yawgpu::wgpuDeviceGetQueue(device);
        let ctx = Context {
            instance,
            device,
            queue,
            errors,
            configs,
            max_subgroup_size,
        };
        body(&ctx);
        yawgpu::wgpuQueueRelease(queue);
        yawgpu::wgpuDeviceRelease(device);
        release_adapter_instance(adapter, instance);
    }
}

/// Runs an adapter-info query with the configuration node chained and returns the
/// configurations plus the reported subgroup size range.
unsafe fn query_configs(
    query: impl FnOnce(*mut native::WGPUAdapterInfo) -> native::WGPUStatus,
) -> (Vec<Config>, u32, u32) {
    let mut node: native::WGPUAdapterPropertiesSubgroupMatrixConfigs = std::mem::zeroed();
    node.chain.sType = native::WGPUSType_AdapterPropertiesSubgroupMatrixConfigs;
    let mut info: native::WGPUAdapterInfo = std::mem::zeroed();
    info.nextInChain = &mut node.chain;
    assert_eq!(query(&mut info), native::WGPUStatus_Success);
    assert_eq!(
        info.nextInChain, &raw mut node.chain,
        "the caller's chain must be preserved"
    );
    let configs = if node.configCount == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(node.configs, node.configCount).to_vec()
    };
    let range = (info.subgroupMinSize, info.subgroupMaxSize);
    yawgpu::wgpuAdapterPropertiesSubgroupMatrixConfigsFreeMembers(node);
    yawgpu::wgpuAdapterInfoFreeMembers(info);
    (configs, range.0, range.1)
}

unsafe fn adapter_has_matrix(adapter: native::WGPUAdapter) -> bool {
    yawgpu::wgpuAdapterHasFeature(
        adapter,
        native::WGPUFeatureName_ChromiumExperimentalSubgroupMatrix,
    ) != 0
}

unsafe fn release_adapter_instance(adapter: native::WGPUAdapter, instance: native::WGPUInstance) {
    yawgpu::wgpuAdapterRelease(adapter);
    yawgpu::wgpuInstanceRelease(instance);
}

// ---- helpers ----

unsafe fn create_buffer_sized(
    device: native::WGPUDevice,
    size: u64,
    usage: native::WGPUBufferUsage,
) -> native::WGPUBuffer {
    let descriptor = native::WGPUBufferDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        usage,
        size,
        mappedAtCreation: 0,
    };
    let buffer = yawgpu::wgpuDeviceCreateBuffer(device, &descriptor);
    assert!(!buffer.is_null());
    buffer
}

fn buffer_entry(binding: u32, buffer: native::WGPUBuffer, size: u64) -> native::WGPUBindGroupEntry {
    native::WGPUBindGroupEntry {
        nextInChain: std::ptr::null_mut(),
        binding,
        buffer,
        offset: 0,
        size,
        sampler: std::ptr::null(),
        textureView: std::ptr::null(),
    }
}

unsafe fn create_compute_pipeline(
    device: native::WGPUDevice,
    module: native::WGPUShaderModule,
) -> native::WGPUComputePipeline {
    let descriptor = native::WGPUComputePipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: std::ptr::null(),
        compute: native::WGPUComputeState {
            nextInChain: std::ptr::null_mut(),
            module,
            entryPoint: string_view("main"),
            constantCount: 0,
            constants: std::ptr::null(),
        },
    };
    yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor)
}

unsafe fn read_buffer(
    instance: native::WGPUInstance,
    buffer: native::WGPUBuffer,
    len: usize,
) -> Vec<u8> {
    let mut status = native::WGPUMapAsyncStatus_Error;
    let callback_info = native::WGPUBufferMapCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(map_callback),
        userdata1: (&mut status as *mut native::WGPUMapAsyncStatus).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future =
        yawgpu::wgpuBufferMapAsync(buffer, native::WGPUMapMode_Read, 0, len, callback_info);
    wait(instance, future);
    assert_eq!(status, native::WGPUMapAsyncStatus_Success);
    let ptr = yawgpu::wgpuBufferGetConstMappedRange(buffer, 0, len);
    assert!(!ptr.is_null());
    let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
    yawgpu::wgpuBufferUnmap(buffer);
    bytes
}

unsafe fn create_wgsl_module(device: native::WGPUDevice, source: &str) -> native::WGPUShaderModule {
    let mut wgsl = native::WGPUShaderSourceWGSL {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_ShaderSourceWGSL,
        },
        code: string_view(source),
    };
    let descriptor = native::WGPUShaderModuleDescriptor {
        nextInChain: (&mut wgsl.chain) as *mut _,
        label: empty_string_view(),
    };
    let module = yawgpu::wgpuDeviceCreateShaderModule(device, &descriptor);
    assert!(!module.is_null());
    module
}

unsafe fn create_backend_instance(select: u32) -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: select,
    };
    let descriptor = native::WGPUInstanceDescriptor {
        nextInChain: (&mut backend.chain) as *mut native::WGPUChainedStruct,
        requiredFeatureCount: 0,
        requiredFeatures: std::ptr::null(),
        requiredLimits: std::ptr::null(),
    };
    let instance = yawgpu::wgpuCreateInstance(&descriptor);
    assert!(!instance.is_null());
    instance
}

unsafe fn request_adapter(instance: native::WGPUInstance) -> native::WGPUAdapter {
    let mut adapter: native::WGPUAdapter = std::ptr::null();
    let callback_info = native::WGPURequestAdapterCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_adapter_callback),
        userdata1: (&mut adapter as *mut native::WGPUAdapter).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuInstanceRequestAdapter(instance, std::ptr::null(), callback_info);
    wait(instance, future);
    assert!(!adapter.is_null());
    adapter
}

unsafe fn request_device_with(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
    features: &[native::WGPUFeatureName],
) -> native::WGPUDevice {
    let descriptor = native::WGPUDeviceDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        requiredFeatureCount: features.len(),
        requiredFeatures: features.as_ptr(),
        requiredLimits: std::ptr::null(),
        defaultQueue: native::WGPUQueueDescriptor {
            nextInChain: std::ptr::null_mut(),
            label: empty_string_view(),
        },
        deviceLostCallbackInfo: std::mem::zeroed(),
        uncapturedErrorCallbackInfo: std::mem::zeroed(),
    };
    let mut device: native::WGPUDevice = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, &descriptor, callback_info);
    wait(instance, future);
    assert!(!device.is_null());
    device
}

unsafe fn install_error_capture(
    device: native::WGPUDevice,
) -> Arc<Mutex<Vec<yawgpu_core::DeviceError>>> {
    let errors = Arc::new(Mutex::new(Vec::new()));
    let captured_errors = Arc::clone(&errors);
    yawgpu::testing_set_uncaptured_error_callback(
        device,
        Some(move |error| captured_errors.lock().expect("error lock").push(error)),
    );
    errors
}

unsafe extern "C" fn request_adapter_callback(
    status: native::WGPURequestAdapterStatus,
    adapter: native::WGPUAdapter,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestAdapterStatus_Success);
    *(userdata1 as *mut native::WGPUAdapter) = adapter;
}

unsafe extern "C" fn request_device_callback(
    status: native::WGPURequestDeviceStatus,
    device: native::WGPUDevice,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    assert_eq!(status, native::WGPURequestDeviceStatus_Success);
    *(userdata1 as *mut native::WGPUDevice) = device;
}

unsafe extern "C" fn map_callback(
    status: native::WGPUMapAsyncStatus,
    _message: native::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    *(userdata1 as *mut native::WGPUMapAsyncStatus) = status;
}

fn string_view(value: &str) -> native::WGPUStringView {
    native::WGPUStringView {
        data: value.as_ptr().cast(),
        length: value.len(),
    }
}

fn empty_string_view() -> native::WGPUStringView {
    native::WGPUStringView {
        data: std::ptr::null(),
        length: 0,
    }
}
