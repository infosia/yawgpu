//! Block 95 compile-churn probe (canary-shaped, real Metal).
//!
//! Mirrors the external CTS `cts_compile_canary` workload
//! (`webgpu-native-cts/specs/compile-canary.md`):
//!
//! - **Phase A** — one shader module, N identical auto-layout compute
//!   pipelines created and released in a loop (cache probe). With codegen
//!   memoization (Block 95 S1) iterations >= 2 skip Tint entirely and the
//!   Metal driver's content-addressed cache absorbs `newLibraryWithSource`,
//!   so per-iteration time collapses versus iteration 1.
//! - **Phase B** — N unique-constant shader variants (module + pipeline per
//!   iteration): the full-compile cost baseline; every variant is expected
//!   to be a distinct driver variant.
//!
//! Pass contract (the canary's exit-0 condition): `A_median < 0.30 x
//! B_median`, with a 2 ms noise floor and a warn band up to 0.70. Pre-fix
//! this test FAILS (A_median ~= B_median) — that failing run is the recorded
//! baseline. Iteration counts are tunable via `YAWGPU_CHURN_ITERS_A` /
//! `YAWGPU_CHURN_ITERS_B`.

#![cfg(feature = "metal")]

use std::os::raw::c_void;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use yawgpu::native;
use yawgpu::{
    YaWGPUInstanceBackendSelect, YAWGPU_INSTANCE_BACKEND_METAL,
    YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
};
use yawgpu_test::{real_backend_skip_reason, wait, RealBackend};

const DEFAULT_ITERS_A: usize = 100;
const DEFAULT_ITERS_B: usize = 50;
/// Healthy cache-hit threshold (canary exit-0): A_median < 0.30 x B_median.
const HEALTHY_RATIO: f64 = 0.30;
/// Noise floor: an A_median at/below this is healthy regardless of ratio.
const NOISE_FLOOR: Duration = Duration::from_millis(2);

/// Returns the churn WGSL: a compute kernel whose body is a long unrolled
/// arithmetic chain (compile cost dominates API overhead). `seed` is folded
/// into the arithmetic so unique seeds cannot be optimized into one variant.
fn churn_shader(seed: u32) -> String {
    let mut body = String::new();
    for i in 0..192u32 {
        body.push_str(&format!(
            "    acc = acc * 1664525u + 1013904223u + (SEED ^ {}u);\n    \
             acc = (acc << 3u) | (acc >> 29u);\n",
            i.wrapping_mul(2654435761)
        ));
    }
    format!(
        r#"
const SEED: u32 = {seed}u;

@group(0) @binding(0) var<storage, read_write> out_data: array<u32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    var acc: u32 = id.x + SEED;
{body}
    out_data[id.x] = acc;
}}
"#
    )
}

fn iters_from_env(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse().ok())
        .filter(|&value| value > 0)
        .unwrap_or(default)
}

struct PhaseStats {
    first: Duration,
    min: Duration,
    median: Duration,
    p90: Duration,
    max: Duration,
}

fn stats(samples: &[Duration]) -> PhaseStats {
    let mut sorted = samples.to_vec();
    sorted.sort();
    let pick = |fraction: f64| {
        let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
        sorted[index]
    };
    PhaseStats {
        first: samples[0],
        min: sorted[0],
        median: pick(0.5),
        p90: pick(0.9),
        max: sorted[sorted.len() - 1],
    }
}

fn print_stats(name: &str, stats: &PhaseStats) {
    println!(
        "{name}: iter1={:?} min={:?} median={:?} p90={:?} max={:?}",
        stats.first, stats.min, stats.median, stats.p90, stats.max
    );
}

#[test]
#[ignore = "manual real-backend test"]
fn metal_compile_churn_identical_pipeline_loop_collapses_vs_unique_variants() {
    if real_backend_skip_reason(RealBackend::Metal).is_some() {
        return;
    }
    let iters_a = iters_from_env("YAWGPU_CHURN_ITERS_A", DEFAULT_ITERS_A);
    let iters_b = iters_from_env("YAWGPU_CHURN_ITERS_B", DEFAULT_ITERS_B);

    unsafe {
        let instance = create_metal_instance();
        let adapter = request_adapter(instance);
        let device = request_device(instance, adapter);
        let errors = install_error_capture(device);

        // Phase A — identical auto-layout pipeline loop against one live module.
        let source_a = churn_shader(0);
        let module = create_wgsl_module(device, &source_a);
        let mut phase_a = Vec::with_capacity(iters_a);
        for _ in 0..iters_a {
            let start = Instant::now();
            let pipeline = create_auto_layout_pipeline(device, module);
            phase_a.push(start.elapsed());
            yawgpu::wgpuComputePipelineRelease(pipeline);
        }
        yawgpu::wgpuShaderModuleRelease(module);

        // Phase B — unique-constant variants: module + pipeline per iteration.
        let mut phase_b = Vec::with_capacity(iters_b);
        for i in 0..iters_b {
            let source = churn_shader(i as u32 + 1);
            let module = create_wgsl_module(device, &source);
            let start = Instant::now();
            let pipeline = create_auto_layout_pipeline(device, module);
            phase_b.push(start.elapsed());
            yawgpu::wgpuComputePipelineRelease(pipeline);
            yawgpu::wgpuShaderModuleRelease(module);
        }

        let stats_a = stats(&phase_a);
        let stats_b = stats(&phase_b);
        print_stats("phase A (identical pipeline)", &stats_a);
        print_stats("phase B (unique variants)   ", &stats_b);
        let ratio = stats_a.median.as_secs_f64() / stats_b.median.as_secs_f64();
        println!("A_median / B_median = {ratio:.3}");

        assert!(
            errors.lock().expect("error lock").is_empty(),
            "device errors during churn: {:?}",
            errors.lock().expect("error lock")
        );
        assert!(
            ratio < HEALTHY_RATIO || stats_a.median <= NOISE_FLOOR,
            "compile-cache probe failed: A_median {:?} is {ratio:.3} x B_median {:?} \
             (healthy: < {HEALTHY_RATIO}, or A_median <= {NOISE_FLOOR:?})",
            stats_a.median,
            stats_b.median,
        );

        yawgpu::wgpuDeviceRelease(device);
        yawgpu::wgpuAdapterRelease(adapter);
        yawgpu::wgpuInstanceRelease(instance);
    }
}

unsafe fn create_auto_layout_pipeline(
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
    let pipeline = yawgpu::wgpuDeviceCreateComputePipeline(device, &descriptor);
    assert!(!pipeline.is_null());
    pipeline
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

unsafe fn create_metal_instance() -> native::WGPUInstance {
    let mut backend = YaWGPUInstanceBackendSelect {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: YAWGPU_STYPE_INSTANCE_BACKEND_SELECT,
        },
        backend: YAWGPU_INSTANCE_BACKEND_METAL,
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

unsafe fn request_device(
    instance: native::WGPUInstance,
    adapter: native::WGPUAdapter,
) -> native::WGPUDevice {
    let mut device: native::WGPUDevice = std::ptr::null();
    let callback_info = native::WGPURequestDeviceCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode: native::WGPUCallbackMode_AllowProcessEvents,
        callback: Some(request_device_callback),
        userdata1: (&mut device as *mut native::WGPUDevice).cast(),
        userdata2: std::ptr::null_mut(),
    };
    let future = yawgpu::wgpuAdapterRequestDevice(adapter, std::ptr::null(), callback_info);
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
