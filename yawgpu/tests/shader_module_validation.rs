use std::os::raw::c_void;

use yawgpu::native;
use yawgpu_test::{assert_device_error, ValidationTest};

#[derive(Default)]
struct CompilationInfoState {
    calls: usize,
    statuses: Vec<native::WGPUCompilationInfoRequestStatus>,
    message_counts: Vec<usize>,
    error_messages: Vec<String>,
}

#[test]
fn descriptor_requires_exactly_one_chained_source() {
    let test = ValidationTest::new();
    unsafe {
        let descriptor = shader_module_descriptor(std::ptr::null_mut());
        let mut module = std::ptr::null();
        assert_device_error!({
            module = yawgpu::wgpuDeviceCreateShaderModule(test.device(), &descriptor);
        });
        assert!(!module.is_null());
        yawgpu::wgpuShaderModuleRelease(module);

        let words = [0x0723_0203_u32, 0, 0, 0, 0];
        let mut spirv = spirv_source(&words);
        let mut wgsl = wgsl_source("");
        wgsl.chain.next = (&mut spirv.chain) as *mut _;
        let descriptor = shader_module_descriptor((&mut wgsl.chain) as *mut _);
        assert_device_error!({
            module = yawgpu::wgpuDeviceCreateShaderModule(test.device(), &descriptor);
        });
        assert!(!module.is_null());
        yawgpu::wgpuShaderModuleRelease(module);
    }
}

#[test]
fn wgsl_parse_and_validation_errors_create_error_modules() {
    let test = ValidationTest::new();
    unsafe {
        assert_wgsl_ok(&test, "@compute @workgroup_size(1) fn main() {}");
        assert_wgsl_error(&test, "not wgsl @@@");
        assert_wgsl_error(
            &test,
            "@fragment fn main(input: vec4<f32>) -> @location(0) vec4<f32> { return input; }",
        );
        assert_wgsl_error(
            &test,
            "@group(0) @binding(1000) var s: sampler; @compute @workgroup_size(1) fn main() { _ = s; }",
        );
    }
}

#[test]
#[cfg(not(feature = "shader-passthrough"))]
fn spirv_source_requires_passthrough_feature() {
    let test = ValidationTest::new();
    unsafe {
        let words = [0x0723_0203_u32, 0, 0, 0, 0];
        let mut source = spirv_source(&words);
        let descriptor = shader_module_descriptor((&mut source.chain) as *mut _);

        test.clear_errors();
        let module = yawgpu::wgpuDeviceCreateShaderModule(test.device(), &descriptor);
        assert!(!module.is_null());
        assert!(test
            .errors()
            .iter()
            .any(|error| error.message.contains("SPIR-V passthrough not enabled")));
        yawgpu::wgpuShaderModuleRelease(module);
    }
}

#[test]
fn duplicate_override_numeric_ids_are_rejected() {
    let test = ValidationTest::new();
    unsafe {
        assert_wgsl_error(
            &test,
            r"
@id(7) override c0: u32;
@id(7) override c1: u32;

struct Buf {
    data: array<u32, 2>,
}

@group(0) @binding(0) var<storage, read_write> buf: Buf;

@compute @workgroup_size(1) fn main() {
    buf.data[0] = c0;
    buf.data[1] = c1;
}
",
        );
    }
}

#[test]
fn unsafe_wgsl_features_are_rejected_by_naga() {
    let test = ValidationTest::new();
    unsafe {
        assert_wgsl_error(
            &test,
            r"
enable chromium_disable_uniformity_analysis;

@compute @workgroup_size(8) fn main(@builtin(local_invocation_id) id: vec3u) {
    if (id.x == 0u) {
        workgroupBarrier();
    }
}
",
        );
        assert_wgsl_error(
            &test,
            r"
@group(0) @binding(0) var textures: binding_array<texture_2d<f32>, 1>;
@fragment fn fs() -> @location(0) u32 {
    let _ = textures[0];
    return 0u;
}
",
        );
        assert_wgsl_error(
            &test,
            r"
enable chromium_experimental_subgroup_matrix;

@compute @workgroup_size(1) fn main() {}
",
        );
    }
}

#[test]
fn shader_module_release_is_safe_for_valid_and_error_modules() {
    let test = ValidationTest::new();
    unsafe {
        let module = create_wgsl_module(&test, "@compute @workgroup_size(1) fn main() {}");
        yawgpu::wgpuShaderModuleRelease(module);

        let mut error_module = std::ptr::null();
        assert_device_error!({
            error_module = create_wgsl_module(&test, "not wgsl @@@");
        });
        assert!(!error_module.is_null());
        yawgpu::wgpuShaderModuleRelease(error_module);
    }
}

#[test]
fn invalid_module_compilation_info_reports_error_message() {
    let test = ValidationTest::new();
    unsafe {
        let mut module = std::ptr::null();
        assert_device_error!({
            module = create_wgsl_module(&test, "not wgsl @@@");
        });

        let mut state = CompilationInfoState::default();
        get_compilation_info(
            module,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );

        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.calls, 1);
        assert_eq!(
            state.statuses,
            vec![native::WGPUCompilationInfoRequestStatus_Success]
        );
        assert_eq!(state.message_counts, vec![1]);
        assert_eq!(state.error_messages.len(), 1);
        assert!(!state.error_messages[0].is_empty());

        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.calls, 1);
        yawgpu::wgpuShaderModuleRelease(module);
    }
}

#[test]
fn valid_module_compilation_info_has_no_messages() {
    let test = ValidationTest::new();
    unsafe {
        let module = create_wgsl_module(&test, "@compute @workgroup_size(1) fn main() {}");
        let mut state = CompilationInfoState::default();
        get_compilation_info(
            module,
            native::WGPUCallbackMode_AllowProcessEvents,
            &mut state,
        );

        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.calls, 1);
        assert_eq!(
            state.statuses,
            vec![native::WGPUCompilationInfoRequestStatus_Success]
        );
        assert_eq!(state.message_counts, vec![0]);
        assert!(state.error_messages.is_empty());
        yawgpu::wgpuShaderModuleRelease(module);
    }
}

#[test]
fn wait_any_only_compilation_info_waits_for_wait_any() {
    let test = ValidationTest::new();
    unsafe {
        let module = create_wgsl_module(&test, "@compute @workgroup_size(1) fn main() {}");
        let mut state = CompilationInfoState::default();
        let future = get_compilation_info(module, native::WGPUCallbackMode_WaitAnyOnly, &mut state);

        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.calls, 0);

        let mut wait_info = native::WGPUFutureWaitInfo {
            future,
            completed: 0,
        };
        assert_eq!(
            yawgpu::wgpuInstanceWaitAny(test.instance(), 1, &mut wait_info, 0),
            native::WGPUWaitStatus_Success
        );
        assert_eq!(wait_info.completed, 1);
        assert_eq!(state.calls, 1);

        yawgpu::wgpuInstanceProcessEvents(test.instance());
        assert_eq!(state.calls, 1);
        yawgpu::wgpuShaderModuleRelease(module);
    }
}

unsafe fn assert_wgsl_ok(test: &ValidationTest, source: &str) {
    test.clear_errors();
    let module = create_wgsl_module(test, source);
    assert!(!module.is_null());
    assert!(test.errors().is_empty());
    yawgpu::wgpuShaderModuleRelease(module);
}

unsafe fn assert_wgsl_error(test: &ValidationTest, source: &str) {
    let mut module = std::ptr::null();
    assert_device_error!({
        module = create_wgsl_module(test, source);
    });
    assert!(!module.is_null());
    yawgpu::wgpuShaderModuleRelease(module);
}

unsafe fn create_wgsl_module(test: &ValidationTest, source: &str) -> native::WGPUShaderModule {
    let mut wgsl = wgsl_source(source);
    let descriptor = shader_module_descriptor((&mut wgsl.chain) as *mut _);
    yawgpu::wgpuDeviceCreateShaderModule(test.device(), &descriptor)
}

fn shader_module_descriptor(
    next_in_chain: *mut native::WGPUChainedStruct,
) -> native::WGPUShaderModuleDescriptor {
    native::WGPUShaderModuleDescriptor {
        nextInChain: next_in_chain,
        label: native::WGPUStringView {
            data: std::ptr::null(),
            length: 0,
        },
    }
}

fn wgsl_source(source: &str) -> native::WGPUShaderSourceWGSL {
    native::WGPUShaderSourceWGSL {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_ShaderSourceWGSL,
        },
        code: native::WGPUStringView {
            data: source.as_ptr().cast(),
            length: source.len(),
        },
    }
}

fn spirv_source(words: &[u32]) -> native::WGPUShaderSourceSPIRV {
    native::WGPUShaderSourceSPIRV {
        chain: native::WGPUChainedStruct {
            next: std::ptr::null_mut(),
            sType: native::WGPUSType_ShaderSourceSPIRV,
        },
        codeSize: words.len() as u32,
        code: words.as_ptr(),
    }
}

unsafe fn get_compilation_info(
    module: native::WGPUShaderModule,
    mode: native::WGPUCallbackMode,
    state: &mut CompilationInfoState,
) -> native::WGPUFuture {
    let callback_info = native::WGPUCompilationInfoCallbackInfo {
        nextInChain: std::ptr::null_mut(),
        mode,
        callback: Some(compilation_info_callback),
        userdata1: (state as *mut CompilationInfoState).cast(),
        userdata2: std::ptr::null_mut(),
    };
    yawgpu::wgpuShaderModuleGetCompilationInfo(module, callback_info)
}

unsafe extern "C" fn compilation_info_callback(
    status: native::WGPUCompilationInfoRequestStatus,
    info: *const native::WGPUCompilationInfo,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    let state = &mut *(userdata1 as *mut CompilationInfoState);
    state.calls += 1;
    state.statuses.push(status);

    let info = info.as_ref().expect("compilation info must not be null");
    state.message_counts.push(info.messageCount);
    for index in 0..info.messageCount {
        let message = &*info.messages.add(index);
        if message.type_ == native::WGPUCompilationMessageType_Error {
            state
                .error_messages
                .push(string_view_to_string(message.message));
        }
    }
}

unsafe fn string_view_to_string(value: native::WGPUStringView) -> String {
    if value.data.is_null() {
        return String::new();
    }
    let bytes = std::slice::from_raw_parts(value.data.cast::<u8>(), value.length);
    String::from_utf8_lossy(bytes).into_owned()
}
