use yawgpu::native;
use yawgpu_test::ValidationTest;

use crate::common::{
    assert_render_pipeline_descriptor, color_target, create_wgsl_module, depth_state,
    empty_string_view,
};

pub const VERTEX_NO_INPUT: &str =
    "@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }";

pub const FRAGMENT_COLOR: &str = "@fragment fn main() -> @location(0) vec4f { return vec4f(); }";

pub unsafe fn expect_render_pipeline(
    test: &ValidationTest,
    is_async: bool,
    success: bool,
    case: RenderPipelineCase<'_>,
) {
    let vertex_module = case
        .vertex_module
        .unwrap_or_else(|| unsafe { create_wgsl_module(test.device(), case.vertex_source) });
    let owned_vertex = case.vertex_module.is_none();
    let fragment_module = match (case.fragment_module, case.fragment_source) {
        (Some(module), _) => Some(module),
        (None, Some(source)) => Some(unsafe { create_wgsl_module(test.device(), source) }),
        (None, None) => None,
    };
    let owned_fragment = case.fragment_module.is_none() && case.fragment_source.is_some();
    let default_target = color_target();
    let targets = case
        .fragment_targets
        .unwrap_or(std::slice::from_ref(&default_target));
    let fragment_state = fragment_module.map(|module| native::WGPUFragmentState {
        nextInChain: std::ptr::null_mut(),
        module,
        entryPoint: empty_string_view(),
        constantCount: 0,
        constants: std::ptr::null(),
        targetCount: if case.fragment_has_target {
            targets.len()
        } else {
            0
        },
        targets: if case.fragment_has_target {
            targets.as_ptr()
        } else {
            std::ptr::null()
        },
    });
    let fragment_ptr = fragment_state
        .as_ref()
        .map_or(std::ptr::null(), |state| state as *const _);
    let default_depth = case.depth_stencil.then(depth_state);
    let depth_ptr = case.depth_stencil_state.map_or_else(
        || {
            default_depth
                .as_ref()
                .map_or(std::ptr::null(), |state| state)
        },
        |state| state as *const _,
    );
    let primitive = case.primitive.unwrap_or_else(default_primitive);
    let multisample = case.multisample.unwrap_or_else(default_multisample);
    let descriptor = native::WGPURenderPipelineDescriptor {
        nextInChain: std::ptr::null_mut(),
        label: empty_string_view(),
        layout: case.layout.unwrap_or(std::ptr::null()),
        vertex: native::WGPUVertexState {
            nextInChain: std::ptr::null_mut(),
            module: vertex_module,
            entryPoint: empty_string_view(),
            constantCount: 0,
            constants: std::ptr::null(),
            bufferCount: case.buffers.len(),
            buffers: case.buffers.as_ptr(),
        },
        primitive,
        depthStencil: depth_ptr,
        multisample,
        fragment: fragment_ptr,
    };
    unsafe {
        assert_render_pipeline_descriptor(test, is_async, success, &descriptor);
        if owned_fragment {
            yawgpu::wgpuShaderModuleRelease(fragment_module.expect("owned fragment module"));
        }
        if owned_vertex {
            yawgpu::wgpuShaderModuleRelease(vertex_module);
        }
    }
}

#[derive(Clone, Copy)]
pub struct RenderPipelineCase<'a> {
    pub vertex_source: &'a str,
    pub vertex_module: Option<native::WGPUShaderModule>,
    pub fragment_source: Option<&'a str>,
    pub fragment_module: Option<native::WGPUShaderModule>,
    pub fragment_has_target: bool,
    pub fragment_targets: Option<&'a [native::WGPUColorTargetState]>,
    pub buffers: &'a [native::WGPUVertexBufferLayout],
    pub primitive: Option<native::WGPUPrimitiveState>,
    pub multisample: Option<native::WGPUMultisampleState>,
    pub depth_stencil: bool,
    pub depth_stencil_state: Option<&'a native::WGPUDepthStencilState>,
    pub layout: Option<native::WGPUPipelineLayout>,
}

impl<'a> Default for RenderPipelineCase<'a> {
    fn default() -> Self {
        Self {
            vertex_source: VERTEX_NO_INPUT,
            vertex_module: None,
            fragment_source: Some(FRAGMENT_COLOR),
            fragment_module: None,
            fragment_has_target: true,
            fragment_targets: None,
            buffers: &[],
            primitive: None,
            multisample: None,
            depth_stencil: false,
            depth_stencil_state: None,
            layout: None,
        }
    }
}

pub fn color_target_with(
    format: native::WGPUTextureFormat,
    blend: Option<&native::WGPUBlendState>,
    write_mask: native::WGPUColorWriteMask,
) -> native::WGPUColorTargetState {
    native::WGPUColorTargetState {
        nextInChain: std::ptr::null_mut(),
        format,
        blend: blend.map_or(std::ptr::null(), |state| state as *const _),
        writeMask: write_mask,
    }
}

pub fn blend_state(
    color: native::WGPUBlendComponent,
    alpha: native::WGPUBlendComponent,
) -> native::WGPUBlendState {
    native::WGPUBlendState { color, alpha }
}

pub fn blend_component(
    operation: native::WGPUBlendOperation,
    src_factor: native::WGPUBlendFactor,
    dst_factor: native::WGPUBlendFactor,
) -> native::WGPUBlendComponent {
    native::WGPUBlendComponent {
        operation,
        srcFactor: src_factor,
        dstFactor: dst_factor,
    }
}

pub fn default_blend_component() -> native::WGPUBlendComponent {
    blend_component(
        native::WGPUBlendOperation_Add,
        native::WGPUBlendFactor_One,
        native::WGPUBlendFactor_Zero,
    )
}

pub fn depth_stencil_state(
    format: native::WGPUTextureFormat,
    depth_write_enabled: native::WGPUOptionalBool,
    depth_compare: native::WGPUCompareFunction,
) -> native::WGPUDepthStencilState {
    native::WGPUDepthStencilState {
        nextInChain: std::ptr::null_mut(),
        format,
        depthWriteEnabled: depth_write_enabled,
        depthCompare: depth_compare,
        stencilFront: default_stencil_face(),
        stencilBack: default_stencil_face(),
        stencilReadMask: 0xFFFF_FFFF,
        stencilWriteMask: 0xFFFF_FFFF,
        depthBias: 0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
    }
}

pub fn default_stencil_face() -> native::WGPUStencilFaceState {
    native::WGPUStencilFaceState {
        compare: native::WGPUCompareFunction_Undefined,
        failOp: native::WGPUStencilOperation_Undefined,
        depthFailOp: native::WGPUStencilOperation_Undefined,
        passOp: native::WGPUStencilOperation_Undefined,
    }
}

pub fn vertex_buffer(
    array_stride: u64,
    attributes: &[native::WGPUVertexAttribute],
) -> native::WGPUVertexBufferLayout {
    native::WGPUVertexBufferLayout {
        nextInChain: std::ptr::null_mut(),
        stepMode: native::WGPUVertexStepMode_Vertex,
        arrayStride: array_stride,
        attributeCount: attributes.len(),
        attributes: attributes.as_ptr(),
    }
}

pub fn empty_vertex_buffer() -> native::WGPUVertexBufferLayout {
    native::WGPUVertexBufferLayout {
        nextInChain: std::ptr::null_mut(),
        stepMode: native::WGPUVertexStepMode_Undefined,
        arrayStride: 0,
        attributeCount: 0,
        attributes: std::ptr::null(),
    }
}

pub fn vertex_attribute(
    format: native::WGPUVertexFormat,
    offset: u64,
    shader_location: u32,
) -> native::WGPUVertexAttribute {
    native::WGPUVertexAttribute {
        nextInChain: std::ptr::null_mut(),
        format,
        offset,
        shaderLocation: shader_location,
    }
}

pub fn default_primitive() -> native::WGPUPrimitiveState {
    native::WGPUPrimitiveState {
        nextInChain: std::ptr::null_mut(),
        topology: native::WGPUPrimitiveTopology_TriangleList,
        stripIndexFormat: native::WGPUIndexFormat_Undefined,
        frontFace: native::WGPUFrontFace_Undefined,
        cullMode: native::WGPUCullMode_Undefined,
        unclippedDepth: 0,
    }
}

pub fn default_multisample() -> native::WGPUMultisampleState {
    native::WGPUMultisampleState {
        nextInChain: std::ptr::null_mut(),
        count: 1,
        mask: 0xFFFF_FFFF,
        alphaToCoverageEnabled: 0,
    }
}

pub fn vertex_input_shader(location: u32, ty: &str) -> String {
    format!(
        "@vertex fn main(@location({location}) a: {ty}) -> @builtin(position) vec4f {{
            _ = a;
            return vec4f();
        }}"
    )
}

pub fn inter_stage_vertex(outputs: &[&str]) -> String {
    let fields = outputs
        .iter()
        .enumerate()
        .map(|(i, output)| output.replace("__", &format!("v{i}")))
        .collect::<Vec<_>>()
        .join(",\n");
    let assigns = outputs
        .iter()
        .enumerate()
        .map(|(i, output)| {
            let value = if output.contains("u32") {
                "1u"
            } else if output.contains("i32") {
                "1"
            } else if output.contains("vec2") {
                "vec2f()"
            } else if output.contains("vec3") {
                "vec3f()"
            } else if output.contains("vec4") {
                "vec4f()"
            } else {
                "1.0"
            };
            format!("out.v{i} = {value};")
        })
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "struct VertexOut {{
            {fields},
            @builtin(position) pos: vec4f,
        }}
        @vertex fn main() -> VertexOut {{
            var out: VertexOut;
            out.pos = vec4f();
            {assigns}
            return out;
        }}"
    )
}

pub fn inter_stage_fragment(inputs: &[&str]) -> String {
    let fields = inputs
        .iter()
        .enumerate()
        .map(|(i, input)| input.replace("__", &format!("v{i}")))
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        "struct FragmentIn {{
            {fields},
        }}
        @fragment fn main(input: FragmentIn) -> @location(0) vec4f {{
            _ = input;
            return vec4f();
        }}"
    )
}
