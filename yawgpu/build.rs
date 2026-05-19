use std::env;
use std::path::PathBuf;

const OBJECT_HANDLES: &[&str] = &[
    "Adapter",
    "BindGroup",
    "BindGroupLayout",
    "Buffer",
    "CommandBuffer",
    "CommandEncoder",
    "ComputePassEncoder",
    "ComputePipeline",
    "Device",
    "Instance",
    "PipelineLayout",
    "QuerySet",
    "Queue",
    "RenderBundle",
    "RenderBundleEncoder",
    "RenderPassEncoder",
    "RenderPipeline",
    "Sampler",
    "ShaderModule",
    "Surface",
    "Texture",
    "TextureView",
];

fn main() {
    let header = "ffi/webgpu-headers/webgpu.h";
    println!("cargo:rerun-if-changed={header}");

    let mut builder = bindgen::Builder::default()
        .header(header)
        .allowlist_item("WGPU.*")
        .allowlist_item("wgpu.*")
        .prepend_enum_name(false)
        .size_t_is_usize(true)
        .ignore_functions();

    for handle in OBJECT_HANDLES {
        let wgpu_name = format!("WGPU{handle}");
        builder = builder.blocklist_type(&wgpu_name).raw_line(format!(
            "pub type {wgpu_name} = *const crate::{wgpu_name}Impl;"
        ));
    }

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR is set by Cargo"));
    builder
        .generate()
        .expect("failed to generate WebGPU bindings")
        .write_to_file(out_path.join("bindings.rs"))
        .expect("failed to write WebGPU bindings");
}
