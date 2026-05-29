#![allow(dead_code)]

#[path = "cts/validation/pipeline_common.rs"]
mod common;

#[path = "cts/validation/render_pipeline/common.rs"]
mod render_common;

#[path = "cts/validation/render_pipeline/vertex_state.rs"]
mod vertex_state;

#[path = "cts/validation/render_pipeline/inter_stage.rs"]
mod inter_stage;

#[path = "cts/validation/render_pipeline/primitive_state.rs"]
mod primitive_state;

#[path = "cts/validation/render_pipeline/multisample_state.rs"]
mod multisample_state;

#[path = "cts/validation/render_pipeline/misc.rs"]
mod misc;

#[path = "cts/validation/render_pipeline/shader_module.rs"]
mod shader_module;
