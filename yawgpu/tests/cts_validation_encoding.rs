#![allow(dead_code)]

#[path = "cts/validation/encoding/common.rs"]
mod common;

#[path = "cts/validation/encoding/encoder_state.rs"]
mod encoder_state;

#[path = "cts/validation/encoding/encoder_open_state.rs"]
mod encoder_open_state;

#[path = "cts/validation/encoding/begin_render_pass.rs"]
mod begin_render_pass;

#[path = "cts/validation/encoding/begin_compute_pass.rs"]
mod begin_compute_pass;

#[path = "cts/validation/encoding/cmds/clear_buffer.rs"]
mod clear_buffer;

#[path = "cts/validation/encoding/cmds/copy_buffer_to_buffer.rs"]
mod copy_buffer_to_buffer;

#[path = "cts/validation/encoding/cmds/copy_texture_to_texture.rs"]
mod copy_texture_to_texture;

#[path = "cts/validation/encoding/cmds/render/draw.rs"]
mod render_draw;

#[path = "cts/validation/encoding/cmds/render/set_vertex_buffer.rs"]
mod render_set_vertex_buffer;

#[path = "cts/validation/encoding/cmds/render/set_index_buffer.rs"]
mod render_set_index_buffer;

#[path = "cts/validation/encoding/cmds/render/set_pipeline.rs"]
mod render_set_pipeline;

#[path = "cts/validation/encoding/cmds/render/state_tracking.rs"]
mod render_state_tracking;

#[path = "cts/validation/encoding/cmds/index_access.rs"]
mod index_access;

#[path = "cts/validation/encoding/cmds/render/dynamic_state.rs"]
mod render_dynamic_state;

#[path = "cts/validation/encoding/cmds/render/indirect_draw.rs"]
mod render_indirect_draw;

#[path = "cts/validation/encoding/cmds/render/indirect_multi_draw.rs"]
mod render_indirect_multi_draw;

#[path = "cts/validation/encoding/cmds/compute_pass.rs"]
mod compute_pass_cmds;

#[path = "cts/validation/encoding/cmds/set_bind_group.rs"]
mod set_bind_group;

#[path = "cts/validation/encoding/cmds/set_immediates.rs"]
mod set_immediates;

#[path = "cts/validation/encoding/cmds/debug.rs"]
mod debug_cmds;

#[path = "cts/validation/dispatch.rs"]
mod dispatch;

#[path = "cts/validation/debug_marker.rs"]
mod debug_marker;
