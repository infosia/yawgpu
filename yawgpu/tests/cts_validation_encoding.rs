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
