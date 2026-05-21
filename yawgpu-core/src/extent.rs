#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent3d {
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Origin3d {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
