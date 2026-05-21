/// Stores extent metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent3d {
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Depth or array layers.
    pub depth_or_array_layers: u32,
}

/// Stores origin metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Origin3d {
    /// X.
    pub x: u32,
    /// Y.
    pub y: u32,
    /// Z.
    pub z: u32,
}
