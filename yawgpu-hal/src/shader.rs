#[derive(Debug, Clone)]
pub enum HalShaderSource {
    Msl(String),
    SpirV(Vec<u32>),
    SpirVStages {
        vertex: Vec<u32>,
        fragment: Vec<u32>,
    },
}
