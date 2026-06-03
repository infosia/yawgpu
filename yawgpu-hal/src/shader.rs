/// Enumerates HAL shader source values.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HalShaderSource {
    /// Msl variant.
    Msl(String),
    /// Per-stage MSL render sources.
    MslStages {
        /// Vertex stage MSL source.
        vertex: String,
        /// Optional fragment stage MSL source.
        fragment: Option<String>,
    },
    /// Spir v variant.
    SpirV(Vec<u32>),
    /// Spir vstages variant.
    SpirVStages {
        /// Vertex variant.
        vertex: Vec<u32>,
        /// Optional fragment variant.
        fragment: Option<Vec<u32>>,
    },
    /// GLSL render stages.
    GlslStages {
        /// Vertex stage GLSL ES source.
        vertex: String,
        /// Optional fragment stage GLSL ES source.
        fragment: Option<String>,
    },
    /// Glsl variant.
    Glsl {
        /// Source variant.
        source: String,
        /// Stage variant.
        stage: HalShaderStage,
    },
}

/// Enumerates shader stages for stage-specific source formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HalShaderStage {
    /// Vertex variant.
    Vertex,
    /// Fragment variant.
    Fragment,
    /// Compute variant.
    Compute,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hal_shader_source_msl_stages_round_trips_sources() {
        let source = HalShaderSource::MslStages {
            vertex: "vertex".to_owned(),
            fragment: Some("fragment".to_owned()),
        };

        assert!(matches!(
            source,
            HalShaderSource::MslStages { vertex, fragment }
                if vertex == "vertex" && fragment.as_deref() == Some("fragment")
        ));
    }
}
