use crate::{HalTextureFormat, HalTextureUsage};

#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct HalSurfaceConfiguration {
    pub format: HalTextureFormat,
    pub usage: HalTextureUsage,
    pub width: u32,
    pub height: u32,
    pub present_mode: HalPresentMode,
}

impl HalSurfaceConfiguration {
    #[must_use]
    pub fn new(
        format: HalTextureFormat,
        usage: HalTextureUsage,
        width: u32,
        height: u32,
        present_mode: HalPresentMode,
    ) -> Self {
        Self {
            format,
            usage,
            width,
            height,
            present_mode,
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum HalPresentMode {
    Fifo,
    Immediate,
    Mailbox,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hal_surface_configuration_new_round_trips_fields() {
        let usage = HalTextureUsage {
            copy_src: true,
            copy_dst: false,
            texture_binding: true,
            storage_binding: false,
            render_attachment: true,
        };
        let config = HalSurfaceConfiguration::new(
            HalTextureFormat::Rgba8Unorm,
            usage,
            320,
            240,
            HalPresentMode::Mailbox,
        );

        assert!(matches!(config.format, HalTextureFormat::Rgba8Unorm));
        assert_eq!(config.usage.copy_src, usage.copy_src);
        assert_eq!(config.usage.copy_dst, usage.copy_dst);
        assert_eq!(config.usage.texture_binding, usage.texture_binding);
        assert_eq!(config.usage.storage_binding, usage.storage_binding);
        assert_eq!(config.usage.render_attachment, usage.render_attachment);
        assert_eq!(config.width, 320);
        assert_eq!(config.height, 240);
        assert!(matches!(config.present_mode, HalPresentMode::Mailbox));
    }
}
