use crate::{HalTextureFormat, HalTextureUsage};

/// Wraps  for the selected backend.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct HalSurfaceConfiguration {
    /// Format.
    pub format: HalTextureFormat,
    /// Usage.
    pub usage: HalTextureUsage,
    /// Width.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Composite alpha mode.
    pub alpha_mode: HalCompositeAlphaMode,
    /// Present mode.
    pub present_mode: HalPresentMode,
}

impl HalSurfaceConfiguration {
    /// Creates a new instance.
    #[must_use]
    pub fn new(
        format: HalTextureFormat,
        usage: HalTextureUsage,
        width: u32,
        height: u32,
        present_mode: HalPresentMode,
        alpha_mode: HalCompositeAlphaMode,
    ) -> Self {
        Self {
            format,
            usage,
            width,
            height,
            present_mode,
            alpha_mode,
        }
    }
}

/// Enumerates HAL present mode values.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalPresentMode {
    /// Fifo variant.
    Fifo,
    /// Fifo relaxed variant.
    FifoRelaxed,
    /// Immediate variant.
    Immediate,
    /// Mailbox variant.
    Mailbox,
}

/// Alpha compositing used by a presentation surface.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalCompositeAlphaMode {
    /// Ignore the image alpha channel.
    Opaque,
    /// Color channels are multiplied by alpha.
    Premultiplied,
    /// Color channels are independent of alpha.
    Unpremultiplied,
    /// Inherit the window system's compositing policy.
    Inherit,
}

/// Supported surface configurations, in backend preference order.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct HalSurfaceCapabilities {
    /// Supported image usages.
    pub usages: HalTextureUsage,
    /// Supported formats, preferred first.
    pub formats: Vec<HalTextureFormat>,
    /// Supported presentation modes.
    pub present_modes: Vec<HalPresentMode>,
    /// Supported alpha modes; the first resolves Auto.
    pub alpha_modes: Vec<HalCompositeAlphaMode>,
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
            transient: false,
        };
        let config = HalSurfaceConfiguration::new(
            HalTextureFormat::Rgba8Unorm,
            usage,
            320,
            240,
            HalPresentMode::Mailbox,
            crate::HalCompositeAlphaMode::Opaque,
        );

        assert!(matches!(config.format, HalTextureFormat::Rgba8Unorm));
        assert_eq!(config.usage.copy_src, usage.copy_src);
        assert_eq!(config.usage.copy_dst, usage.copy_dst);
        assert_eq!(config.usage.texture_binding, usage.texture_binding);
        assert_eq!(config.usage.storage_binding, usage.storage_binding);
        assert_eq!(config.usage.render_attachment, usage.render_attachment);
        assert_eq!(config.alpha_mode, HalCompositeAlphaMode::Opaque);
        assert_eq!(config.width, 320);
        assert_eq!(config.height, 240);
        assert!(matches!(config.present_mode, HalPresentMode::Mailbox));
    }
}
