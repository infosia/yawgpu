use super::*;

pub struct VulkanSurface {
    pub(super) instance: Arc<VulkanInstanceInner>,
    pub(super) surface: vk::SurfaceKHR,
    pub(super) swapchain: Option<Arc<VulkanSwapchainInner>>,
    pub(super) config: Option<HalSurfaceConfiguration>,
    pub(super) current_image_index: Option<u32>,
}

impl fmt::Debug for VulkanSurface {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanSurface")
            .field("surface", &self.surface)
            .field("configured", &self.config.is_some())
            .finish()
    }
}

unsafe impl Send for VulkanSurface {}
unsafe impl Sync for VulkanSurface {}

impl Drop for VulkanSurface {
    fn drop(&mut self) {
        self.swapchain = None;
        let loader =
            ash::khr::surface::Instance::new(&self.instance._entry, &self.instance.instance);
        unsafe {
            loader.destroy_surface(self.surface, None);
        }
    }
}

impl VulkanSurface {
    pub fn configure(
        &mut self,
        device: &VulkanDevice,
        config: HalSurfaceConfiguration,
    ) -> Result<(), HalError> {
        if self.surface == vk::SurfaceKHR::null() {
            return Err(HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "surface is null",
            });
        }
        self.swapchain = None;
        let swapchain = create_swapchain(Arc::clone(&device.inner), self.surface, config)?;
        self.config = Some(config);
        self.current_image_index = None;
        self.swapchain = Some(swapchain);
        Ok(())
    }

    pub fn unconfigure(&mut self) {
        self.swapchain = None;
        self.config = None;
        self.current_image_index = None;
    }

    pub fn acquire_next_texture(&mut self) -> Result<VulkanTexture, HalError> {
        let swapchain = self.swapchain.as_ref().ok_or(HalError::AcquireFailed {
            backend: BACKEND,
            message: "surface is not configured",
        })?;
        let fence_info = vk::FenceCreateInfo::default();
        let fence =
            unsafe { swapchain.device.device.create_fence(&fence_info, None) }.map_err(|_| {
                HalError::AcquireFailed {
                    backend: BACKEND,
                    message: "fence creation failed",
                }
            })?;
        let acquire = unsafe {
            swapchain.loader.acquire_next_image(
                swapchain.swapchain,
                u64::MAX,
                vk::Semaphore::null(),
                fence,
            )
        };
        let image_index = match acquire {
            Ok((image_index, _suboptimal)) => image_index,
            Err(_) => {
                unsafe {
                    swapchain.device.device.destroy_fence(fence, None);
                }
                return Err(HalError::AcquireFailed {
                    backend: BACKEND,
                    message: "vkAcquireNextImageKHR failed",
                });
            }
        };
        let wait = unsafe {
            swapchain
                .device
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
        };
        unsafe {
            swapchain.device.device.destroy_fence(fence, None);
        }
        wait.map_err(|_| HalError::AcquireFailed {
            backend: BACKEND,
            message: "waiting for acquired image failed",
        })?;
        self.current_image_index = Some(image_index);
        let mut texture = swapchain
            .images
            .get(usize::try_from(image_index).unwrap_or(usize::MAX))
            .cloned()
            .ok_or(HalError::AcquireFailed {
                backend: BACKEND,
                message: "acquired image index is out of range",
            })?;
        texture.swapchain = Some(Arc::clone(swapchain));
        Ok(texture)
    }

    pub fn present(&mut self, queue: &VulkanQueue) -> Result<(), HalError> {
        let image_index = self
            .current_image_index
            .take()
            .ok_or(HalError::PresentFailed {
                backend: BACKEND,
                message: "no acquired image to present",
            })?;
        let swapchain = self.swapchain.as_ref().ok_or(HalError::PresentFailed {
            backend: BACKEND,
            message: "surface is not configured",
        })?;
        let texture = swapchain
            .images
            .get(usize::try_from(image_index).unwrap_or(usize::MAX))
            .ok_or(HalError::PresentFailed {
                backend: BACKEND,
                message: "acquired image index is out of range",
            })?;
        transition_swapchain_image_to_present(queue, texture)?;
        let swapchains = [swapchain.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        unsafe {
            swapchain
                .loader
                .queue_present(queue.inner.queue, &present_info)
        }
        .map_err(|_| HalError::PresentFailed {
            backend: BACKEND,
            message: "vkQueuePresentKHR failed",
        })?;
        unsafe { queue.inner.device.device.queue_wait_idle(queue.inner.queue) }.map_err(|_| {
            HalError::PresentFailed {
                backend: BACKEND,
                message: "queue wait after present failed",
            }
        })?;
        Ok(())
    }
}

pub(super) struct VulkanSwapchainInner {
    pub(super) device: Arc<VulkanDeviceInner>,
    pub(super) loader: ash::khr::swapchain::Device,
    pub(super) swapchain: vk::SwapchainKHR,
    pub(super) images: Vec<VulkanTexture>,
}

impl fmt::Debug for VulkanSwapchainInner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VulkanSwapchainInner")
            .field("swapchain", &self.swapchain)
            .field("image_count", &self.images.len())
            .finish()
    }
}

impl Drop for VulkanSwapchainInner {
    fn drop(&mut self) {
        self.images.clear();
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

pub(super) fn create_swapchain(
    device: Arc<VulkanDeviceInner>,
    surface: vk::SurfaceKHR,
    config: HalSurfaceConfiguration,
) -> Result<Arc<VulkanSwapchainInner>, HalError> {
    let (format, bytes_per_pixel) = map_texture_format(config.format)?;
    let surface_loader =
        ash::khr::surface::Instance::new(&device._instance._entry, &device._instance.instance);
    let capabilities = unsafe {
        surface_loader.get_physical_device_surface_capabilities(device.physical_device, surface)
    }
    .map_err(|_| HalError::SwapchainCreationFailed {
        backend: BACKEND,
        message: "surface capabilities query failed",
    })?;
    let mut image_count = capabilities.min_image_count.saturating_add(1).max(2);
    if capabilities.max_image_count > 0 {
        image_count = image_count.min(capabilities.max_image_count);
    }
    let extent = if capabilities.current_extent.width == u32::MAX {
        vk::Extent2D {
            width: config.width,
            height: config.height,
        }
    } else {
        capabilities.current_extent
    };
    let present_mode = match config.present_mode {
        crate::HalPresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
        crate::HalPresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        crate::HalPresentMode::Fifo => vk::PresentModeKHR::FIFO,
    };
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;
    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format)
        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(usage)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true);
    let loader = ash::khr::swapchain::Device::new(&device._instance.instance, &device.device);
    let swapchain = unsafe { loader.create_swapchain(&create_info, None) }.map_err(|_| {
        HalError::SwapchainCreationFailed {
            backend: BACKEND,
            message: "vkCreateSwapchainKHR failed",
        }
    })?;
    let images = unsafe { loader.get_swapchain_images(swapchain) }.map_err(|_| {
        unsafe {
            loader.destroy_swapchain(swapchain, None);
        }
        HalError::SwapchainCreationFailed {
            backend: BACKEND,
            message: "vkGetSwapchainImagesKHR failed",
        }
    })?;
    let textures = images
        .into_iter()
        .map(|image| {
            create_swapchain_texture(
                Arc::clone(&device),
                image,
                format,
                config.format,
                extent,
                bytes_per_pixel,
            )
        })
        .collect::<Result<Vec<_>, HalError>>()
        .inspect_err(|_| unsafe {
            loader.destroy_swapchain(swapchain, None);
        })?;
    Ok(Arc::new(VulkanSwapchainInner {
        device,
        loader,
        swapchain,
        images: textures,
    }))
}

pub(super) fn create_swapchain_texture(
    device: Arc<VulkanDeviceInner>,
    image: vk::Image,
    vk_format: vk::Format,
    format: HalTextureFormat,
    extent: vk::Extent2D,
    bytes_per_pixel: u32,
) -> Result<VulkanTexture, HalError> {
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(vk_format)
        .subresource_range(color_subresource_range());
    let view = unsafe { device.device.create_image_view(&view_info, None) }.map_err(|_| {
        HalError::SwapchainCreationFailed {
            backend: BACKEND,
            message: "swapchain image view creation failed",
        }
    })?;
    Ok(VulkanTexture {
        inner: Some(Arc::new(VulkanTextureInner {
            device,
            image,
            view,
            memory: None,
            owns_image: false,
            layout: AtomicU8::new(IMAGE_LAYOUT_UNDEFINED),
        })),
        swapchain: None,
        width: extent.width,
        height: extent.height,
        depth_or_array_layers: 1,
        bytes_per_pixel,
        format,
    })
}

#[cfg(test)]
mod tests {
    use super::super::test_helpers::*;
    use super::*;

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_surface_configure_errors_for_null_surface() {
        let instance = VulkanInstance::new().expect("create Vulkan instance");
        let device = vulkan_device();
        let mut surface = dummy_surface(&instance);
        let error = surface
            .configure(&device, surface_config())
            .expect_err("null surface must fail");
        assert!(matches!(
            error,
            HalError::SwapchainCreationFailed {
                backend: "vulkan",
                message: "surface is null"
            }
        ));
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_surface_unconfigure_is_idempotent() {
        let instance = VulkanInstance::new().expect("create Vulkan instance");
        let mut surface = dummy_surface(&instance);
        surface.unconfigure();
        surface.unconfigure();
        assert!(surface.config.is_none());
        assert!(surface.swapchain.is_none());
        assert!(surface.current_image_index.is_none());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_surface_acquire_next_texture_errors_when_unconfigured() {
        let instance = VulkanInstance::new().expect("create Vulkan instance");
        let mut surface = dummy_surface(&instance);
        let error = surface
            .acquire_next_texture()
            .expect_err("unconfigured surface must fail");
        assert!(matches!(
            error,
            HalError::AcquireFailed {
                backend: "vulkan",
                message: "surface is not configured"
            }
        ));
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_surface_present_errors_without_acquired_image() {
        let instance = VulkanInstance::new().expect("create Vulkan instance");
        let device = vulkan_device();
        let mut surface = dummy_surface(&instance);
        let error = surface
            .present(device.queue())
            .expect_err("surface without image must fail");
        assert!(matches!(
            error,
            HalError::PresentFailed {
                backend: "vulkan",
                message: "no acquired image to present"
            }
        ));
    }
}
