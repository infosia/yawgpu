use super::*;
use crate::{HalCompositeAlphaMode, HalPresentMode, HalSurfaceCapabilities, HalTextureDimension};
use std::sync::atomic::AtomicBool;

pub(super) const RETIRE_RING_SIZE: usize = 3;

#[derive(Debug)]
pub(super) struct VulkanSurfaceInner {
    pub(super) instance: Arc<VulkanInstanceInner>,
    pub(super) surface: vk::SurfaceKHR,
}

impl VulkanSurfaceInner {
    pub(super) fn new(instance: Arc<VulkanInstanceInner>, surface: vk::SurfaceKHR) -> Self {
        Self { instance, surface }
    }
}

impl Drop for VulkanSurfaceInner {
    fn drop(&mut self) {
        if self.surface == vk::SurfaceKHR::null() {
            return;
        }
        let loader =
            ash::khr::surface::Instance::new(self.instance._entry, &self.instance.instance);
        unsafe {
            loader.destroy_surface(self.surface, None);
        }
    }
}

#[derive(Debug)]
pub(super) struct SurfacePendingState {
    pub(super) pending_acquire: Option<PendingAcquire>,
    pub(super) retire: RetireRing,
    pub(super) transition_command_pool: Option<vk::CommandPool>,
}

impl SurfacePendingState {
    pub(super) fn new() -> Self {
        Self {
            pending_acquire: None,
            retire: RetireRing::new(RETIRE_RING_SIZE),
            transition_command_pool: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PendingAcquire {
    pub(super) image_index: u32,
    pub(super) acquired_sem: vk::Semaphore,
    pub(super) render_finished_sem: vk::Semaphore,
    pub(super) present_ready_sem: vk::Semaphore,
    pub(super) in_flight_fence: vk::Fence,
    pub(super) consumed: bool,
}

#[derive(Debug)]
pub(super) enum RetireOp {
    DescriptorPool(vk::DescriptorPool),
    RenderPass(vk::RenderPass),
    Framebuffer(vk::Framebuffer),
    ImageView(vk::ImageView),
    CommandPool(vk::CommandPool),
    CommandBuffer {
        pool: vk::CommandPool,
        buffer: vk::CommandBuffer,
    },
}

#[derive(Debug)]
pub(super) enum RetainedResource {
    Buffer {
        _inner: Arc<VulkanBufferInner>,
    },
    Texture {
        _inner: Arc<VulkanTextureInner>,
    },
    Sampler {
        _inner: Arc<VulkanSamplerInner>,
    },
    QuerySet {
        _inner: Arc<VulkanQuerySetInner>,
    },
    ComputePipeline {
        _inner: Arc<VulkanComputePipelineInner>,
    },
    RenderPipeline {
        _inner: Arc<VulkanRenderPipelineInner>,
    },
}

#[derive(Debug)]
pub(super) struct InFlightFrame {
    fence: vk::Fence,
    cleanup: Vec<RetireOp>,
    retained: Vec<RetainedResource>,
    destroy_fence: bool,
    tracked_submission: Option<TrackedSubmission>,
}

#[derive(Debug)]
pub(super) struct RetireRing {
    frames: Vec<Option<InFlightFrame>>,
    overflow: Vec<InFlightFrame>,
    next: usize,
}

impl RetireRing {
    pub(super) fn new(size: usize) -> Self {
        Self {
            frames: (0..size).map(|_| None).collect(),
            overflow: Vec::new(),
            next: 0,
        }
    }

    pub(super) fn retire(
        &mut self,
        device: &ash::Device,
        fence: vk::Fence,
        cleanup: Vec<RetireOp>,
        retained: Vec<RetainedResource>,
        destroy_fence: bool,
    ) -> Result<(), HalError> {
        self.retire_tracked(device, fence, cleanup, retained, destroy_fence, None)
    }

    pub(super) fn retire_tracked(
        &mut self,
        device: &ash::Device,
        fence: vk::Fence,
        cleanup: Vec<RetireOp>,
        retained: Vec<RetainedResource>,
        destroy_fence: bool,
        tracked_submission: Option<TrackedSubmission>,
    ) -> Result<(), HalError> {
        self.drain_ready_overflow(device);
        let new_frame = InFlightFrame {
            fence,
            cleanup,
            retained,
            destroy_fence,
            tracked_submission,
        };
        if self.frames.is_empty() {
            if let Err(error) = wait_for_frame(device, &new_frame) {
                self.overflow.push(new_frame);
                return Err(queue_submission_error("vkWaitForFences", error));
            }
            cleanup_frame(device, new_frame);
            return Ok(());
        }

        if let Some(frame) = self.frames[self.next].as_ref() {
            if let Err(error) = wait_for_frame(device, frame) {
                self.overflow.push(new_frame);
                return Err(queue_submission_error("vkWaitForFences", error));
            }
        }
        if let Some(frame) = self.frames[self.next].take() {
            cleanup_frame(device, frame);
        }
        self.frames[self.next] = Some(new_frame);
        self.next = (self.next + 1) % self.frames.len();
        Ok(())
    }

    fn drain_ready_overflow(&mut self, device: &ash::Device) {
        let parked = std::mem::take(&mut self.overflow);
        for frame in parked {
            match unsafe { device.get_fence_status(frame.fence) } {
                Ok(true) => cleanup_frame(device, frame),
                Ok(false) | Err(_) => self.overflow.push(frame),
            }
        }
    }

    pub(super) fn wait_all(&mut self, device: &ash::Device) -> Result<(), HalError> {
        let mut first_error = None;
        for slot in &mut self.frames {
            if let Some(frame) = slot.as_ref() {
                match wait_for_frame(device, frame) {
                    Ok(()) => {
                        if let Some(frame) = slot.take() {
                            cleanup_frame(device, frame);
                        }
                    }
                    Err(error) => {
                        if first_error.is_none() {
                            first_error = Some(error);
                        }
                    }
                }
            }
        }
        let parked = std::mem::take(&mut self.overflow);
        for frame in parked {
            match wait_for_frame(device, &frame) {
                Ok(()) => cleanup_frame(device, frame),
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                    self.overflow.push(frame);
                }
            }
        }

        if let Some(error) = first_error {
            Err(queue_submission_error("vkWaitForFences", error))
        } else {
            Ok(())
        }
    }
}

fn wait_for_frame(device: &ash::Device, frame: &InFlightFrame) -> Result<(), vk::Result> {
    unsafe { device.wait_for_fences(&[frame.fence], true, u64::MAX) }
}

fn cleanup_frame(device: &ash::Device, frame: InFlightFrame) {
    let InFlightFrame {
        fence,
        cleanup,
        retained,
        destroy_fence,
        tracked_submission,
    } = frame;
    unsafe {
        cleanup_retire_ops(device, cleanup);
        drop(retained);
        if let Some(tracked_submission) = tracked_submission {
            let mut tracker = tracked_submission
                .tracker
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            tracker.mark_completed(tracked_submission.index);
            let removed_fence = tracker.remove_fence(tracked_submission.index);
            debug_assert_eq!(removed_fence, Some(fence));
            // If every waiter unpinned first, retirement owns and destroys the
            // fence here. If retirement wins the race, it defers destruction;
            // the last waiter owns destruction after dropping its final pin.
            if destroy_fence && !tracker.defer_destroy_if_pinned(fence) {
                device.destroy_fence(fence, None);
            }
        } else if destroy_fence {
            device.destroy_fence(fence, None);
        }
    }
}

pub(super) unsafe fn cleanup_retire_ops(device: &ash::Device, cleanup: Vec<RetireOp>) {
    for op in cleanup {
        match op {
            RetireOp::DescriptorPool(pool) => device.destroy_descriptor_pool(pool, None),
            RetireOp::RenderPass(render_pass) => device.destroy_render_pass(render_pass, None),
            RetireOp::Framebuffer(framebuffer) => device.destroy_framebuffer(framebuffer, None),
            RetireOp::ImageView(view) => device.destroy_image_view(view, None),
            RetireOp::CommandPool(command_pool) => device.destroy_command_pool(command_pool, None),
            RetireOp::CommandBuffer { pool, buffer } => {
                device.free_command_buffers(pool, &[buffer]);
            }
        }
    }
}

/// Stores vulkan surface data used by validation and backend submission.
pub struct VulkanSurface {
    pub(super) surface: vk::SurfaceKHR,
    pub(super) surface_inner: Arc<VulkanSurfaceInner>,
    pub(super) swapchain: Option<Arc<VulkanSwapchainInner>>,
    pub(super) config: Option<HalSurfaceConfiguration>,
    pub(super) current_image_index: Option<u32>,
    pub(super) pending_state: Arc<Mutex<SurfacePendingState>>,
    pub(super) image_acquired_semaphores: Vec<vk::Semaphore>,
    pub(super) render_finished_semaphores: Vec<vk::Semaphore>,
    pub(super) present_ready_semaphores: Vec<vk::Semaphore>,
    pub(super) in_flight_fences: Vec<vk::Fence>,
    pub(super) next_sync_index: usize,
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
        self.wait_all_in_flight_frames();
        self.destroy_sync_objects();
        self.destroy_transition_command_pool();
        self.swapchain = None;
    }
}

impl VulkanSurface {
    /// Queries the driver for this surface's supported configurations.
    pub fn capabilities(
        &self,
        adapter: &VulkanAdapter,
    ) -> Result<HalSurfaceCapabilities, HalError> {
        if !Arc::ptr_eq(&self.surface_inner.instance, &adapter.instance)
            || self.surface == vk::SurfaceKHR::null()
        {
            return Err(surface_query_error());
        }
        let instance = &self.surface_inner.instance;
        let loader = ash::khr::surface::Instance::new(instance._entry, &instance.instance);
        let (caps, formats, modes) = unsafe {
            (
                loader
                    .get_physical_device_surface_capabilities(adapter.physical_device, self.surface)
                    .map_err(|_| surface_query_error())?,
                loader
                    .get_physical_device_surface_formats(adapter.physical_device, self.surface)
                    .map_err(|_| surface_query_error())?,
                loader
                    .get_physical_device_surface_present_modes(
                        adapter.physical_device,
                        self.surface,
                    )
                    .map_err(|_| surface_query_error())?,
            )
        };
        // A driver reports one `VkSurfaceFormatKHR` per (format, colour space)
        // pair (MoltenVK: every format times a dozen colour spaces), while the
        // WebGPU capability list is per format — keep the first occurrence so
        // the preferred entry stays first, as Dawn's list does.
        Ok(HalSurfaceCapabilities {
            usages: vk_surface_usages(caps.supported_usage_flags),
            formats: dedup_preserving_order(
                formats
                    .into_iter()
                    .filter_map(|f| vk_surface_format_to_hal(f.format)),
            ),
            present_modes: dedup_preserving_order(
                modes.into_iter().filter_map(vk_present_mode_to_hal),
            ),
            alpha_modes: vk_composite_alpha_modes(caps.supported_composite_alpha),
        })
    }

    /// Configures the surface's swapchain for the given format, size, and present mode.
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
        self.wait_all_in_flight_frames();
        self.destroy_sync_objects();
        self.destroy_transition_command_pool();
        self.swapchain = None;
        let swapchain = create_swapchain(
            Arc::clone(&device.inner),
            Arc::clone(&self.surface_inner),
            config,
            Arc::clone(&self.pending_state),
        )?;
        self.create_sync_objects(&device.inner.device, swapchain.images.len())?;
        self.config = Some(config);
        self.current_image_index = None;
        self.next_sync_index = 0;
        self.swapchain = Some(swapchain);
        Ok(())
    }

    /// Tears down the surface's swapchain.
    pub fn unconfigure(&mut self) {
        self.wait_all_in_flight_frames();
        self.destroy_sync_objects();
        self.destroy_transition_command_pool();
        self.swapchain = None;
        self.config = None;
        self.current_image_index = None;
        let mut state = self
            .pending_state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.pending_acquire = None;
    }

    /// Returns acquire next texture.
    pub fn acquire_next_texture(&mut self) -> Result<VulkanTexture, HalError> {
        let swapchain = self.swapchain.as_ref().ok_or(HalError::AcquireFailed {
            backend: BACKEND,
            message: "surface is not configured",
        })?;
        let device = &swapchain.device.device;
        let sync_index = self.next_sync_index;
        let acquired_sem =
            *self
                .image_acquired_semaphores
                .get(sync_index)
                .ok_or(HalError::AcquireFailed {
                    backend: BACKEND,
                    message: "surface synchronization is not configured",
                })?;
        let render_finished_sem = self.render_finished_semaphores[sync_index];
        let present_ready_sem = self.present_ready_semaphores[sync_index];
        let in_flight_fence = self.in_flight_fences[sync_index];
        unsafe {
            device
                .wait_for_fences(&[in_flight_fence], true, u64::MAX)
                .map_err(|_| HalError::AcquireFailed {
                    backend: BACKEND,
                    message: "waiting for image fence failed",
                })?;
        }
        let acquire = unsafe {
            swapchain.loader.acquire_next_image(
                swapchain.swapchain,
                u64::MAX,
                acquired_sem,
                vk::Fence::null(),
            )
        };
        let image_index = match acquire {
            Ok((image_index, _suboptimal)) => image_index,
            Err(_) => {
                return Err(HalError::AcquireFailed {
                    backend: BACKEND,
                    message: "vkAcquireNextImageKHR failed",
                });
            }
        };
        unsafe {
            device
                .reset_fences(&[in_flight_fence])
                .map_err(|_| HalError::AcquireFailed {
                    backend: BACKEND,
                    message: "resetting image fence failed",
                })?;
        }
        let index = usize::try_from(image_index).unwrap_or(usize::MAX);
        if !self.image_acquired_semaphores.is_empty() {
            self.next_sync_index = (sync_index + 1) % self.image_acquired_semaphores.len();
        }
        self.current_image_index = Some(image_index);
        self.pending_state
            .lock()
            .map_err(|_| HalError::AcquireFailed {
                backend: BACKEND,
                message: "surface pending state lock failed",
            })?
            .pending_acquire = Some(PendingAcquire {
            image_index,
            acquired_sem,
            render_finished_sem,
            present_ready_sem,
            in_flight_fence,
            consumed: false,
        });
        let mut texture = swapchain
            .images
            .get(index)
            .cloned()
            .ok_or(HalError::AcquireFailed {
                backend: BACKEND,
                message: "acquired image index is out of range",
            })?;
        texture.swapchain = Some(Arc::clone(swapchain));
        Ok(texture)
    }

    /// Submits a present operation using the per-acquire semaphore chain.
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
        let pending = self
            .pending_state
            .lock()
            .map_err(|_| HalError::PresentFailed {
                backend: BACKEND,
                message: "surface pending state lock failed",
            })?
            .pending_acquire
            .ok_or(HalError::PresentFailed {
                backend: BACKEND,
                message: "no acquired image to present",
            })?;
        if pending.image_index != image_index {
            return Err(HalError::PresentFailed {
                backend: BACKEND,
                message: "pending acquire image mismatch",
            });
        }
        let wait_semaphore = if pending.consumed {
            pending.render_finished_sem
        } else {
            pending.acquired_sem
        };
        transition_swapchain_image_to_present(
            queue,
            texture,
            Arc::clone(&self.pending_state),
            wait_semaphore,
            pending.present_ready_sem,
            pending.in_flight_fence,
        )?;
        let swapchains = [swapchain.swapchain];
        let image_indices = [image_index];
        let wait_semaphores = [pending.present_ready_sem];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        {
            let _queue_access = queue
                .inner
                .queue_access
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            unsafe {
                swapchain
                    .loader
                    .queue_present(queue.inner.queue, &present_info)
            }
        }
        .map_err(|_| HalError::PresentFailed {
            backend: BACKEND,
            message: "vkQueuePresentKHR failed",
        })?;
        self.pending_state
            .lock()
            .map_err(|_| HalError::PresentFailed {
                backend: BACKEND,
                message: "surface pending state lock failed",
            })?
            .pending_acquire = None;
        Ok(())
    }

    fn create_sync_objects(&mut self, device: &ash::Device, count: usize) -> Result<(), HalError> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        for _ in 0..count {
            let acquired =
                unsafe { device.create_semaphore(&semaphore_info, None) }.map_err(|_| {
                    HalError::SwapchainCreationFailed {
                        backend: BACKEND,
                        message: "image-acquired semaphore creation failed",
                    }
                })?;
            let render_finished = unsafe { device.create_semaphore(&semaphore_info, None) }
                .map_err(|_| {
                    unsafe {
                        device.destroy_semaphore(acquired, None);
                    }
                    HalError::SwapchainCreationFailed {
                        backend: BACKEND,
                        message: "render-finished semaphore creation failed",
                    }
                })?;
            let present_ready =
                unsafe { device.create_semaphore(&semaphore_info, None) }.map_err(|_| {
                    unsafe {
                        device.destroy_semaphore(render_finished, None);
                        device.destroy_semaphore(acquired, None);
                    }
                    HalError::SwapchainCreationFailed {
                        backend: BACKEND,
                        message: "present-ready semaphore creation failed",
                    }
                })?;
            let fence = unsafe { device.create_fence(&fence_info, None) }.map_err(|_| {
                unsafe {
                    device.destroy_semaphore(present_ready, None);
                    device.destroy_semaphore(render_finished, None);
                    device.destroy_semaphore(acquired, None);
                }
                HalError::SwapchainCreationFailed {
                    backend: BACKEND,
                    message: "in-flight fence creation failed",
                }
            })?;
            self.image_acquired_semaphores.push(acquired);
            self.render_finished_semaphores.push(render_finished);
            self.present_ready_semaphores.push(present_ready);
            self.in_flight_fences.push(fence);
        }
        Ok(())
    }

    fn destroy_sync_objects(&mut self) {
        let Some(swapchain) = self.swapchain.as_ref() else {
            self.image_acquired_semaphores.clear();
            self.render_finished_semaphores.clear();
            self.present_ready_semaphores.clear();
            self.in_flight_fences.clear();
            return;
        };
        unsafe {
            // Frame fences do not cover the subsequent present semaphore wait.
            // Device idle retires submissions using all three semaphore arrays
            // and the fences, plus the transition commands whose pool is freed next.
            // Teardown must still proceed if the device has been lost.
            wait_for_teardown_once(&swapchain.teardown_waited, || {
                let _ = swapchain.device.device.device_wait_idle();
            });
            for semaphore in self.image_acquired_semaphores.drain(..) {
                swapchain.device.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.render_finished_semaphores.drain(..) {
                swapchain.device.device.destroy_semaphore(semaphore, None);
            }
            for semaphore in self.present_ready_semaphores.drain(..) {
                swapchain.device.device.destroy_semaphore(semaphore, None);
            }
            for fence in self.in_flight_fences.drain(..) {
                swapchain.device.device.destroy_fence(fence, None);
            }
        }
    }

    pub(super) fn wait_all_in_flight_frames(&mut self) {
        if let Some(swapchain) = self.swapchain.as_ref() {
            let mut state = self
                .pending_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _ = state.retire.wait_all(&swapchain.device.device);
        }
    }

    fn destroy_transition_command_pool(&mut self) {
        if let Some(swapchain) = self.swapchain.as_ref() {
            let mut state = self
                .pending_state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(command_pool) = state.transition_command_pool.take() {
                unsafe {
                    swapchain
                        .device
                        .device
                        .destroy_command_pool(command_pool, None);
                }
            }
        }
    }
}

/// Holds shared state for the vulkan swapchain handle.
pub(super) struct VulkanSwapchainInner {
    pub(super) device: Arc<VulkanDeviceInner>,
    pub(super) _surface: Arc<VulkanSurfaceInner>,
    pub(super) loader: ash::khr::swapchain::Device,
    pub(super) swapchain: vk::SwapchainKHR,
    pub(super) images: Vec<VulkanTexture>,
    teardown_waited: AtomicBool,
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

/// Surface teardown and the final swapchain drop share one device-idle wait.
fn wait_for_teardown_once(waited: &AtomicBool, wait: impl FnOnce()) {
    if !waited.load(Ordering::Acquire) {
        wait();
        waited.store(true, Ordering::Release);
    }
}

impl Drop for VulkanSwapchainInner {
    fn drop(&mut self) {
        // Guard the image views and swapchain even when outstanding texture
        // handles defer this drop beyond the surface's synchronization teardown.
        unsafe {
            wait_for_teardown_once(&self.teardown_waited, || {
                let _ = self.device.device.device_wait_idle();
            });
        }
        self.images.clear();
        unsafe {
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}

/// Creates swapchain and reports validation errors through the owning device.
pub(super) fn create_swapchain(
    device: Arc<VulkanDeviceInner>,
    surface: Arc<VulkanSurfaceInner>,
    config: HalSurfaceConfiguration,
    pending_state: Arc<Mutex<SurfacePendingState>>,
) -> Result<Arc<VulkanSwapchainInner>, HalError> {
    let (format, bytes_per_pixel) = map_texture_format(config.format)?;
    let surface_loader =
        ash::khr::surface::Instance::new(device._instance._entry, &device._instance.instance);
    let capabilities = unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(device.physical_device, surface.surface)
    }
    .map_err(|_| HalError::SwapchainCreationFailed {
        backend: BACKEND,
        message: "surface capabilities query failed",
    })?;
    let mut image_count = capabilities.min_image_count.saturating_add(1).max(2);
    if capabilities.max_image_count > 0 {
        image_count = image_count.min(capabilities.max_image_count);
    }
    let extent = swapchain_extent(config.width, config.height, &capabilities)?;
    let present_modes = unsafe {
        surface_loader
            .get_physical_device_surface_present_modes(device.physical_device, surface.surface)
    }
    .unwrap_or_else(|_| vec![vk::PresentModeKHR::FIFO]);
    let present_mode = select_present_mode(config.present_mode, &present_modes);
    let usage = swapchain_image_usage(config.usage, capabilities.supported_usage_flags)?;
    let alpha = hal_alpha_to_vk(config.alpha_mode);
    if !capabilities.supported_composite_alpha.contains(alpha) {
        return Err(surface_query_error());
    }
    let formats = unsafe {
        surface_loader.get_physical_device_surface_formats(device.physical_device, surface.surface)
    }
    .map_err(|_| surface_query_error())?;
    let surface_format = formats
        .iter()
        .find(|entry| entry.format == format)
        .ok_or_else(surface_query_error)?;
    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface.surface)
        .min_image_count(image_count)
        .image_format(format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(usage)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(alpha)
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
                config,
                extent,
                (bytes_per_pixel, usage),
                Arc::clone(&pending_state),
            )
        })
        .collect::<Result<Vec<_>, HalError>>()
        .inspect_err(|_| unsafe {
            loader.destroy_swapchain(swapchain, None);
        })?;
    Ok(Arc::new(VulkanSwapchainInner {
        device,
        _surface: surface,
        loader,
        swapchain,
        images: textures,
        teardown_waited: AtomicBool::new(false),
    }))
}

/// Selects the fixed surface extent or bounds the configured extent to its limits.
fn swapchain_extent(
    config_w: u32,
    config_h: u32,
    caps: &vk::SurfaceCapabilitiesKHR,
) -> Result<vk::Extent2D, HalError> {
    let extent = if caps.current_extent.width == u32::MAX {
        vk::Extent2D {
            width: config_w
                .max(caps.min_image_extent.width)
                .min(caps.max_image_extent.width),
            height: config_h
                .max(caps.min_image_extent.height)
                .min(caps.max_image_extent.height),
        }
    } else {
        caps.current_extent
    };
    if extent.width == 0 || extent.height == 0 {
        return Err(HalError::SwapchainCreationFailed {
            backend: BACKEND,
            message: "surface reports a zero extent",
        });
    }
    Ok(extent)
}

fn select_present_mode(
    requested: crate::HalPresentMode,
    supported: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    let preferred = match requested {
        crate::HalPresentMode::Immediate => vk::PresentModeKHR::IMMEDIATE,
        crate::HalPresentMode::Mailbox => vk::PresentModeKHR::MAILBOX,
        crate::HalPresentMode::Fifo => vk::PresentModeKHR::FIFO,
        crate::HalPresentMode::FifoRelaxed => vk::PresentModeKHR::FIFO_RELAXED,
    };
    if supported.contains(&preferred) {
        preferred
    } else {
        vk::PresentModeKHR::FIFO
    }
}

/// Creates swapchain texture and reports validation errors through the owning device.
pub(super) fn create_swapchain_texture(
    device: Arc<VulkanDeviceInner>,
    image: vk::Image,
    vk_format: vk::Format,
    config: HalSurfaceConfiguration,
    extent: vk::Extent2D,
    pixel_info: (u32, vk::ImageUsageFlags),
    pending_state: Arc<Mutex<SurfacePendingState>>,
) -> Result<VulkanTexture, HalError> {
    let (bytes_per_pixel, usage) = pixel_info;
    let view = if super::texture::texture_usage_needs_view(config.usage) {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk_format)
            .subresource_range(color_subresource_range(1, 1));
        unsafe { device.device.create_image_view(&view_info, None) }.map_err(|_| {
            HalError::SwapchainCreationFailed {
                backend: BACKEND,
                message: "swapchain image view creation failed",
            }
        })?
    } else {
        vk::ImageView::null()
    };
    Ok(VulkanTexture {
        inner: Some(Arc::new(VulkanTextureInner {
            device,
            image,
            usage,
            view,
            bgra8_storage_view: vk::ImageView::null(),
            memory: None,
            owns_image: false,
            mip_level_count: 1,
            array_layers: 1,
            // Swapchain images are always color images.
            aspect_flags: vk::ImageAspectFlags::COLOR,
            layouts: SubresourceLayouts::new(1, 1),
        })),
        swapchain: None,
        surface_pending: Some(pending_state),
        dimension: HalTextureDimension::D2,
        width: extent.width,
        height: extent.height,
        depth_or_array_layers: 1,
        sample_count: 1,
        bytes_per_pixel,
        format: config.format,
        transient: false,
    })
}

/// Keeps the first occurrence of every value, in order (`PartialEq`, small lists).
fn dedup_preserving_order<T: PartialEq>(values: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut out = Vec::new();
    for value in values {
        if !out.contains(&value) {
            out.push(value);
        }
    }
    out
}

fn surface_query_error() -> HalError {
    HalError::SwapchainCreationFailed {
        backend: BACKEND,
        message: "surface capabilities query or configuration failed",
    }
}

fn vk_surface_format_to_hal(format: vk::Format) -> Option<HalTextureFormat> {
    Some(match format {
        vk::Format::R8G8B8A8_UNORM => HalTextureFormat::Rgba8Unorm,
        vk::Format::R8G8B8A8_SRGB => HalTextureFormat::Rgba8UnormSrgb,
        vk::Format::B8G8R8A8_UNORM => HalTextureFormat::Bgra8Unorm,
        vk::Format::B8G8R8A8_SRGB => HalTextureFormat::Bgra8UnormSrgb,
        vk::Format::A2B10G10R10_UNORM_PACK32 => HalTextureFormat::Rgb10a2Unorm,
        vk::Format::R16G16B16A16_SFLOAT => HalTextureFormat::Rgba16Float,
        _ => return None,
    })
}

fn vk_present_mode_to_hal(mode: vk::PresentModeKHR) -> Option<HalPresentMode> {
    Some(match mode {
        vk::PresentModeKHR::FIFO => HalPresentMode::Fifo,
        vk::PresentModeKHR::FIFO_RELAXED => HalPresentMode::FifoRelaxed,
        vk::PresentModeKHR::IMMEDIATE => HalPresentMode::Immediate,
        vk::PresentModeKHR::MAILBOX => HalPresentMode::Mailbox,
        _ => return None,
    })
}

fn hal_alpha_to_vk(mode: HalCompositeAlphaMode) -> vk::CompositeAlphaFlagsKHR {
    match mode {
        HalCompositeAlphaMode::Opaque => vk::CompositeAlphaFlagsKHR::OPAQUE,
        HalCompositeAlphaMode::Premultiplied => vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        HalCompositeAlphaMode::Unpremultiplied => vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        HalCompositeAlphaMode::Inherit => vk::CompositeAlphaFlagsKHR::INHERIT,
    }
}

fn vk_composite_alpha_modes(flags: vk::CompositeAlphaFlagsKHR) -> Vec<HalCompositeAlphaMode> {
    [
        HalCompositeAlphaMode::Opaque,
        HalCompositeAlphaMode::Premultiplied,
        HalCompositeAlphaMode::Unpremultiplied,
        HalCompositeAlphaMode::Inherit,
    ]
    .into_iter()
    .filter(|mode| flags.contains(hal_alpha_to_vk(*mode)))
    .collect()
}

fn vk_surface_usages(flags: vk::ImageUsageFlags) -> HalTextureUsage {
    HalTextureUsage {
        copy_src: flags.contains(vk::ImageUsageFlags::TRANSFER_SRC),
        copy_dst: flags.contains(vk::ImageUsageFlags::TRANSFER_DST),
        texture_binding: flags.contains(vk::ImageUsageFlags::SAMPLED),
        storage_binding: flags.contains(vk::ImageUsageFlags::STORAGE),
        render_attachment: flags.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT),
        transient: false,
    }
}

fn swapchain_image_usage(
    requested: HalTextureUsage,
    supported: vk::ImageUsageFlags,
) -> Result<vk::ImageUsageFlags, HalError> {
    let usage = hal_usage_to_vk_image_usage(requested);
    if usage.is_empty() || requested.transient || !supported.contains(usage) {
        return Err(surface_query_error());
    }
    Ok(usage | (supported & vk::ImageUsageFlags::TRANSFER_DST))
}

fn hal_usage_to_vk_image_usage(usage: HalTextureUsage) -> vk::ImageUsageFlags {
    let mut flags = vk::ImageUsageFlags::empty();
    for (enabled, flag) in [
        (usage.copy_src, vk::ImageUsageFlags::TRANSFER_SRC),
        (usage.copy_dst, vk::ImageUsageFlags::TRANSFER_DST),
        (usage.texture_binding, vk::ImageUsageFlags::SAMPLED),
        (usage.storage_binding, vk::ImageUsageFlags::STORAGE),
        (
            usage.render_attachment,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ),
    ] {
        if enabled {
            flags |= flag;
        }
    }
    flags
}

#[cfg(test)]
mod tests {
    #[test]
    fn teardown_waits_only_once_including_direct_drop() {
        let waited = AtomicBool::new(false);
        let mut count = 0;
        wait_for_teardown_once(&waited, || count += 1);
        wait_for_teardown_once(&waited, || count += 1);
        assert_eq!(count, 1);
        assert!(waited.load(Ordering::Acquire));
        let direct_drop = AtomicBool::new(false);
        wait_for_teardown_once(&direct_drop, || count += 1);
        assert_eq!(count, 2);
    }

    #[test]
    fn swapchain_image_usage_validates_requested_bits_before_adding_clear_usage() {
        let requested = HalTextureUsage {
            render_attachment: true,
            copy_src: true,
            ..vk_surface_usages(vk::ImageUsageFlags::empty())
        };
        let flags = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;
        assert_eq!(swapchain_image_usage(requested, flags).unwrap(), flags);
        assert_eq!(
            swapchain_image_usage(requested, flags | vk::ImageUsageFlags::TRANSFER_DST).unwrap(),
            flags | vk::ImageUsageFlags::TRANSFER_DST
        );
        assert!(swapchain_image_usage(
            requested,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST
        )
        .is_err());
        assert!(
            swapchain_image_usage(vk_surface_usages(vk::ImageUsageFlags::empty()), flags).is_err()
        );
        assert!(swapchain_image_usage(
            HalTextureUsage {
                transient: true,
                ..requested
            },
            flags
        )
        .is_err());
    }

    #[test]
    fn swapchain_extent_uses_current_extent() {
        let caps = vk::SurfaceCapabilitiesKHR::default().current_extent(vk::Extent2D {
            width: 64,
            height: 64,
        });
        assert_eq!(
            swapchain_extent(300, 200, &caps).unwrap(),
            caps.current_extent
        );
    }

    #[test]
    fn swapchain_extent_clamps_configured_extent() {
        let caps = vk::SurfaceCapabilitiesKHR::default()
            .current_extent(vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            })
            .min_image_extent(vk::Extent2D {
                width: 16,
                height: 16,
            })
            .max_image_extent(vk::Extent2D {
                width: 256,
                height: 256,
            });
        assert_eq!(
            swapchain_extent(300, 200, &caps).unwrap(),
            vk::Extent2D {
                width: 256,
                height: 200
            }
        );
        assert_eq!(
            swapchain_extent(0, 1, &caps).unwrap(),
            caps.min_image_extent
        );
    }

    #[test]
    fn swapchain_extent_rejects_zero_dimensions() {
        for (width, height) in [(0, 0), (0, 64), (64, 0)] {
            let caps = vk::SurfaceCapabilitiesKHR::default()
                .current_extent(vk::Extent2D { width, height });
            assert!(matches!(
                swapchain_extent(64, 64, &caps),
                Err(HalError::SwapchainCreationFailed {
                    message: "surface reports a zero extent",
                    ..
                })
            ));
        }
        let caps = vk::SurfaceCapabilitiesKHR::default()
            .current_extent(vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            })
            .max_image_extent(vk::Extent2D {
                width: 256,
                height: 256,
            });
        assert!(swapchain_extent(0, 64, &caps).is_err());
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_swapchain_texture_skips_view_for_copy_only_usage() {
        let device = vulkan_device();
        let mut config = surface_config();
        config.usage = vk_surface_usages(
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
        );
        // A null image is safe here: copy-only wrapping must make no image-view calls.
        let texture = create_swapchain_texture(
            Arc::clone(&device.inner),
            vk::Image::null(),
            vk::Format::B8G8R8A8_UNORM,
            config,
            vk::Extent2D {
                width: 4,
                height: 4,
            },
            (4, hal_usage_to_vk_image_usage(config.usage)),
            Arc::new(Mutex::new(SurfacePendingState::new())),
        )
        .unwrap();
        assert_eq!(texture.inner.as_ref().unwrap().view, vk::ImageView::null());
        assert_eq!(texture.format, config.format);
    }

    #[test]
    fn dedup_preserving_order_keeps_first_occurrences() {
        assert_eq!(
            dedup_preserving_order([
                HalTextureFormat::Bgra8Unorm,
                HalTextureFormat::Rgba16Float,
                HalTextureFormat::Bgra8Unorm,
                HalTextureFormat::Bgra8UnormSrgb,
                HalTextureFormat::Rgba16Float,
            ]),
            [
                HalTextureFormat::Bgra8Unorm,
                HalTextureFormat::Rgba16Float,
                HalTextureFormat::Bgra8UnormSrgb,
            ]
        );
        assert!(dedup_preserving_order(Vec::<HalPresentMode>::new()).is_empty());
    }

    #[test]
    fn vulkan_surface_format_mapping_filters_unknown_formats() {
        for (vk, hal) in [
            (vk::Format::R8G8B8A8_UNORM, HalTextureFormat::Rgba8Unorm),
            (vk::Format::R8G8B8A8_SRGB, HalTextureFormat::Rgba8UnormSrgb),
            (vk::Format::B8G8R8A8_UNORM, HalTextureFormat::Bgra8Unorm),
            (vk::Format::B8G8R8A8_SRGB, HalTextureFormat::Bgra8UnormSrgb),
            (
                vk::Format::A2B10G10R10_UNORM_PACK32,
                HalTextureFormat::Rgb10a2Unorm,
            ),
            (
                vk::Format::R16G16B16A16_SFLOAT,
                HalTextureFormat::Rgba16Float,
            ),
        ] {
            assert_eq!(vk_surface_format_to_hal(vk), Some(hal));
        }
        assert_eq!(vk_surface_format_to_hal(vk::Format::UNDEFINED), None);
        assert_eq!(vk_surface_format_to_hal(vk::Format::D32_SFLOAT), None);
    }

    #[test]
    fn vulkan_surface_present_mapping_preserves_modes() {
        for (vk, hal) in [
            (vk::PresentModeKHR::FIFO, HalPresentMode::Fifo),
            (
                vk::PresentModeKHR::FIFO_RELAXED,
                HalPresentMode::FifoRelaxed,
            ),
            (vk::PresentModeKHR::IMMEDIATE, HalPresentMode::Immediate),
            (vk::PresentModeKHR::MAILBOX, HalPresentMode::Mailbox),
        ] {
            assert_eq!(vk_present_mode_to_hal(vk), Some(hal));
            assert_eq!(select_present_mode(hal, &[vk]), vk);
            assert_eq!(select_present_mode(hal, &[]), vk::PresentModeKHR::FIFO);
        }
        assert_eq!(
            vk_present_mode_to_hal(vk::PresentModeKHR::from_raw(-1)),
            None
        );
    }

    #[test]
    fn vulkan_surface_alpha_mapping_preserves_dawn_order() {
        let modes = [
            HalCompositeAlphaMode::Opaque,
            HalCompositeAlphaMode::Premultiplied,
            HalCompositeAlphaMode::Unpremultiplied,
            HalCompositeAlphaMode::Inherit,
        ];
        let mut flags = vk::CompositeAlphaFlagsKHR::empty();
        assert!(vk_composite_alpha_modes(flags).is_empty());
        for mode in modes {
            let flag = hal_alpha_to_vk(mode);
            assert_eq!(vk_composite_alpha_modes(flag), [mode]);
            flags |= flag;
        }
        assert_eq!(vk_composite_alpha_modes(flags), modes);
    }

    #[test]
    fn vulkan_surface_usage_mapping_round_trips_supported_bits() {
        let bits = [
            vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageUsageFlags::TRANSFER_DST,
            vk::ImageUsageFlags::SAMPLED,
            vk::ImageUsageFlags::STORAGE,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ];
        for mask in 0..32 {
            let flags = bits
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .fold(vk::ImageUsageFlags::empty(), |flags, (_, bit)| flags | *bit);
            let usage = vk_surface_usages(flags | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);
            assert_eq!(hal_usage_to_vk_image_usage(usage), flags);
            assert!(!usage.transient);
        }
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    fn vulkan_surface_capabilities_rejects_dummy_surface() {
        let instance = VulkanInstance::new().unwrap();
        let adapter = instance.enumerate_adapters().remove(0);
        // The helper has a null VkSurfaceKHR, so a driver query is not valid.
        assert!(dummy_surface(&instance).capabilities(&adapter).is_err());
    }

    use super::super::test_helpers::*;
    use super::*;

    #[test]
    fn select_present_mode_honors_fifo_relaxed_when_supported() {
        assert_eq!(
            select_present_mode(
                crate::HalPresentMode::FifoRelaxed,
                &[vk::PresentModeKHR::FIFO_RELAXED]
            ),
            vk::PresentModeKHR::FIFO_RELAXED
        );
        assert_eq!(
            select_present_mode(
                crate::HalPresentMode::FifoRelaxed,
                &[vk::PresentModeKHR::FIFO]
            ),
            vk::PresentModeKHR::FIFO
        );
    }

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
    fn vulkan_retire_ring_wait_all_retires_signaled_fence() {
        let device = vulkan_device();
        let fence = signaled_fence(&device);
        let mut retire = RetireRing::new(1);
        retire
            .retire(&device.inner.device, fence, Vec::new(), Vec::new(), true)
            .expect("retire signaled fence");
        retire
            .wait_all(&device.inner.device)
            .expect("wait all retired frames");
        assert!(retire.frames.iter().all(Option::is_none));
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_retire_ring_wait_all_releases_retained_buffer() {
        let device = vulkan_device();
        let buffer = device
            .create_buffer(16, HalBufferUsage::default())
            .expect("Vulkan buffer allocation should succeed");
        let inner = Arc::clone(buffer.inner.as_ref().expect("buffer allocation"));
        let fence = signaled_fence(&device);
        let mut retire = RetireRing::new(1);
        retire
            .retire(
                &device.inner.device,
                fence,
                Vec::new(),
                vec![RetainedResource::Buffer {
                    _inner: Arc::clone(&inner),
                }],
                true,
            )
            .expect("retire signaled fence");
        assert_eq!(Arc::strong_count(&inner), 3);
        drop(buffer);
        assert_eq!(Arc::strong_count(&inner), 2);
        retire
            .wait_all(&device.inner.device)
            .expect("wait all retired frames");
        assert_eq!(Arc::strong_count(&inner), 1);
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_retire_ring_slot_reuse_releases_retained_texture() {
        let device = vulkan_device();
        let texture = device
            .create_texture(&texture_descriptor())
            .expect("Vulkan texture allocation should succeed");
        let inner = Arc::clone(texture.inner.as_ref().expect("texture allocation"));
        let first_fence = signaled_fence(&device);
        let second_fence = signaled_fence(&device);
        let mut retire = RetireRing::new(1);
        retire
            .retire(
                &device.inner.device,
                first_fence,
                Vec::new(),
                vec![RetainedResource::Texture {
                    _inner: Arc::clone(&inner),
                }],
                true,
            )
            .expect("retire first fence");
        assert_eq!(Arc::strong_count(&inner), 3);
        drop(texture);
        assert_eq!(Arc::strong_count(&inner), 2);
        retire
            .retire(
                &device.inner.device,
                second_fence,
                Vec::new(),
                Vec::new(),
                true,
            )
            .expect("reuse retire slot");
        assert_eq!(Arc::strong_count(&inner), 1);
        retire
            .wait_all(&device.inner.device)
            .expect("wait remaining retired frame");
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_retire_ring_retire_drains_signaled_overflow_without_shrinking_slots() {
        let device = vulkan_device();
        let mut retire = RetireRing::new(1);
        let first_fence = signaled_fence(&device);
        retire
            .retire(
                &device.inner.device,
                first_fence,
                Vec::new(),
                Vec::new(),
                true,
            )
            .expect("retire first fence");

        let overflow_fence = signaled_fence(&device);
        retire.overflow.push(InFlightFrame {
            fence: overflow_fence,
            cleanup: Vec::new(),
            retained: Vec::new(),
            destroy_fence: true,
            tracked_submission: None,
        });
        let slot_count = retire.frames.len();
        let replacement_fence = signaled_fence(&device);
        retire
            .retire(
                &device.inner.device,
                replacement_fence,
                Vec::new(),
                Vec::new(),
                true,
            )
            .expect("retire replacement fence");

        assert!(retire.overflow.is_empty());
        assert_eq!(retire.frames.len(), slot_count);
        assert!(retire.frames.iter().all(Option::is_some));
        retire
            .wait_all(&device.inner.device)
            .expect("wait remaining retired frame");
    }

    #[test]
    #[ignore = "manual real Vulkan backend test"]
    #[cfg(feature = "vulkan")]
    fn vulkan_retire_ring_wait_all_drains_overflow() {
        let device = vulkan_device();
        let mut retire = RetireRing::new(1);
        let overflow_fence = signaled_fence(&device);
        retire.overflow.push(InFlightFrame {
            fence: overflow_fence,
            cleanup: Vec::new(),
            retained: Vec::new(),
            destroy_fence: true,
            tracked_submission: None,
        });

        retire
            .wait_all(&device.inner.device)
            .expect("wait all retired frames");

        assert!(retire.overflow.is_empty());
        assert!(retire.frames.iter().all(Option::is_none));
    }

    #[cfg(feature = "vulkan")]
    fn signaled_fence(device: &VulkanDevice) -> vk::Fence {
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        unsafe {
            device
                .inner
                .device
                .create_fence(&fence_info, None)
                .expect("create signaled fence")
        }
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
