use std::{ffi::c_void, sync::Arc};

use cudarc::driver::{CudaStream, sys::CUstream};

use crate::PmeRecip;

#[repr(C)]
pub struct VkContext {
    pub handle: *mut c_void,
}

// We don't necessarily expect to use this, but required if a struct
// containing this derives it.
impl Default for VkContext {
    fn default() -> Self {
        unsafe {
            let handle = vk_make_context_default();
            assert!(!handle.is_null(), "vk_make_context_default() returned null");
            VkContext { handle }
        }
    }
}

impl VkContext {
    /// Adopt an existing cudarc stream (we will NOT destroy it).
    pub fn from_cudarc_stream(stream: &Arc<CudaStream>) -> Self {
        let cu: CUstream = stream.cu_stream();
        let handle = unsafe { vk_make_context_from_stream(cu as *mut c_void) };
        assert!(
            !handle.is_null(),
            "vk_make_context_from_stream returned null"
        );
        VkContext { handle }
    }
}

unsafe extern "C" {
    // plan lifecycle
    pub fn make_plan_r2c_c2r_many(ctx: *mut c_void, nx: i32, ny: i32, nz: i32) -> *mut c_void;
    pub fn destroy_plan(plan: *mut c_void);

    pub fn exec_inverse_c2r(
        plan: *mut c_void,
        complex_in_dev: *mut c_void,
        real_out_dev: *mut c_void,
    );

    // (optional) context lifecycle if you want to create it here
    pub fn vk_make_context_default() -> *mut c_void;
    fn vk_make_context_from_stream(cu_stream: *mut c_void) -> *mut c_void;
    pub fn vk_destroy_context(ctx: *mut c_void);
}

impl PmeRecip {
    // vkFFT plan handle + cached dims
    pub(crate) fn vk_plan_ptr(&self) -> *mut c_void {
        self.planner_gpu
    }
}

/// Create the GPU plan. Run this at init, or when dimensions change.
pub(crate) fn create_gpu_plan(dims: (usize, usize, usize), ctx: &Arc<VkContext>) -> *mut c_void {
    let (nx, ny, nz) = (dims.0 as i32, dims.1 as i32, dims.2 as i32);

    unsafe { make_plan_r2c_c2r_many(ctx.handle, nx, ny, nz) }
}
