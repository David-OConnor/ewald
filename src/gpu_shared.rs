//! Used by both vkFFT and cuFFT pipelines.

use std::{ffi::c_void, sync::Arc};

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};

use crate::PmeRecip;
#[cfg(feature = "cufft")]
use crate::cufft::spme_destroy_plan_r2c_c2r_many;
#[cfg(feature = "vkfft")]
use crate::vk_fft::vkfft_destroy_plan;

pub(crate) struct GpuTables {
    pub kx: CudaSlice<f32>,
    pub ky: CudaSlice<f32>,
    pub kz: CudaSlice<f32>,
    pub bx: CudaSlice<f32>,
    pub by: CudaSlice<f32>,
    pub bz: CudaSlice<f32>,
}

impl GpuTables {
    pub(crate) fn new(
        k: (&Vec<f32>, &Vec<f32>, &Vec<f32>),
        bmod2: (&Vec<f32>, &Vec<f32>, &Vec<f32>),
        stream: &Arc<CudaStream>,
    ) -> Self {
        Self {
            kx: stream.memcpy_stod(k.0).unwrap(),
            ky: stream.memcpy_stod(k.1).unwrap(),
            kz: stream.memcpy_stod(k.2).unwrap(),
            bx: stream.memcpy_stod(bmod2.0).unwrap(),
            by: stream.memcpy_stod(bmod2.1).unwrap(),
            bz: stream.memcpy_stod(bmod2.2).unwrap(),
        }
    }
}

unsafe extern "C" {
    pub(crate) fn gather_forces_to_atoms_launch(
        pos: *const c_void,
        ex: *const c_void,
        ey: *const c_void,
        ez: *const c_void,
        q: *const c_void,
        out_f: *mut c_void,
        n_atoms: i32,
        nx: i32,
        ny: i32,
        nz: i32,
        lx: f32,
        ly: f32,
        lz: f32,
        inv_n: f32,
        cu_stream: *mut c_void,
    );

    pub(crate) fn scale_ExEyEz_after_c2r(
        ex: *mut c_void,
        ey: *mut c_void,
        ez: *mut c_void,
        nx: i32,
        ny: i32,
        nz: i32,
        cu_stream: *mut c_void,
    );

    pub(crate) fn scatter_rho_4x4x4_launch(
        pos: *const c_void,
        q: *const c_void,
        rho: *mut c_void,
        n_atoms: i32,
        nx: i32,
        ny: i32,
        nz: i32,
        lx: f32,
        ly: f32,
        lz: f32,
        cu_stream: *mut c_void,
    );
}

impl Drop for PmeRecip {
    fn drop(&mut self) {
        unsafe {
            if !self.planner_gpu.is_null() {
                #[cfg(feature = "vkfft")]
                vkfft_destroy_plan(self.planner_gpu);
                #[cfg(feature = "cufft")]
                spme_destroy_plan_r2c_c2r_many(self.planner_gpu);
                self.planner_gpu = std::ptr::null_mut();
            }
        }
    }
}

pub(crate) fn dev_ptr<T>(buf: &CudaSlice<T>, stream: &Arc<CudaStream>) -> *const c_void {
    let (p, _) = buf.device_ptr(stream);
    p as *const c_void
}

pub(crate) fn dev_ptr_mut<T>(buf: &CudaSlice<T>, stream: &Arc<CudaStream>) -> *mut c_void {
    let (p, _) = buf.device_ptr(stream);
    p as *mut c_void
}

pub(crate) fn split3(
    buf: &CudaSlice<f32>,
    len: usize,
    stream: &Arc<CudaStream>,
) -> (*mut c_void, *mut c_void, *mut c_void) {
    let (base, _) = buf.device_ptr(stream);
    let base = base as usize;

    let stride = len * size_of::<f32>();
    (
        base as *mut c_void,
        (base + stride) as *mut c_void,
        (base + 2 * stride) as *mut c_void,
    )
}
