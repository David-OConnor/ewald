//! We use *host-side* cuFFT functions for long-range reciprical FFTs. This module contains FFI
//! bindings between the rust code, and cuFFT FFI functions.

// todo: Organize both this and teh .cu file. REmove unused, make order sensitible, and cyn order.

use std::{ffi::c_void, sync::Arc};

use cudarc::driver::{CudaSlice, CudaStream, sys::CUstream};

use crate::{GpuTables, PmeRecip, cufft};

unsafe extern "C" {
    pub(crate) fn spme_make_plan_r2c_c2r_many(
        nx: i32,
        ny: i32,
        nz: i32,
        cu_stream: *mut c_void,
    ) -> *mut c_void;

    pub(crate) fn spme_exec_inverse_ExEyEz_c2r(
        plan: *mut c_void,
        exk: *mut c_void, // cufftComplex*
        eyk: *mut c_void,
        ezk: *mut c_void,
        ex: *mut c_void,
        ey: *mut c_void,
        ez: *mut c_void,
    );

    pub(crate) fn spme_scale_ExEyEz_after_c2r(
        ex: *mut c_void,
        ey: *mut c_void,
        ez: *mut c_void,
        nx: i32,
        ny: i32,
        nz: i32,
        cu_stream: *mut c_void,
    );

    pub(crate) fn spme_apply_ghat_and_grad_launch(
        rho: *const c_void,
        exk: *mut c_void,
        eyk: *mut c_void,
        ezk: *mut c_void,
        kx: *const c_void,
        ky: *const c_void,
        kz: *const c_void,
        bx: *const c_void,
        by: *const c_void,
        bz: *const c_void,
        nx: i32,
        ny: i32,
        nz: i32,
        vol: f32,
        alpha: f32,
        cu_stream: *mut c_void,
    );

    pub(crate) fn spme_destroy_plan_r2c_c2r_many(plan: *mut c_void);

    pub(crate) fn spme_energy_half_spectrum_launch(
        rho_k: *const c_void,
        kx: *const c_void,
        ky: *const c_void,
        kz: *const c_void,
        bx: *const c_void,
        by: *const c_void,
        bz: *const c_void,
        nx: i32,
        ny: i32,
        nz: i32,
        vol: f32,
        alpha: f32,
        partial_sums: *mut c_void, // device buffer double[blocks]
        blocks: i32,
        threads: i32,
        cu_stream: *mut c_void,
    );

    pub(crate) fn spme_scatter_rho_4x4x4_launch(
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

    pub(crate) fn spme_exec_forward_r2c(
        plan: *mut c_void,
        rho_real: *mut c_void,
        rho_k: *mut c_void,
    );

    pub(crate) fn spme_gather_forces_to_atoms_launch(
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
}

impl Drop for PmeRecip {
    fn drop(&mut self) {
        unsafe {
            if !self.planner_gpu.is_null() {
                spme_destroy_plan_r2c_c2r_many(self.planner_gpu);
                self.planner_gpu = std::ptr::null_mut();
            }
        }
    }
}

impl PmeRecip {
    pub fn ensure_gpu_plan(&mut self, stream: &Arc<CudaStream>) {
        use std::ffi::c_void;
        let dims = (self.nx as i32, self.ny as i32, self.nz as i32);

        let cu_stream: CUstream = stream.cu_stream();
        let raw_stream: *mut c_void = cu_stream as *mut c_void;

        unsafe {
            if self.planner_gpu.is_null() || self.plan_dims != dims {
                if !self.planner_gpu.is_null() {
                    spme_destroy_plan_r2c_c2r_many(self.planner_gpu);
                    self.planner_gpu = std::ptr::null_mut();
                }
                self.planner_gpu = spme_make_plan_r2c_c2r_many(dims.0, dims.1, dims.2, raw_stream);
                self.plan_dims = dims;
                #[cfg(feature = "cuda")]
                {
                    self.gpu_tables = None;
                }
            }
        }
    }
}

impl PmeRecip {
    pub fn ensure_gpu_tables(&mut self, stream: &Arc<CudaStream>) {
        let dims = (self.nx as i32, self.ny as i32, self.nz as i32);
        if self.plan_dims != dims {
            return;
        }

        if self.gpu_tables.is_none() {
            self.gpu_tables = Some(GpuTables {
                kx: stream.memcpy_stod(&self.kx).unwrap(),
                ky: stream.memcpy_stod(&self.ky).unwrap(),
                kz: stream.memcpy_stod(&self.kz).unwrap(),
                bx: stream.memcpy_stod(&self.bmod2_x).unwrap(),
                by: stream.memcpy_stod(&self.bmod2_y).unwrap(),
                bz: stream.memcpy_stod(&self.bmod2_z).unwrap(),
            });
        }
    }
}
