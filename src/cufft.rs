//! We use *host-side* cuFFT functions for long-range reciprical FFTs. This module contains FFI
//! bindings between the rust code, and cuFFT FFI functions.

// todo: Organize both this and teh .cu file. REmove unused, make order sensitible, and cyn order.

// todo: Rremove the spme_ prefix here and in the modules

use std::{ffi::c_void, ptr::null_mut, sync::Arc};

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use lin_alg::f32::Vec3;

use crate::{GpuTables, PmeRecip, dev_ptr, dev_ptr_mut};

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

    // todo: Used by both.
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

    // todo: Used by both.
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

impl PmeRecip {
    pub fn forces_gpu(
        &mut self,
        stream: &Arc<CudaStream>,
        pos: &[Vec3],
        q: &[f32],
    ) -> (Vec<Vec3>, f32) {
        if self.gpu_tables.is_none() {
            // First run
            let k = (&self.kx, &self.ky, &self.kz);
            let bmod2 = (&self.bmod2_x, &self.bmod2_y, &self.bmod2_z);

            self.planner_gpu = create_gpu_plan(self.plan_dims, stream);
            self.gpu_tables = Some(GpuTables::new(k, bmod2, stream));
        }

        assert_eq!(pos.len(), q.len());

        let (lx, ly, lz) = self.box_dims;
        let (nx, ny, nz) = self.plan_dims;

        let n_real = nx * ny * nz;
        let n_cmplx = nx * ny * (nz / 2 + 1); // half-spectrum length
        let complex_len = n_cmplx * 2; // (re,im) interleaved

        // Contiguous real buffer: [ex | ey | ez]
        let ex_ey_ez_d: CudaSlice<f32> = stream.alloc_zeros(3 * n_real).unwrap();
        let (base_r_ptr, _) = ex_ey_ez_d.device_ptr(stream);
        let base_r = base_r_ptr as usize;
        let stride_r_bytes = n_real * size_of::<f32>();

        let ex_ptr = base_r as *mut c_void;
        let ey_ptr = (base_r + stride_r_bytes) as *mut c_void;
        let ez_ptr = (base_r + 2 * stride_r_bytes) as *mut c_void;

        // Contiguous complex buffer: [exk | eyk | ezk]
        let exeyezk_d: CudaSlice<f32> = stream.alloc_zeros(3 * complex_len).unwrap();
        let (base_ptr, _) = exeyezk_d.device_ptr(stream);
        let base = base_ptr as usize;
        let stride_bytes = complex_len * size_of::<f32>();

        let exk_ptr = base as *mut c_void;
        let eyk_ptr = (base + stride_bytes) as *mut c_void;
        let ezk_ptr = (base + 2 * stride_bytes) as *mut c_void;

        let cu_stream = stream.cu_stream() as *mut c_void;

        let tables = self.gpu_tables.as_ref().unwrap();

        let (kx_ptr, _) = tables.kx.device_ptr(stream);
        let (ky_ptr, _) = tables.ky.device_ptr(stream);
        let (kz_ptr, _) = tables.kz.device_ptr(stream);
        let (bx_ptr, _) = tables.bx.device_ptr(stream);
        let (by_ptr, _) = tables.by.device_ptr(stream);
        let (bz_ptr, _) = tables.bz.device_ptr(stream);

        // H2D: positions & charges (flattened)
        let pos_flat: Vec<f32> = pos.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let pos_d: CudaSlice<f32> = stream.memcpy_stod(&pos_flat).unwrap();
        let q_d: CudaSlice<f32> = stream.memcpy_stod(q).unwrap();

        // rho_real on device
        let rho_real_d: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();

        // Scatter on GPU
        unsafe {
            spme_scatter_rho_4x4x4_launch(
                dev_ptr(&pos_d, stream),
                dev_ptr(&q_d, stream),
                dev_ptr_mut(&rho_real_d, stream),
                pos.len() as i32,
                nx as i32,
                ny as i32,
                nz as i32,
                lx,
                ly,
                lz,
                cu_stream,
            );
        }

        // rho(k) (half-spectrum) on device
        let rho_k_d: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();
        let (rho_k_ptr, _) = rho_k_d.device_ptr(stream);

        // Forward FFT on GPU: real -> half complex
        unsafe {
            spme_exec_forward_r2c(
                self.planner_gpu,
                rho_real_ptr as *mut _,
                rho_k_ptr as *mut _,
            );

            // Apply G(k) and gradient to get Exk/Eyk/Ezk
            spme_apply_ghat_and_grad_launch(
                rho_k_ptr as *const _,
                exk_ptr as *mut _,
                eyk_ptr as *mut _,
                ezk_ptr as *mut _,
                kx_ptr as *const _,
                ky_ptr as *const _,
                kz_ptr as *const _,
                bx_ptr as *const _,
                by_ptr as *const _,
                bz_ptr as *const _,
                nx as i32,
                ny as i32,
                nz as i32,
                self.vol,
                self.alpha,
                cu_stream,
            );

            // Inverse batched C2R: (exk,eyk,ezk) -> (ex,ey,ez)
            spme_exec_inverse_ExEyEz_c2r(
                self.planner_gpu,
                exk_ptr as *mut _,
                eyk_ptr as *mut _,
                ezk_ptr as *mut _,
                ex_ptr as *mut _,
                ey_ptr as *mut _,
                ez_ptr as *mut _,
            );

            // Scale by 1/N once here
            spme_scale_ExEyEz_after_c2r(
                ex_ptr, ey_ptr, ez_ptr, nx as i32, ny as i32, nz as i32, cu_stream,
            );
        }

        // Gather forces F = q * E
        let f_d: CudaSlice<f32> = stream.alloc_zeros(pos.len() * 3).unwrap();
        let (f_ptr, _) = f_d.device_ptr(stream);
        let inv_n = 1.0f32 / (nx as f32 * ny as f32 * nz as f32);
        unsafe {
            spme_gather_forces_to_atoms_launch(
                pos_ptr as *const _,
                ex_ptr as *const _,
                ey_ptr as *const _,
                ez_ptr as *const _,
                q_ptr as *const _,
                f_ptr as *mut _,
                pos.len() as i32,
                nx as i32,
                ny as i32,
                nz as i32,
                self.box_dims.0,
                self.box_dims.1,
                self.box_dims.2,
                inv_n,
                cu_stream,
            );
        }

        // Energy (half spectrum)
        let n_threads = 256;
        let blocks = {
            let n = n_cmplx as i32;
            (n + n_threads - 1) / n_threads
        };
        let partial_d: CudaSlice<f64> = stream.alloc_zeros(blocks as usize).unwrap();
        let (partial_ptr, _) = partial_d.device_ptr(stream);
        unsafe {
            spme_energy_half_spectrum_launch(
                rho_k_ptr as *const _,
                kx_ptr as *const _,
                ky_ptr as *const _,
                kz_ptr as *const _,
                bx_ptr as *const _,
                by_ptr as *const _,
                bz_ptr as *const _,
                nx as i32,
                ny as i32,
                nz as i32,
                self.vol,
                self.alpha,
                partial_ptr as *mut _,
                blocks,
                n_threads,
                cu_stream,
            );
        }
        let energy = stream
            .memcpy_dtov(&partial_d)
            .unwrap()
            .into_iter()
            .sum::<f64>() as f32;

        // D2H forces
        let f_host: Vec<f32> = stream.memcpy_dtov(&f_d).unwrap();
        let mut f = Vec::with_capacity(pos.len());
        for i in 0..pos.len() {
            f.push(Vec3 {
                x: f_host[i * 3 + 0],
                y: f_host[i * 3 + 1],
                z: f_host[i * 3 + 2],
            });
        }

        (f, energy)
    }
}

impl Drop for PmeRecip {
    fn drop(&mut self) {
        unsafe {
            if !self.planner_gpu.is_null() {
                spme_destroy_plan_r2c_c2r_many(self.planner_gpu);
                self.planner_gpu = null_mut();
            }
        }
    }
}

/// Create the GPU plan. Run this at init, or when dimensions change.
pub(crate) fn create_gpu_plan(
    dims: (usize, usize, usize),
    stream: &Arc<CudaStream>,
) -> *mut c_void {
    let raw_stream: *mut c_void = stream.cu_stream() as *mut c_void;

    let (nx, ny, nz) = (dims.0 as i32, dims.1 as i32, dims.2 as i32);

    unsafe { spme_make_plan_r2c_c2r_many(nx, ny, nz, raw_stream) }
}
