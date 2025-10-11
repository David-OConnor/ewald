use std::{ffi::c_void, mem::size_of, sync::Arc};

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, sys::CUstream};
use lin_alg::f32::Vec3;

use crate::{
    PmeRecip, SQRT_PI,
    gpu_shared::{
        GpuTables, dev_ptr, dev_ptr_mut, gather_forces_to_atoms_launch, scale_ExEyEz_after_c2r,
    },
};
use crate::gpu_shared::{scatter_rho_4x4x4_launch};

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

#[link(name = "vk_fft")] // built from vk_fft.c via build.rs
unsafe extern "C" {
    // plan lifecycle
    pub fn vkfft_make_plan_r2c_c2r_many(ctx: *mut c_void, nx: i32, ny: i32, nz: i32)
                                        -> *mut c_void;
    pub fn vkfft_destroy_plan(plan: *mut c_void);

    // device memory helpers
    // pub fn vk_alloc_and_upload(ctx: *mut c_void, host_src: *const u8, nbytes: u64) -> *mut c_void;
    // pub fn vk_alloc_zeroed(ctx: *mut c_void, nbytes: u64) -> *mut c_void;
    // pub fn vk_download(ctx: *mut c_void, dev_buf: *mut c_void, host_dst: *mut u8, nbytes: u64);
    // pub fn vk_free(ctx: *mut c_void, dev_buf: *mut c_void);

    // FFT executions
    pub fn vkfft_exec_forward_r2c(
        plan: *mut c_void,
        real_in_dev: *mut c_void,
        complex_out_dev: *mut c_void,
    );

    pub fn vkfft_exec_inverse_c2r(
        plan: *mut c_void,
        complex_in_dev: *mut c_void,
        real_out_dev: *mut c_void,
    );

    // SPME kernels on k-grid (compute shaders)
    pub fn vk_apply_ghat_and_grad(
        ctx: *mut c_void,
        rho_k: *mut c_void,
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
    );

    pub fn vk_spme_scale_real_fields(
        ex: *mut c_void,
        ey: *mut c_void,
        ez: *mut c_void,
        nx: i32,
        ny: i32,
        nz: i32,
    );

    pub fn vk_energy_half_spectrum_sum(
        ctx: *mut c_void,
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
    ) -> f64;

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

impl PmeRecip {
    // todo: DRY with the cuFFT version; unify.
    pub fn forces_gpu(
        &mut self,
        // todo: Do we still need VkContext if we are using CudaStream?
        ctx: &Arc<VkContext>,
        stream: &Arc<CudaStream>,
        pos: &[Vec3],
        q: &[f32],
    ) -> (Vec<Vec3>, f32) {
        assert_eq!(pos.len(), q.len());

        if self.gpu_tables.is_none() {
            // First run
            let k = (&self.kx, &self.ky, &self.kz);
            let bmod2 = (&self.bmod2_x, &self.bmod2_y, &self.bmod2_z);

            self.planner_gpu = create_gpu_plan(self.plan_dims, ctx);
            // self.gpu_tables = Some(GpuTables::new(k, bmod2, ctx));
            self.gpu_tables = Some(GpuTables::new(k, bmod2, stream));
        }

        let cu_stream = stream.cu_stream() as *mut c_void;
        let tables = self.gpu_tables.as_ref().unwrap();
        let f32_size = size_of::<f32>();

        let (lx, ly, lz) = self.box_dims;
        let (nx, ny, nz) = self.plan_dims;

        let n_real = nx * ny * nz;
        let n_cmplx = nx * ny * (nz / 2 + 1); // half-spectrum length
        let complex_len = n_cmplx * 2; // interleaved re,im

        // Contiguous real buffer: [ex | ey | ez]
        let ex_ey_ez_d: CudaSlice<f32> = stream.alloc_zeros(3 * n_real).unwrap();
        let (base_r_ptr, _) = ex_ey_ez_d.device_ptr(stream);
        let base_r = base_r_ptr as usize;
        let strike_r_bytes = n_real * f32_size;

        let ex_ptr = base_r as *mut c_void;
        let ey_ptr = (base_r + strike_r_bytes) as *mut c_void;
        let ez_ptr = (base_r + 2 * strike_r_bytes) as *mut c_void;

        // Contiguous complex buffer: [exk | eyk | ezk]
        let exeyezk_d: CudaSlice<f32> = stream.alloc_zeros(3 * complex_len).unwrap();
        let (base_ptr, _) = exeyezk_d.device_ptr(stream);
        let base = base_ptr as usize;
        let stride = complex_len * size_of::<f32>();

        let exk_ptr = base as *mut c_void;
        let eyk_ptr = (base + stride) as *mut c_void;
        let ezk_ptr = (base + 2 * stride) as *mut c_void;

        let kx_ptr = dev_ptr(&tables.kx, stream);
        let ky_ptr = dev_ptr(&tables.ky, stream);
        let kz_ptr = dev_ptr(&tables.kz, stream);

        let bx_ptr = dev_ptr(&tables.bx, stream);
        let by_ptr = dev_ptr(&tables.by, stream);
        let bz_ptr = dev_ptr(&tables.bz, stream);

        // H2D: positions & charges (flattened)
        let pos_flat: Vec<f32> = pos.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let pos_d: CudaSlice<f32> = stream.memcpy_stod(&pos_flat).unwrap();
        let q_d: CudaSlice<f32> = stream.memcpy_stod(q).unwrap();

        // rho_real on device
        let rho_real_d: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
        // rho(k) (half-spectrum) on device
        let rho_k_d: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();

        // Scatter on GPU
        unsafe {
            scatter_rho_4x4x4_launch(
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

        // Forward FFT on GPU: real -> half complex
        unsafe {
            vkfft_exec_forward_r2c(
                self.planner_gpu,
                dev_ptr_mut(&rho_real_d, stream),
                dev_ptr_mut(&rho_k_d, stream),
            );
        }

        unsafe {
            // apply G(k), grad on device
            vk_apply_ghat_and_grad(
                ctx.handle,
                dev_ptr_mut(&rho_k_d, stream),
                exk_ptr,
                eyk_ptr,
                ezk_ptr,
                kx_ptr,
                ky_ptr,
                kz_ptr,
                bx_ptr,
                by_ptr,
                bz_ptr,
                nx as i32,
                ny as i32,
                nz as i32,
                self.vol,
                self.alpha,
            );
        }

        unsafe {
            // todo: Make the vk plan many?
            // if your vk plan is "many", add a wrapper; else call 3x:
            vkfft_exec_inverse_c2r(self.planner_gpu, exk_ptr, ex_ptr);
            vkfft_exec_inverse_c2r(self.planner_gpu, eyk_ptr, ey_ptr);
            vkfft_exec_inverse_c2r(self.planner_gpu, ezk_ptr, ez_ptr);

            // same scaling kernel as cuFFT
            scale_ExEyEz_after_c2r(
                ex_ptr,
                ey_ptr,
                ez_ptr,
                nx as i32,
                ny as i32,
                nz as i32,
                cu_stream,
            );
        }
        // Gather forces F = q * E
        let f_d: CudaSlice<f32> = stream.alloc_zeros(pos.len() * 3).unwrap();
        let inv_n = 1.0f32 / (nx as f32 * ny as f32 * nz as f32);

        unsafe {
            gather_forces_to_atoms_launch(
                dev_ptr(&pos_d, stream),
                ex_ptr as *const _,
                ey_ptr as *const _,
                ez_ptr as *const _,
                dev_ptr(&q_d, stream),
                dev_ptr_mut(&f_d, stream),
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

        // energy on device (keep your vk reduction)
        let energy = unsafe {
            vk_energy_half_spectrum_sum(
                ctx.handle,
                dev_ptr(&rho_k_d, stream),
                dev_ptr(&tables.kx, stream),
                dev_ptr(&tables.ky, stream),
                dev_ptr(&tables.kz, stream),
                dev_ptr(&tables.bx, stream),
                dev_ptr(&tables.by, stream),
                dev_ptr(&tables.bz, stream),
                nx as i32,
                ny as i32,
                nz as i32,
                self.vol,
                self.alpha,
            )
        } as f32;

        let self_energy: f64 = -(self.alpha as f64 / SQRT_PI as f64)
            * q.iter().map(|&qi| (qi as f64) * (qi as f64)).sum::<f64>();
        let energy = (energy as f64 + self_energy) as f32;

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

// /// todo: Compare this to spread_charges (cplx version) in lib.rs. Figure out why you need
// /// todo this for vkfft, but use the cplx version there.
// fn spread_charges_real(&self, pos: &[Vec3], q: &[f32], rho: &mut [f32]) {
//     let (lx, ly, lz) = (self.box_dims.0, self.box_dims.1, self.box_dims.2);
//
//     let (nx, ny, nz) = (self.plan_dims.0, self.plan_dims.1, self.plan_dims.2);
//
//     let nxny = nx * ny;
//
//     for (r, &qi) in pos.iter().zip(q.iter()) {
//         let sx = r.x / lx * nx as f32;
//         let sy = r.y / ly * ny as f32;
//         let sz = r.z / lz * nz as f32;
//
//         let (ix0, wx) = bspline4_weights(sx);
//         let (iy0, wy) = bspline4_weights(sy);
//         let (iz0, wz) = bspline4_weights(sz);
//
//         for a in 0..4 {
//             let ix = wrap(ix0 + a as isize, nx);
//             let wxa = wx[a];
//             for b in 0..4 {
//                 let iy = wrap(iy0 + b as isize, ny);
//                 let wxy = wxa * wy[b];
//                 for c in 0..4 {
//                     let iz = wrap(iz0 + c as isize, nz);
//                     let idx = iz * nxny + iy * nx + ix;
//                     rho[idx] += qi * wxy * wz[c];
//                 }
//             }
//         }
//     }
// }
// }

// // todo: Can we make this CudaSlice<f32> instead of c_void, like we do with cuFFT?
// pub(crate) struct GpuTables {
//     kx: *mut c_void,
//     ky: *mut c_void,
//     kz: *mut c_void,
//     bx: *mut c_void,
//     by: *mut c_void,
//     bz: *mut c_void,
// }
//
// impl GpuTables {
//     pub(crate) fn new(
//         k: (&Vec<f32>, &Vec<f32>, &Vec<f32>),
//         bmod2: (&Vec<f32>, &Vec<f32>, &Vec<f32>),
//         ctx: &Arc<VkContext>,
//     ) -> Self {
//         let f32_size = size_of::<f32>();
//         unsafe {
//             let kx = vk_alloc_and_upload(
//                 ctx.handle,
//                 k.0.as_ptr() as *const u8,
//                 (k.0.len() * f32_size) as u64,
//             );
//             let ky = vk_alloc_and_upload(
//                 ctx.handle,
//                 k.1.as_ptr() as *const u8,
//                 (k.1.len() * f32_size) as u64,
//             );
//             let kz = vk_alloc_and_upload(
//                 ctx.handle,
//                 k.2.as_ptr() as *const u8,
//                 (k.2.len() * f32_size) as u64,
//             );
//             let bx = vk_alloc_and_upload(
//                 ctx.handle,
//                 bmod2.0.as_ptr() as *const u8,
//                 (bmod2.0.len() * f32_size) as u64,
//             );
//             let by = vk_alloc_and_upload(
//                 ctx.handle,
//                 bmod2.1.as_ptr() as *const u8,
//                 (bmod2.1.len() * f32_size) as u64,
//             );
//             let bz = vk_alloc_and_upload(
//                 ctx.handle,
//                 bmod2.2.as_ptr() as *const u8,
//                 (bmod2.2.len() * f32_size) as u64,
//             );
//
//             Self {
//                 kx,
//                 ky,
//                 kz,
//                 bx,
//                 by,
//                 bz,
//             }
//         }
//     }
// }


/// Create the GPU plan. Run this at init, or when dimensions change.
pub(crate) fn create_gpu_plan(dims: (usize, usize, usize), ctx: &Arc<VkContext>) -> *mut c_void {
    let (nx, ny, nz) = (dims.0 as i32, dims.1 as i32, dims.2 as i32);

    unsafe { vkfft_make_plan_r2c_c2r_many(ctx.handle, nx, ny, nz) }
}
