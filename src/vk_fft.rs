use std::{ffi::c_void, mem::size_of, ptr::null_mut, sync::Arc};

use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, sys::CUstream};
use lin_alg::f32::Vec3;

use crate::{
    GpuTables, PmeRecip, SQRT_PI, bspline4_weights,
    cufft::{
        spme_gather_forces_to_atoms_launch, spme_scale_ExEyEz_after_c2r,
        spme_scatter_rho_4x4x4_launch,
    },
    dev_ptr, dev_ptr_mut, wrap,
};

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
    pub fn vk_spme_apply_ghat_and_grad(
        ctx: *mut c_void,
        rho_k: *mut c_void,
        exk: *mut c_void,
        eyk: *mut c_void,
        ezk: *mut c_void,
        kx: *mut c_void,
        ky: *mut c_void,
        kz: *mut c_void,
        bx: *mut c_void,
        by: *mut c_void,
        bz: *mut c_void,
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

    pub fn vk_spme_energy_half_spectrum_sum(
        ctx: *mut c_void,
        rho_k: *mut c_void,
        kx: *mut c_void,
        ky: *mut c_void,
        kz: *mut c_void,
        bx: *mut c_void,
        by: *mut c_void,
        bz: *mut c_void,
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
    // todo: This may be very slow, i.e. you are making the same mistakes as your initial
    // todo: cuFFT sim, wherein you keep transfering data to and from the GPU instead
    // todo of keeping it there.
    pub fn forces_gpu(
        &mut self,
        // todo: Do we still need VkContext if we are using CudaStream?
        ctx: &Arc<VkContext>,
        stream: &Arc<CudaStream>,
        pos: &[Vec3],
        q: &[f32],
    ) -> (Vec<Vec3>, f32) {
        if self.gpu_tables.is_none() {
            // First run
            let k = (&self.kx, &self.ky, &self.kz);
            let bmod2 = (&self.bmod2_x, &self.bmod2_y, &self.bmod2_z);

            self.planner_gpu = create_gpu_plan(self.plan_dims, ctx);
            // self.gpu_tables = Some(GpuTables::new(k, bmod2, ctx));
            self.gpu_tables = Some(GpuTables::new(k, bmod2, stream));
        }

        assert_eq!(pos.len(), q.len());
        let (lx, ly, lz) = self.box_dims;
        let (nx, ny, nz) = self.plan_dims;

        let n_real = nx * ny * nz;
        let n_cmplx = nx * ny * (nz / 2 + 1); // half-spectrum length
        let complex_len = n_cmplx * 2; // interleaved re,im

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

        // real grids (ex|ey|ez) and complex (exk|eyk|ezk)
        let ex_ey_ez_d: CudaSlice<f32> = stream.alloc_zeros(3 * n_real).unwrap();
        let exeyezk_d: CudaSlice<f32> = stream.alloc_zeros(3 * complex_len).unwrap();
        let rho_real_d: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
        let rho_k_d: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();

        // scatter on device (same kernel as cuFFT)
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

        // // todo: Why is this a real scatter instead of a complex scatter like
        // // todo in the GPU branch?
        // let mut rho_real = vec![0.; n_real];
        // self.spread_charges_real(pos, q, &mut rho_real);

        // forward R2C
        unsafe {
            vkfft_exec_forward_r2c(
                self.planner_gpu,
                dev_ptr(&rho_real_d, stream),
                dev_ptr(&rho_k_d, stream),
            );

            // exk/eyk/ezk slices out of exeyezk_d
            let (base_ptr, _) = exeyezk_d.device_ptr(stream);
            let base = base_ptr as usize;
            let stride = complex_len * std::mem::size_of::<f32>();
            let exk = base as *mut c_void;
            let eyk = (base + stride) as *mut c_void;
            let ezk = (base + 2 * stride) as *mut c_void;

            // apply G(k), grad on device
            vk_spme_apply_ghat_and_grad(
                ctx.handle,
                dev_ptr(&rho_k_d, stream),
                exk,
                eyk,
                ezk,
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
            );

            // inverse C2R (batched or three calls)
            let (base_r_ptr, _) = ex_ey_ez_d.device_ptr(stream);
            let base_r = base_r_ptr as usize;
            let stride_r = n_real * std::mem::size_of::<f32>();
            let ex = base_r as *mut c_void;
            let ey = (base_r + stride_r) as *mut c_void;
            let ez = (base_r + 2 * stride_r) as *mut c_void;

            // if your vk plan is "many", add a wrapper; else call 3x:
            vkfft_exec_inverse_c2r(self.planner_gpu, exk, ex);
            vkfft_exec_inverse_c2r(self.planner_gpu, eyk, ey);
            vkfft_exec_inverse_c2r(self.planner_gpu, ezk, ez);

            // same scaling kernel as cuFFT
            spme_scale_ExEyEz_after_c2r(
                ex,
                ey,
                ez,
                nx as i32,
                ny as i32,
                nz as i32,
                stream.cu_stream() as *mut _,
            );
        }

        // gather on device (same as cuFFT)
        let f_d: CudaSlice<f32> = stream.alloc_zeros(pos.len() * 3).unwrap();
        let inv_n = 1.0f32 / (nx as f32 * ny as f32 * nz as f32);
        unsafe {
            spme_gather_forces_to_atoms_launch(
                dev_ptr(&pos_d, stream),
                dev_ptr(&ex_ey_ez_d, stream), // ex at offset 0
                (ex_ey_ez_d.as_device_ptr().0 as usize + n_real * size_of::<f32>()) as *const _,
                (ex_ey_ez_d.as_device_ptr().0 as usize + 2 * n_real * size_of::<f32>()) as *const _,
                dev_ptr(&q_d, stream),
                dev_ptr(&f_d, stream),
                pos.len() as i32,
                nx as i32,
                ny as i32,
                nz as i32,
                self.box_dims.0,
                self.box_dims.1,
                self.box_dims.2,
                inv_n,
                stream.cu_stream() as *mut _,
            );
        }

        // energy on device (keep your vk reduction)
        let energy = unsafe {
            vk_spme_energy_half_spectrum_sum(
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
