use std::{ffi::c_void, sync::Arc};

use lin_alg::f32::Vec3;

use crate::{PmeRecip, SQRT_PI, bspline4_weights, wrap};

#[repr(C)]
pub struct VkContext {
    pub handle: *mut c_void, // opaque (created in C)
}

#[link(name = "vk_fft")] // built from vk_fft.c via build.rs
unsafe extern "C" {
    // plan lifecycle
    pub fn vkfft_make_plan_r2c_c2r_many(ctx: *mut c_void, nx: i32, ny: i32, nz: i32)
    -> *mut c_void;
    pub fn vkfft_destroy_plan(plan: *mut c_void);

    // device memory helpers
    pub fn vk_alloc_and_upload(ctx: *mut c_void, host_src: *const u8, nbytes: u64) -> *mut c_void;
    pub fn vk_alloc_zeroed(ctx: *mut c_void, nbytes: u64) -> *mut c_void;
    pub fn vk_download(ctx: *mut c_void, dev_buf: *mut c_void, host_dst: *mut u8, nbytes: u64);
    pub fn vk_free(ctx: *mut c_void, dev_buf: *mut c_void);

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
    pub fn vk_destroy_context(ctx: *mut c_void);
}

impl PmeRecip {
    // vkFFT plan handle + cached dims
    pub(crate) fn vk_plan_ptr(&self) -> *mut c_void {
        self.planner_gpu_vk
    }
}

pub(crate) struct GpuTablesVk {
    kx: *mut c_void,
    ky: *mut c_void,
    kz: *mut c_void,
    bx: *mut c_void,
    by: *mut c_void,
    bz: *mut c_void,
}

impl Default for GpuTablesVk {
    fn default() -> Self {
        Self {
            kx: std::ptr::null_mut(),
            ky: std::ptr::null_mut(),
            kz: std::ptr::null_mut(),
            bx: std::ptr::null_mut(),
            by: std::ptr::null_mut(),
            bz: std::ptr::null_mut(),
        }
    }
}

impl PmeRecip {
    fn ensure_vkfft_plan(&mut self, ctx: &Arc<VkContext>) {
        let dims = (self.nx as i32, self.ny as i32, self.nz as i32);
        unsafe {
            if self.planner_gpu_vk.is_null() || self.plan_dims != dims {
                if !self.planner_gpu_vk.is_null() {
                    vkfft_destroy_plan(self.planner_gpu_vk);
                    self.planner_gpu_vk = std::ptr::null_mut();
                }
                self.planner_gpu_vk =
                    vkfft_make_plan_r2c_c2r_many(ctx.handle, dims.0, dims.1, dims.2);
                self.plan_dims = dims;
                self.gpu_tables_vk = None;
            }
        }
    }

    fn ensure_vk_tables(&mut self, ctx: &Arc<VkContext>) {
        if self.gpu_tables_vk.is_some() {
            return;
        }
        unsafe {
            let kx = vk_alloc_and_upload(
                ctx.handle,
                self.kx.as_ptr() as *const u8,
                (self.kx.len() * size_of::<f32>()) as u64,
            );
            let ky = vk_alloc_and_upload(
                ctx.handle,
                self.ky.as_ptr() as *const u8,
                (self.ky.len() * size_of::<f32>()) as u64,
            );
            let kz = vk_alloc_and_upload(
                ctx.handle,
                self.kz.as_ptr() as *const u8,
                (self.kz.len() * size_of::<f32>()) as u64,
            );
            let bx = vk_alloc_and_upload(
                ctx.handle,
                self.bmod2_x.as_ptr() as *const u8,
                (self.bmod2_x.len() * size_of::<f32>()) as u64,
            );
            let by = vk_alloc_and_upload(
                ctx.handle,
                self.bmod2_y.as_ptr() as *const u8,
                (self.bmod2_y.len() * size_of::<f32>()) as u64,
            );
            let bz = vk_alloc_and_upload(
                ctx.handle,
                self.bmod2_z.as_ptr() as *const u8,
                (self.bmod2_z.len() * size_of::<f32>()) as u64,
            );
            self.gpu_tables_vk = Some(GpuTablesVk {
                kx,
                ky,
                kz,
                bx,
                by,
                bz,
            });
        }
    }
}

impl PmeRecip {
    pub fn forces_vkfft(
        &mut self,
        ctx: &Arc<VkContext>,
        pos: &[Vec3],
        q: &[f32],
    ) -> (Vec<Vec3>, f32) {
        assert_eq!(pos.len(), q.len());
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let n_real = nx * ny * nz;
        let n_cmplx = nx * ny * (nz / 2 + 1);
        let complex_len = n_cmplx * 2; // interleaved re,im

        self.ensure_vkfft_plan(ctx);
        self.ensure_vk_tables(ctx);
        let tabs = self.gpu_tables_vk.as_ref().unwrap();

        // Host scatter (keep it simple; you can port your GPU scatter later)
        let mut rho_real = vec![0f32; n_real];
        self.spread_charges_real(pos, q, &mut rho_real);

        unsafe {
            // Upload rho
            let rho_real_d = vk_alloc_and_upload(
                ctx.handle,
                rho_real.as_ptr() as *const u8,
                (n_real * size_of::<f32>()) as u64,
            );
            let rho_k_d = vk_alloc_zeroed(ctx.handle, (complex_len * size_of::<f32>()) as u64);

            // Exec forward R2C
            vkfft_exec_forward_r2c(self.planner_gpu_vk, rho_real_d, rho_k_d);

            // Allocate exk,eyk,ezk (complex half-spectra, interleaved)
            let exk_d = vk_alloc_zeroed(ctx.handle, (complex_len * size_of::<f32>()) as u64);
            let eyk_d = vk_alloc_zeroed(ctx.handle, (complex_len * size_of::<f32>()) as u64);
            let ezk_d = vk_alloc_zeroed(ctx.handle, (complex_len * size_of::<f32>()) as u64);

            // Apply G(k) and gradient: writes exk, eyk, ezk
            vk_spme_apply_ghat_and_grad(
                ctx.handle, rho_k_d, exk_d, eyk_d, ezk_d, tabs.kx, tabs.ky, tabs.kz, tabs.bx,
                tabs.by, tabs.bz, nx as i32, ny as i32, nz as i32, self.vol, self.alpha,
            );

            // Inverse C2R on each field → ex,ey,ez
            let ex_d = vk_alloc_zeroed(ctx.handle, (n_real * size_of::<f32>()) as u64);
            let ey_d = vk_alloc_zeroed(ctx.handle, (n_real * size_of::<f32>()) as u64);
            let ez_d = vk_alloc_zeroed(ctx.handle, (n_real * size_of::<f32>()) as u64);

            vkfft_exec_inverse_c2r(self.planner_gpu_vk, exk_d, ex_d);
            vkfft_exec_inverse_c2r(self.planner_gpu_vk, eyk_d, ey_d);
            vkfft_exec_inverse_c2r(self.planner_gpu_vk, ezk_d, ez_d);

            // Scale by 1/N
            vk_spme_scale_real_fields(ex_d, ey_d, ez_d, nx as i32, ny as i32, nz as i32);

            // Gather on host (again, simple; port to a compute kernel later if desired)
            let mut ex_host = vec![0f32; n_real];
            let mut ey_host = vec![0f32; n_real];
            let mut ez_host = vec![0f32; n_real];
            vk_download(
                ctx.handle,
                ex_d,
                ex_host.as_mut_ptr() as *mut u8,
                (n_real * size_of::<f32>()) as u64,
            );
            vk_download(
                ctx.handle,
                ey_d,
                ey_host.as_mut_ptr() as *mut u8,
                (n_real * size_of::<f32>()) as u64,
            );
            vk_download(
                ctx.handle,
                ez_d,
                ez_host.as_mut_ptr() as *mut u8,
                (n_real * size_of::<f32>()) as u64,
            );

            let f: Vec<Vec3> = pos
                .iter()
                .enumerate()
                .map(|(i, &r)| {
                    let (ix0, wx) = bspline4_weights(r.x / self.lx * nx as f32);
                    let (iy0, wy) = bspline4_weights(r.y / self.ly * ny as f32);
                    let (iz0, wz) = bspline4_weights(r.z / self.lz * nz as f32);
                    let mut ex = 0.0f64;
                    let mut ey = 0.0f64;
                    let mut ez = 0.0f64;
                    let nxny = nx * ny;
                    for a in 0..4 {
                        let ix = wrap(ix0 + a as isize, nx);
                        let wxa = wx[a];
                        for b in 0..4 {
                            let iy = wrap(iy0 + b as isize, ny);
                            let wxy = wxa * wy[b];
                            for c in 0..4 {
                                let iz = wrap(iz0 + c as isize, nz);
                                let w = (wxy * wz[c]) as f64;
                                let idx = iz * nxny + iy * nx + ix;
                                ex = w.mul_add(ex_host[idx] as f64, ex);
                                ey = w.mul_add(ey_host[idx] as f64, ey);
                                ez = w.mul_add(ez_host[idx] as f64, ez);
                            }
                        }
                    }
                    let qi = q[i] as f64;
                    Vec3 {
                        x: (ex * qi) as f32,
                        y: (ey * qi) as f32,
                        z: (ez * qi) as f32,
                    }
                })
                .collect();

            // Energy (half-spectrum) on device; returns sum
            let energy = vk_spme_energy_half_spectrum_sum(
                ctx.handle, rho_k_d, tabs.kx, tabs.ky, tabs.kz, tabs.bx, tabs.by, tabs.bz,
                nx as i32, ny as i32, nz as i32, self.vol, self.alpha,
            ) as f32;

            // Self-energy
            let self_energy: f64 = -(self.alpha as f64 / SQRT_PI as f64)
                * q.iter().map(|&qi| (qi as f64) * (qi as f64)).sum::<f64>();
            let energy = (energy as f64 + self_energy) as f32;

            // Free
            vk_free(ctx.handle, rho_real_d);
            vk_free(ctx.handle, rho_k_d);
            vk_free(ctx.handle, exk_d);
            vk_free(ctx.handle, eyk_d);
            vk_free(ctx.handle, ezk_d);
            vk_free(ctx.handle, ex_d);
            vk_free(ctx.handle, ey_d);
            vk_free(ctx.handle, ez_d);

            (f, energy)
        }
    }

    /// todo: Compare this to spread_charges (cplx version) in lib.rs. Figure out why you need
    /// todo this for vkfft, but use the cplx version there.
    fn spread_charges_real(&self, pos: &[Vec3], q: &[f32], rho: &mut [f32]) {
        let nxny = self.nx * self.ny;
        for (r, &qi) in pos.iter().zip(q.iter()) {
            let sx = r.x / self.lx * self.nx as f32;
            let sy = r.y / self.ly * self.ny as f32;
            let sz = r.z / self.lz * self.nz as f32;

            let (ix0, wx) = bspline4_weights(sx);
            let (iy0, wy) = bspline4_weights(sy);
            let (iz0, wz) = bspline4_weights(sz);

            for a in 0..4 {
                let ix = wrap(ix0 + a as isize, self.nx);
                let wxa = wx[a];
                for b in 0..4 {
                    let iy = wrap(iy0 + b as isize, self.ny);
                    let wxy = wxa * wy[b];
                    for c in 0..4 {
                        let iz = wrap(iz0 + c as isize, self.nz);
                        let idx = iz * nxny + iy * self.nx + ix;
                        rho[idx] += qi * wxy * wz[c];
                    }
                }
            }
        }
    }
}
