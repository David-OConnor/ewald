use std::sync::Arc;

use crate::{PmeRecip, SQRT_PI};

impl PmeRecip {
    // vkFFT plan handle + cached dims
    pub(crate) fn vk_plan_ptr(&self) -> *mut std::ffi::c_void {
        self.planner_gpu_vk
    }
}

struct VkGpuTables {
    kx: *mut std::ffi::c_void,
    ky: *mut std::ffi::c_void,
    kz: *mut std::ffi::c_void,
    bx: *mut std::ffi::c_void,
    by: *mut std::ffi::c_void,
    bz: *mut std::ffi::c_void,
}

impl Default for VkGpuTables {
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
            if self.planner_gpu_vk.is_null() || self.vk_plan_dims != dims {
                if !self.planner_gpu_vk.is_null() {
                    vkfft_destroy_plan(self.planner_gpu_vk);
                    self.planner_gpu_vk = std::ptr::null_mut();
                }
                self.planner_gpu_vk =
                    vkfft_make_plan_r2c_c2r_many(ctx.handle, dims.0, dims.1, dims.2);
                self.vk_plan_dims = dims;
                self.vk_tables = None;
            }
        }
    }

    fn ensure_vk_tables(&mut self, ctx: &Arc<VkContext>) {
        if self.vk_tables.is_some() {
            return;
        }
        unsafe {
            let kx = vk_alloc_and_upload(
                ctx.handle,
                self.kx.as_ptr() as *const u8,
                (self.kx.len() * std::mem::size_of::<f32>()) as u64,
            );
            let ky = vk_alloc_and_upload(
                ctx.handle,
                self.ky.as_ptr() as *const u8,
                (self.ky.len() * std::mem::size_of::<f32>()) as u64,
            );
            let kz = vk_alloc_and_upload(
                ctx.handle,
                self.kz.as_ptr() as *const u8,
                (self.kz.len() * std::mem::size_of::<f32>()) as u64,
            );
            let bx = vk_alloc_and_upload(
                ctx.handle,
                self.bmod2_x.as_ptr() as *const u8,
                (self.bmod2_x.len() * std::mem::size_of::<f32>()) as u64,
            );
            let by = vk_alloc_and_upload(
                ctx.handle,
                self.bmod2_y.as_ptr() as *const u8,
                (self.bmod2_y.len() * std::mem::size_of::<f32>()) as u64,
            );
            let bz = vk_alloc_and_upload(
                ctx.handle,
                self.bmod2_z.as_ptr() as *const u8,
                (self.bmod2_z.len() * std::mem::size_of::<f32>()) as u64,
            );
            self.vk_tables = Some(VkGpuTables {
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

#[cfg(feature = "vkfft")]
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
        let tabs = self.vk_tables.as_ref().unwrap();

        // Host scatter (keep it simple; you can port your GPU scatter later)
        let mut rho_real = vec![0f32; n_real];
        self.spread_charges(
            pos,
            q,
            bytemuck::cast_slice_mut::<f32, Complex_>(&mut rho_real),
        );

        unsafe {
            // Upload rho
            let rho_real_d = vk_alloc_and_upload(
                ctx.handle,
                rho_real.as_ptr() as *const u8,
                (n_real * std::mem::size_of::<f32>()) as u64,
            );
            let rho_k_d = vk_alloc_zeroed(
                ctx.handle,
                (complex_len * std::mem::size_of::<f32>()) as u64,
            );

            // Exec forward R2C
            vkfft_exec_forward_r2c(self.planner_gpu_vk, rho_real_d, rho_k_d);

            // Allocate exk,eyk,ezk (complex half-spectra, interleaved)
            let exk_d = vk_alloc_zeroed(
                ctx.handle,
                (complex_len * std::mem::size_of::<f32>()) as u64,
            );
            let eyk_d = vk_alloc_zeroed(
                ctx.handle,
                (complex_len * std::mem::size_of::<f32>()) as u64,
            );
            let ezk_d = vk_alloc_zeroed(
                ctx.handle,
                (complex_len * std::mem::size_of::<f32>()) as u64,
            );

            // Apply G(k) and gradient: writes exk, eyk, ezk
            vk_spme_apply_ghat_and_grad(
                ctx.handle, rho_k_d, exk_d, eyk_d, ezk_d, tabs.kx, tabs.ky, tabs.kz, tabs.bx,
                tabs.by, tabs.bz, nx as i32, ny as i32, nz as i32, self.vol, self.alpha,
            );

            // Inverse C2R on each field → ex,ey,ez
            let ex_d = vk_alloc_zeroed(ctx.handle, (n_real * std::mem::size_of::<f32>()) as u64);
            let ey_d = vk_alloc_zeroed(ctx.handle, (n_real * std::mem::size_of::<f32>()) as u64);
            let ez_d = vk_alloc_zeroed(ctx.handle, (n_real * std::mem::size_of::<f32>()) as u64);

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
                (n_real * std::mem::size_of::<f32>()) as u64,
            );
            vk_download(
                ctx.handle,
                ey_d,
                ey_host.as_mut_ptr() as *mut u8,
                (n_real * std::mem::size_of::<f32>()) as u64,
            );
            vk_download(
                ctx.handle,
                ez_d,
                ez_host.as_mut_ptr() as *mut u8,
                (n_real * std::mem::size_of::<f32>()) as u64,
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
}
