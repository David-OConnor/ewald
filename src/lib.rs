#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::excessive_precision)]
// #![feature(core_float_math)] When stable

//! For Smooth-Particle-Mesh Ewald; a standard approximation for Coulomb forces in MD.
//! We use this to handle periodic boundary conditions (e.g. of the solvent) properly.
//! See the Readme for details. The API is split into
//! two main parts: A standalone function to calculate short-range force, and
//! a struct with forces methods for long-range reciprical forces.

extern crate core;

use std::f32::consts::{PI, TAU};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
mod cuda_ffi;
#[cfg(feature = "cuda")]
mod vk_fft;

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaModule, CudaStream, DevicePtr, sys::CUstream};
use lin_alg::f32::Vec3;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use statrs::function::erf::{erf, erfc};

const SQRT_PI: f32 = 1.7724538509055159;
const INV_SQRT_PI: f32 = 1. / SQRT_PI;
const TWO_INV_SQRT_PI: f32 = 2. / SQRT_PI;

// The cardinal B-spline of order used to spread point charges to the mesh and
// to interpolate forces back. 4 is the safe default, and should probably not be changed.
// 3 is faster, but less accurate. 5+ are slow.
const SPLINE_ORDER: usize = 4;

type Complex_ = Complex<f32>;

/// Initialize this once for the application, or once per step.
pub struct PmeRecip {
    nx: usize,
    ny: usize,
    nz: usize,
    /// lx, ly and lz are box-lengths in real space.
    lx: f32,
    ly: f32,
    lz: f32,
    vol: f32,
    /// A tunable variable used in the splitting between short range and long range forces.
    /// A bigger α means more damping, and a smaller real-space contribution. (Cheaper real), but larger
    /// reciprocal load.
    pub alpha: f32,
    // Precomputed k-vectors and B-spline deconvolution |B(k)|^2
    kx: Vec<f32>,
    ky: Vec<f32>,
    kz: Vec<f32>,
    bmod2_x: Vec<f32>,
    bmod2_y: Vec<f32>,
    bmod2_z: Vec<f32>,
    /// For CPU FFTs
    planner: FftPlanner<f32>,
    #[cfg(feature = "cuda")]
    planner_gpu: *mut std::ffi::c_void,
    #[cfg(feature = "cuda")]
    planner_gpu_vk: *mut std::ffi::c_void,
    // todo: Do you want this? Or is it the same as nx, ny, nz?
    #[cfg(feature = "cuda")]
    plan_dims: (i32, i32, i32),
    #[cfg(feature = "cuda")]
    gpu_tables: Option<GpuTables>,
}

impl PmeRecip {
    pub fn new(n: (usize, usize, usize), l: (f32, f32, f32), alpha: f32) -> Self {
        assert!(n.0 >= 4 && n.1 >= 4 && n.2 >= 4);

        let vol = l.0 * l.1 * l.2;

        let kx = make_k_array(n.0, l.0);
        let ky = make_k_array(n.1, l.1);
        let kz = make_k_array(n.2, l.2);

        let bmod2_x = spline_bmod2_1d(n.0, SPLINE_ORDER);
        let bmod2_y = spline_bmod2_1d(n.1, SPLINE_ORDER);
        let bmod2_z = spline_bmod2_1d(n.2, SPLINE_ORDER);

        Self {
            nx: n.0,
            ny: n.1,
            nz: n.2,
            lx: l.0,
            ly: l.1,
            lz: l.2,
            vol,
            alpha,
            kx,
            ky,
            kz,
            bmod2_x,
            bmod2_y,
            bmod2_z,
            planner: FftPlanner::new(),
            #[cfg(feature = "cuda")]
            planner_gpu: std::ptr::null_mut(),
            #[cfg(feature = "cuda")]
            planner_gpu_vk: std::ptr::null_mut(),
            #[cfg(feature = "cuda")]
            plan_dims: (0, 0, 0),
            #[cfg(feature = "cuda")]
            gpu_tables: None,
            // ..Default::default()
        }
    }

    fn spread_charges(&self, pos: &[Vec3], q: &[f32], rho: &mut [Complex_]) {
        let nxny = self.nx * self.ny;
        for (r, &qi) in pos.iter().zip(q.iter()) {
            let sx = r.x / self.lx * self.nx as f32;
            let sy = r.y / self.ly * self.ny as f32;
            let sz = r.z / self.lz * self.nz as f32;

            let (ix0, wx) = bspline4_weights(sx);
            let (iy0, wy) = bspline4_weights(sy);
            let (iz0, wz) = bspline4_weights(sz);

            for a in 0..SPLINE_ORDER {
                let ix = wrap(ix0 + a as isize, self.nx);
                let wxa = wx[a];

                for b in 0..SPLINE_ORDER {
                    let iy = wrap(iy0 + b as isize, self.ny);
                    let wxy = wxa * wy[b];

                    for c in 0..SPLINE_ORDER {
                        let iz = wrap(iz0 + c as isize, self.nz);
                        let idx = iz * nxny + iy * self.nx + ix;
                        rho[idx].re += qi * wxy * wz[c];
                    }
                }
            }
        }
    }

    /// Compute reciprocal-space forces on all positions. Positions must be in the primary box [0,L] per axis.
    /// todo: Is there a way to exclude static targets, or must all sources be targets?
    pub fn forces(&mut self, posits: &[Vec3], q: &[f32]) -> (Vec<Vec3>, f32) {
        assert_eq!(posits.len(), q.len());

        let n_pts = self.nx * self.ny * self.nz;
        let mut rho = vec![Complex::<f32>::new(0.0, 0.0); n_pts];
        self.spread_charges(posits, q, &mut rho);

        fft3_inplace(
            &mut rho,
            (self.nx, self.ny, self.nz),
            &mut self.planner,
            true,
        );

        // Hoist shared refs so the parallel closure doesn't borrow &mut self
        let (nx, ny, _nz) = (self.nx, self.ny, self.nz);
        let (kx, ky, kz) = (&self.kx, &self.ky, &self.kz);
        let (bx, by, bz) = (&self.bmod2_x, &self.bmod2_y, &self.bmod2_z);
        let (vol, alpha) = (self.vol, self.alpha);

        // eAk are per-dimension phase factors.
        // Apply influence function to get φ(k) from ρ(k), then make E(k)=i k φ(k)
        let mut exk = vec![Complex::<f32>::new(0.0, 0.0); n_pts];
        let mut eyk = vec![Complex::<f32>::new(0.0, 0.0); n_pts];
        let mut ezk = vec![Complex::<f32>::new(0.0, 0.0); n_pts];

        let mut energy: f64 = exk
            .par_iter_mut()
            .zip(eyk.par_iter_mut())
            .zip(ezk.par_iter_mut())
            .enumerate()
            .map(|(idx, ((ex, ey), ez))| {
                let ix = idx % nx;
                let iy = (idx / nx) % ny;
                let iz = idx / (nx * ny);

                let kxv = kx[ix];
                let kyv = ky[iy];
                let kzv = kz[iz];

                let k2 = kxv.mul_add(kxv, kyv.mul_add(kyv, kzv * kzv));
                if k2 == 0.0 {
                    *ex = Complex::new(0.0, 0.0);
                    *ey = Complex::new(0.0, 0.0);
                    *ez = Complex::new(0.0, 0.0);
                    return 0.;
                }

                let bmod2 = bx[ix] * by[iy] * bz[iz];
                if bmod2 <= 1e-10 {
                    *ex = Complex::new(0.0, 0.0);
                    *ey = Complex::new(0.0, 0.0);
                    *ez = Complex::new(0.0, 0.0);
                    return 0.;
                }

                // φ(k) = G(k) ρ(k) with B-spline deconvolution
                let ghat = (2.0 * TAU / vol) * (-k2 / (4.0 * alpha * alpha)).exp() / (k2 * bmod2);
                let phi_k = rho[idx] * ghat;

                // E(k) = i k φ(k)
                // todo: QC if you need to flip the sign here on kxv, kyv. (Rem the -)
                *ex = Complex::new(0.0, -kxv) * phi_k;
                *ey = Complex::new(0.0, -kyv) * phi_k;
                *ez = Complex::new(0.0, -kzv) * phi_k;

                // reciprocal-space energy density: (1/2) Re{ ρ*(k) φ(k) }
                // 0.5 * (rho[idx].conj() * phi_k).re as f64
                let re64 = (rho[idx].re as f64) * (phi_k.re as f64)
                    + (rho[idx].im as f64) * (phi_k.im as f64);
                0.5 * re64
            })
            .sum();

        let self_energy: f64 = -(self.alpha / SQRT_PI) as f64
            * q.iter().map(|&qi| (qi as f64) * (qi as f64)).sum::<f64>();
        energy += self_energy;

        // Note: These FFTs are the biggest time bottleneck.
        // let start = Instant::now();
        // Inverse FFT to real-space E grids
        fft3_inplace(
            &mut exk,
            (self.nx, self.ny, self.nz),
            &mut self.planner,
            false,
        );
        fft3_inplace(
            &mut eyk,
            (self.nx, self.ny, self.nz),
            &mut self.planner,
            false,
        );
        fft3_inplace(
            &mut ezk,
            (self.nx, self.ny, self.nz),
            &mut self.planner,
            false,
        );

        // Interpolate E back to particles with the same B-spline weights; F = q E
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let lx = self.lx;
        let ly = self.ly;
        let lz = self.lz;

        let f = posits
            .par_iter()
            .enumerate()
            .map(|(i, &r)| {
                let (ix0, wx) = bspline4_weights(r.x / lx * nx as f32);
                let (iy0, wy) = bspline4_weights(r.y / ly * ny as f32);
                let (iz0, wz) = bspline4_weights(r.z / lz * nz as f32);

                let mut ex = 0.0f64;
                let mut ey = 0.0f64;
                let mut ez = 0.0f64;

                for a in 0..4 {
                    let ix = wrap(ix0 + a as isize, nx);
                    let wxa = wx[a];

                    for b in 0..4 {
                        let iy = wrap(iy0 + b as isize, ny);
                        let wyb = wy[b];
                        let wxy = wxa * wyb;

                        for c in 0..4 {
                            let iz = wrap(iz0 + c as isize, nz);
                            let w = (wxy * wz[c]) as f64;
                            let idx = iz * (nx * ny) + iy * nx + ix;

                            ex = w.mul_add(exk[idx].re as f64, ex);
                            ey = w.mul_add(eyk[idx].re as f64, ey);
                            ez = w.mul_add(ezk[idx].re as f64, ez);
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

        // Interpolate φ to particles and compute ½ Σ q_i φ_i
        // let phi_vals = self.interpolate_scalar(posits, &phi_k);
        // let energy = 0.5 * q.iter().zip(phi_vals.iter()).map(|(qi, phi)| qi * phi).sum::<f64>();

        (f, energy as f32)
    }

    #[cfg(feature = "cuda")]
    pub fn forces_gpu(
        &mut self,
        stream: &Arc<CudaStream>,
        _module: &Arc<CudaModule>,
        pos: &[Vec3],
        q: &[f32],
    ) -> (Vec<Vec3>, f32) {
        assert_eq!(pos.len(), q.len());

        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let n_real = nx * ny * nz;
        let n_cmplx = nx * ny * (nz / 2 + 1); // half-spectrum length
        let complex_len = n_cmplx * 2; // (re,im) interleaved

        // Contiguous real buffer: [ex | ey | ez]
        let ex_ey_ez_d: CudaSlice<f32> = stream.alloc_zeros(3 * n_real).unwrap();
        let (base_r_ptr, _) = ex_ey_ez_d.device_ptr(stream);
        let base_r = base_r_ptr as usize;
        let stride_r_bytes = n_real * std::mem::size_of::<f32>();
        let ex_ptr = base_r as *mut std::ffi::c_void;
        let ey_ptr = (base_r + stride_r_bytes) as *mut std::ffi::c_void;
        let ez_ptr = (base_r + 2 * stride_r_bytes) as *mut std::ffi::c_void;

        // Contiguous complex buffer: [exk | eyk | ezk]
        let exeyezk_d: CudaSlice<f32> = stream.alloc_zeros(3 * complex_len).unwrap();
        let (base_ptr, _) = exeyezk_d.device_ptr(stream);
        let base = base_ptr as usize;
        let stride_bytes = complex_len * std::mem::size_of::<f32>();
        let exk_ptr = base as *mut std::ffi::c_void;
        let eyk_ptr = (base + stride_bytes) as *mut std::ffi::c_void;
        let ezk_ptr = (base + 2 * stride_bytes) as *mut std::ffi::c_void;

        self.ensure_gpu_plan(stream);
        self.ensure_gpu_tables(stream);
        let cu_stream = stream.cu_stream() as *mut std::ffi::c_void;

        // Device tables
        let tabs = self.gpu_tables.as_ref().unwrap();
        let (kx_ptr, _) = tabs.kx.device_ptr(stream);
        let (ky_ptr, _) = tabs.ky.device_ptr(stream);
        let (kz_ptr, _) = tabs.kz.device_ptr(stream);
        let (bx_ptr, _) = tabs.bx.device_ptr(stream);
        let (by_ptr, _) = tabs.by.device_ptr(stream);
        let (bz_ptr, _) = tabs.bz.device_ptr(stream);

        // H2D: positions & charges (flattened)
        let pos_flat: Vec<f32> = pos.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let pos_d: CudaSlice<f32> = stream.memcpy_stod(&pos_flat).unwrap();
        let q_d: CudaSlice<f32> = stream.memcpy_stod(q).unwrap();
        let (pos_ptr, _) = pos_d.device_ptr(stream);
        let (q_ptr, _) = q_d.device_ptr(stream);

        // rho_real on device
        let rho_real_d: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
        let (rho_real_ptr, _) = rho_real_d.device_ptr(stream);

        // Scatter on GPU
        unsafe {
            cuda_ffi::spme_scatter_rho_4x4x4_launch(
                pos_ptr as *const _,
                q_ptr as *const _,
                rho_real_ptr as *mut _,
                pos.len() as i32,
                nx as i32,
                ny as i32,
                nz as i32,
                self.lx,
                self.ly,
                self.lz,
                cu_stream,
            );
        }

        // rho(k) (half-spectrum) on device
        let rho_k_d: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();
        let (rho_k_ptr, _) = rho_k_d.device_ptr(stream);

        // Make/refresh R2C/C2R-many plan
        if self.planner_gpu.is_null() || self.plan_dims != (nx as i32, ny as i32, nz as i32) {
            unsafe {
                if !self.planner_gpu.is_null() {
                    cuda_ffi::spme_destroy_plan_r2c_c2r_many(self.planner_gpu);
                }
                self.planner_gpu = cuda_ffi::spme_make_plan_r2c_c2r_many(
                    nx as i32, ny as i32, nz as i32, cu_stream,
                );
                self.plan_dims = (nx as i32, ny as i32, nz as i32);
            }
        }

        // Forward FFT on GPU: real -> half complex
        unsafe {
            cuda_ffi::spme_exec_forward_r2c(
                self.planner_gpu,
                rho_real_ptr as *mut _,
                rho_k_ptr as *mut _,
            );

            // Apply G(k) and gradient to get Exk/Eyk/Ezk
            cuda_ffi::spme_apply_ghat_and_grad_launch(
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
            cuda_ffi::spme_exec_inverse_ExEyEz_c2r(
                self.planner_gpu,
                exk_ptr as *mut _,
                eyk_ptr as *mut _,
                ezk_ptr as *mut _,
                ex_ptr as *mut _,
                ey_ptr as *mut _,
                ez_ptr as *mut _,
            );

            // Scale by 1/N once here
            cuda_ffi::spme_scale_ExEyEz_after_c2r(
                ex_ptr, ey_ptr, ez_ptr, nx as i32, ny as i32, nz as i32, cu_stream,
            );
        }

        // Gather forces F = q * E
        let f_d: CudaSlice<f32> = stream.alloc_zeros(pos.len() * 3).unwrap();
        let (f_ptr, _) = f_d.device_ptr(stream);
        let inv_n = 1.0f32 / (nx as f32 * ny as f32 * nz as f32);
        unsafe {
            cuda_ffi::spme_gather_forces_to_atoms_launch(
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
                self.lx,
                self.ly,
                self.lz,
                inv_n,
                cu_stream,
            );
        }

        // Energy (half spectrum)
        let threads = 256i32;
        let blocks = {
            let n = n_cmplx as i32;
            (n + threads - 1) / threads
        };
        let partial_d: CudaSlice<f64> = stream.alloc_zeros(blocks as usize).unwrap();
        let (partial_ptr, _) = partial_d.device_ptr(stream);
        unsafe {
            cuda_ffi::spme_energy_half_spectrum_launch(
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
                threads,
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

/// k-array for an orthorhombic cell; FFT index convention → physical wavevector.
fn make_k_array(n: usize, L: f32) -> Vec<f32> {
    let tau_div_l = TAU / L;

    let mut out = vec![0.0; n];
    let n_half = n / 2;

    for (i, out_) in out.iter_mut().enumerate() {
        // map 0..n-1 -> signed frequency bins: 0,1,2,...,n/2,-(n/2-1),..., -1
        let fi = if i <= n_half {
            i as isize
        } else {
            (i as isize) - (n as isize)
        };
        *out_ = tau_div_l * (fi as f32);
    }
    out
}

/// |B(k)|^2 for B-spline of order m (PME deconvolution).
/// Use signed/wrapped index distance to 0 to avoid over-amplifying near Nyquist.
fn spline_bmod2_1d(n: usize, m: usize) -> Vec<f32> {
    assert!(m >= 1);
    let mut v = vec![0.0; n];

    for (i, val) in v.iter_mut().enumerate() {
        let k = i.min(n - i);

        if k == 0 {
            *val = 1.0; // sinc(0) = 1
        } else {
            let t = PI * (k as f32) / (n as f32); // = |ω|/2 with ω=2πk/n
            let s = t.sin() / t; // sinc(|ω|/2)
            *val = s.powi((m as i32) * 2); // |B(ω)|^2 = sinc^(2m)
        }
    }
    v
}

/// Cubic B-spline weights for 4 neighbors; returns starting index and 4 weights.
/// Input s is in grid units (0..n), arbitrary real; we wrap indices to the grid.
fn bspline4_weights(s: f32) -> (isize, [f32; SPLINE_ORDER]) {
    let sfloor = s.floor();
    let u = s - sfloor; // fractional part in [0,1)
    let i0 = sfloor as isize - 1; // left-most point of 4-support

    // Cardinal cubic B-spline weights (order 4)
    let u2 = u * u;
    // let u3 = u2 * u;
    let u3 = u2.mul_add(u, 0.0);

    let w0 = (1.0 - u).powi(3) / 6.0;
    let w1 = (3.0 * u3 - 6.0 * u2 + 4.0) / 6.0;
    let w2 = (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) / 6.0;
    let w3 = u3 / 6.0;

    (i0, [w0, w1, w2, w3])
}

#[inline]
fn wrap(i: isize, n: usize) -> usize {
    let n_isize = n as isize;
    let mut v = i % n_isize;
    if v < 0 {
        v += n_isize;
    }
    v as usize
}

/// Minimal, cache-friendly 3D FFT using rustfft 1D plans along each axis.
/// dir=true => forward; dir=false => inverse (and rustfft handles scaling=1).
fn fft3_inplace(
    data: &mut [Complex_],
    dims: (usize, usize, usize),
    planner: &mut FftPlanner<f32>,
    forward: bool,
) {
    let (nx, ny, nz) = dims;
    let len = nx * ny * nz;
    debug_assert_eq!(data.len(), len);

    let fft_x = if forward {
        planner.plan_fft_forward(nx)
    } else {
        planner.plan_fft_inverse(nx)
    };
    let fft_y = if forward {
        planner.plan_fft_forward(ny)
    } else {
        planner.plan_fft_inverse(ny)
    };
    let fft_z = if forward {
        planner.plan_fft_forward(nz)
    } else {
        planner.plan_fft_inverse(nz)
    };

    // X transforms (contiguous)
    for iz in 0..nz {
        for iy in 0..ny {
            let row = iz * (nx * ny) + iy * nx;
            let slice = &mut data[row..row + nx];
            fft_x.process(slice);
        }
    }

    // Y transforms (strided by nx)
    {
        let mut tmp = vec![Complex::<f32>::new(0.0, 0.0); ny];
        for iz in 0..nz {
            for ix in 0..nx {
                // gather
                for (j, iy) in (0..ny).enumerate() {
                    tmp[j] = data[iz * (nx * ny) + iy * nx + ix];
                }
                // fft
                fft_y.process(&mut tmp);
                // scatter
                for (j, iy) in (0..ny).enumerate() {
                    data[iz * (nx * ny) + iy * nx + ix] = tmp[j];
                }
            }
        }
    }

    // Z transforms (strided by nx*ny)
    {
        let mut tmp = vec![Complex::<f32>::new(0.0, 0.0); nz];
        for iy in 0..ny {
            for ix in 0..nx {
                // gather
                for (k, iz) in (0..nz).enumerate() {
                    tmp[k] = data[iz * (nx * ny) + iy * nx + ix];
                }
                // fft
                fft_z.process(&mut tmp);
                // scatter
                for (k, iz) in (0..nz).enumerate() {
                    data[iz * (nx * ny) + iy * nx + ix] = tmp[k];
                }
            }
        }
    }

    // rustfft inverse is unnormalized; many MD codes keep that and balance elsewhere.
    // If you prefer normalized inverse, scale here by 1/(nx*ny*nz) after inverse passes.
    if !forward {
        let scale = 1.0 / (len as f32);
        for v in data.iter_mut() {
            v.re *= scale;
            v.im *= scale;
        }
    }
}

/// We use this to smoothly switch between short-range and long-range (reciprical) forces.
/// todo: Hard cut off, vice taper, for now.
fn _taper(s: f64) -> (f64, f64) {
    // s in [0,1]; returns (S, dS/dr * dr/ds) but we’ll just return S and dS/ds here.
    // Quintic: S = 1 - 10 s^3 + 15 s^4 - 6 s^5;  dS/ds = -30 s^2 + 60 s^3 - 30 s^4
    let s2 = s * s;
    let s3 = s2 * s;
    let s4 = s3 * s;
    let s5 = s4 * s;
    let s_val = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5;
    let ds = -30.0 * s2 + 60.0 * s3 - 30.0 * s4;

    (s_val, ds)
}

/// We use this for short-range Coulomb forces on the CPU, as part of SPME.
/// `cutoff_dist` is the distance, in Å, we switch between short-range, and long-range reciprical
/// forces. 10Å is a good default. 0.35Å for α is a good default for a custoff of 10Å.
///
/// This assumes diff (and dir) is in order tgt - src.
/// Also returns potential energy.
///
/// `dir` must be a unit vector.
pub fn force_coulomb_short_range(
    dir: Vec3,
    dist: f32,
    // Included to share between this and Lennard Jones.
    inv_dist: f32,
    q_0: f32,
    q_1: f32,
    cutoff_dist: f32,
    α: f32,
) -> (Vec3, f32) {
    // Outside the taper region; return 0. (All the force is handled in the long-range region.)
    if dist > cutoff_dist {
        return (Vec3::new_zero(), 0.);
    }

    let α_r = α * dist;
    let erfc_term = erfc(α_r as f64) as f32;
    let charge_term = q_0 * q_1;

    let energy = charge_term * inv_dist * erfc_term;

    let exp_term = (-α_r * α_r).exp();

    let force_mag = charge_term
        * (erfc_term * inv_dist * inv_dist + 2.0 * α * exp_term * INV_SQRT_PI * inv_dist);

    (dir * force_mag, energy)
}

// // todo: Update this to reflect your changes to the algo above that apply tapering.
// pub fn force_coulomb_ewald_real_x8(
//     dir: Vec3x8,
//     r: f64x8,
//     qi: f64x8,
//     qj: f64x8,
//     α: f64x8,
// ) -> Vec3x8 {
//     // F = q_i q_j [ erfc(αr)/r² + 2α/√π · e^(−α²r²)/r ]  · 4πϵ0⁻¹  · r̂
//     let qfac = qi * qj;
//     let inv_r = f64x8::splat(1.) / r;
//     let inv_r2 = inv_r * inv_r;
//
//     // let erfc_term = erfc(alpha * r);
//     let erfc_term = f64x8::splat(0.); // todo temp: Figure how how to do erfc with SIMD.
//
//     // todo: Figure out how to do exp with SIMD. Probably need powf in lin_alg
//     // let exp_term = (-alpha * alpha * r * r).exp();
//     // let exp_term = f64x8::splat(E).pow(-alpha * alpha * r * r);
//     let exp_term = f64x8::splat(1.); // todo temp
//
//     let force_mag =
//         qfac * (erfc_term * inv_r2 + f64x8::splat(2.) * α * exp_term / (f64x8::splat(SQRT_PI) * r));
//
//     dir * force_mag
// }

/// Useful for scaling corrections, e.g. 1-4 exclusions in AMBER.
pub fn ewald_comp_force(dir: Vec3, r: f32, qi: f32, qj: f32, alpha: f32) -> Vec3 {
    // Complement of the real-space Ewald kernel; this is what “belongs” to reciprocal.
    let qfac = qi * qj;
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;

    let ar = alpha * r;
    // todo: Mul_add when it's stable.
    let fmag = qfac
        * (erf(ar as f64) as f32 * inv_r2 - (alpha * TWO_INV_SQRT_PI) * (-ar * ar).exp() * inv_r);
    // let fmag = qfac * (mul_add(erf(ar as f64) as f32, inv_r2,  -(alpha * TWO_INV_SQRT_PI)) * (-ar * ar).exp() * inv_r);
    dir * fmag
}
