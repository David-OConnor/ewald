#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::excessive_precision)]

//! For Smooth-Particle-Mesh Ewald; a standard approximation for Coulomb forces in MD.
//! We use this to handle periodic boundary conditions (e.g. of the solvent) properly.
//! See the Readme for details. The API is split into
//! two main parts: A standalone function to calculate short-range force, and
//! a struct with forces methods for long-range reciprical forces.

extern crate core;

use std::{
    arch::x86_64::{_CMP_LT_OQ, _mm256_blendv_ps, _mm256_cmp_ps, _mm256_set1_ps},
    f32::consts::{PI, TAU},
};
#[cfg(feature = "cuda")]
use std::{ffi::c_void, sync::Arc};

#[cfg(feature = "cuda")]
use cudarc::{
    driver::{CudaContext, CudaStream, DevicePtr},
    nvrtc::Ptx,
};

// We are for now, exposing
mod fft;

#[cfg(feature = "cufft")]
mod cufft;
#[cfg(feature = "vkfft")]
pub mod vk_fft;

#[cfg(feature = "cuda")]
mod gpu_shared;

use lin_alg::f32::Vec3;
#[cfg(target_arch = "x86_64")]
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use statrs::function::erf::{erf, erfc};

use crate::fft::{fft3d_c2r, fft3d_r2c};
#[cfg(feature = "cuda")]
use crate::gpu_shared::{GpuData, GpuTables, Kernels};

#[cfg(feature = "cuda")]
const PTX: &str = include_str!("../ewald.ptx");

const SQRT_PI: f32 = 1.7724538509055159;
const INV_SQRT_PI: f32 = 1. / SQRT_PI;
const TWO_INV_SQRT_PI: f32 = 2. / SQRT_PI;

// The cardinal B-spline of order used to spread point charges to the mesh and
// to interpolate forces back. 4 is the safe default, and should probably not be changed.
// 3 is faster, but less accurate. 5+ are slow.
const SPLINE_ORDER: usize = 4;

type Complex_ = Complex<f32>;

/// Initialize this once for the application, or once per step.
/// Note:
pub struct PmeRecip {
    /// Simultatoin box-lengths in real space.
    box_dims: (f32, f32, f32),
    /// FFT planner dimensions. B‑spline interpolation X, Y, Z. These should be based on
    /// box length, and mesh spacing. nz should be even.
    plan_dims: (usize, usize, usize),
    vol: f32,
    /// A tunable variable used in the splitting between short range and long range forces.
    /// A bigger α means more damping, and a smaller real-space contribution. (Cheaper real), but larger
    /// reciprocal load.
    pub alpha: f32,
    /// Precomputed k-vectors and B-spline deconvolution |B(k)|^2
    kx: Vec<f32>,
    ky: Vec<f32>,
    kz: Vec<f32>,
    bmod2_x: Vec<f32>,
    bmod2_y: Vec<f32>,
    bmod2_z: Vec<f32>,
    /// For CPU FFTs
    planner: FftPlanner<f32>,
    /// For GPU FFTs. None if compiling with GPU support, but there is a runtime problem,
    /// e.g. no CUDA on the system running it.
    #[cfg(feature = "cuda")]
    gpu_data: Option<GpuData>,
}

impl PmeRecip {
    pub fn new(
        #[cfg(feature = "cuda")] stream: Option<&Arc<CudaStream>>,
        plan_dims: (usize, usize, usize),
        l: (f32, f32, f32),
        alpha: f32,
    ) -> Self {
        assert!(plan_dims.0 >= 4 && plan_dims.1 >= 4 && plan_dims.2 >= 4);

        let vol = l.0 * l.1 * l.2;

        let kx = make_k_array(plan_dims.0, l.0);
        let ky = make_k_array(plan_dims.1, l.1);
        let kz = make_k_array(plan_dims.2, l.2);

        let bmod2_x = spline_bmod2_1d(plan_dims.0, SPLINE_ORDER);
        let bmod2_y = spline_bmod2_1d(plan_dims.1, SPLINE_ORDER);
        let bmod2_z = spline_bmod2_1d(plan_dims.2, SPLINE_ORDER);

        #[cfg(feature = "cuda")]
        let mut gpu_data = None;

        #[cfg(feature = "cuda")]
        if let Some(s) = stream {
            gpu_data = match CudaContext::new(0) {
                Ok(cuda_ctx) => {
                    let kernels = {
                        let module = cuda_ctx.load_module(Ptx::from_src(PTX)).unwrap();

                        let kernel_spread = module.load_function("spread_charges").unwrap();
                        let kernel_ghat = module.load_function("apply_ghat_and_grad").unwrap();
                        let kernel_gather = module.load_function("gather_forces_to_atoms").unwrap();
                        let kernel_half_spectrum =
                            module.load_function("energy_half_spectrum").unwrap();

                        Kernels {
                            kernel_spread,
                            kernel_ghat,
                            kernel_gather,
                            kernel_half_spectrum,
                        }
                    };

                    let gpu_tables = {
                        let k = (&kx, &ky, &kz);
                        let bmod2 = (&bmod2_x, &bmod2_y, &bmod2_z);

                        GpuTables::new(k, bmod2, s)
                    };

                    #[cfg(feature = "cufft")]
                    let planner_gpu = cufft::create_gpu_plan(plan_dims, s);

                    #[cfg(feature = "vkfft")]
                    let vk_ctx = Arc::new(vk_fft::VkContext::default());
                    #[cfg(feature = "vkfft")]
                    let planner_gpu = vk_fft::create_gpu_plan(plan_dims, &vk_ctx);

                    Some(GpuData {
                        planner_gpu,
                        gpu_tables,
                        kernels,
                        #[cfg(feature = "vkfft")]
                        vk_ctx,
                    })
                }
                Err(_) => None,
            };
        }

        // Note: planner_gpu and gpu_tables will be null/None until the first run. We
        // handle it this way since the forces fns have access to the stream or Context,
        // but this construct doens't, if we keep a unified constructor.
        Self {
            box_dims: l,
            plan_dims,
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
            gpu_data,
        }
    }

    /// CPU charge spreading. Z as the fast/contiguous axis.
    /// We place charges on a discrete grid, then apply a 3D FFT to the grid.
    fn spread_charges(&self, pos: &[Vec3], q: &[f32], rho: &mut [f32]) {
        let (nx, ny, nz) = self.plan_dims;
        let (lx, ly, lz) = self.box_dims;

        let nynz = ny * nz; // Z-fast layout helper

        for (r, &qi) in pos.iter().zip(q.iter()) {
            let sx = r.x / lx * nx as f32;
            let sy = r.y / ly * ny as f32;
            let sz = r.z / lz * nz as f32;

            let (ix0, wx) = bspline4_weights(sx);
            let (iy0, wy) = bspline4_weights(sy);
            let (iz0, wz) = bspline4_weights(sz);

            for a in 0..SPLINE_ORDER {
                let ix = wrap(ix0 + a as isize, nx);
                let wxa = wx[a];

                for b in 0..SPLINE_ORDER {
                    let iy = wrap(iy0 + b as isize, ny);
                    let wxy = wxa * wy[b];

                    // Z-fast base for this (ix, iy)
                    let base = ix * nynz + iy * nz;

                    for c in 0..SPLINE_ORDER {
                        let iz = wrap(iz0 + c as isize, nz);
                        let idx = base + iz; // contiguous over z
                        rho[idx] += qi * wxy * wz[c];
                    }
                }
            }
        }
    }

    /// Compute reciprocal-space forces on all positions, using the CPU. Positions must be in the
    /// primary box [0,L] per axis.
    /// todo: Is there a way to exclude static targets, or must all sources be targets?
    pub fn forces(&mut self, posits: &[Vec3], q: &[f32]) -> (Vec<Vec3>, f32) {
        assert_eq!(posits.len(), q.len());

        let (nx, ny, nz) = self.plan_dims;

        // Z is the fast dimension; keep this consistent with the CPU FFT setup, and
        // cuFFT; note that this is cuFFT's default layout.
        let n_real = nx * ny * nz;
        let nzc = nz / 2 + 1;
        let n_k = nx * ny * nzc;

        // Charge density.
        let mut rho_real = vec![0.; n_real];

        self.spread_charges(posits, q, &mut rho_real);

        // for i in 0..10 {
        //     println!("rho CPU pre fwd FFT: {:?}", rho_real[i])
        // }

        // Convert spread charges to K space
        let rho = fft3d_r2c(&mut rho_real, self.plan_dims, &mut self.planner);

        // for i in 220..230 {
        //     println!("rho CPU post fwd FFT: {:?}", rho[i])
        // }

        // eAk are per-dimension phase factors.
        // Apply influence function to get φ(k) from ρ(k), then make E(k)=i k φ(k)
        let mut exk = vec![Complex::new(0.0, 0.0); n_k];
        let mut eyk = vec![Complex::new(0.0, 0.0); n_k];
        let mut ezk = vec![Complex::new(0.0, 0.0); n_k];

        let mut energy = self.apply_ghat_and_grad(&rho, &mut exk, &mut eyk, &mut ezk, ny, nzc);

        energy += self_energy(q, self.alpha);

        // println!("\n");
        // println!("Energy CPU: {:?}", energy);
        //
        // for i in 220..230 {
        //     println!("exk post ghat: {:?}", exk[i])
        // }
        // for i in 220..230 {
        //     println!("eyk post ghat: {:?}", eyk[i])
        // }
        // println!("\n");

        // Inverse FFT to real-space E grids (C2R)
        let ex = fft3d_c2r(&mut exk, self.plan_dims, &mut self.planner);
        let ey = fft3d_c2r(&mut eyk, self.plan_dims, &mut self.planner);
        let ez = fft3d_c2r(&mut ezk, self.plan_dims, &mut self.planner);

        // for i in 220..230 {
        //     println!("exk CPU post inv FFT: {:?}", ex[i])
        // }
        // println!("\n");
        // for i in 220..230 {
        //     println!("eyk CPU post inv FFT: {:?}", ey[i])
        // }

        // todo: QC the minus sign?
        let f = gather_forces_to_atoms(posits, q, &ex, &ey, &ez, self.plan_dims, self.box_dims);

        (f, energy as f32)
    }

    /// Apply ĝ(k) – multiply each Fourier mode of the charge density by the Ewald/PME
    /// influence function to get the potential in k-space. Take the gradient of the potential in
    /// Fourier space to get the electric field.
    ///
    /// Also computes the half-spectrum energy.
    fn apply_ghat_and_grad(
        &self,
        rho: &[Complex_],
        exk: &mut [Complex_],
        eyk: &mut [Complex_],
        ezk: &mut [Complex_],
        ny: usize,
        nzc: usize,
    ) -> f64 {
        // Hoist shared refs so the parallel closure doesn't borrow &mut self
        let (kx, ky, kz) = (&self.kx, &self.ky, &self.kz);
        let (bx, by, bz) = (&self.bmod2_x, &self.bmod2_y, &self.bmod2_z);
        let (vol, alpha) = (self.vol, self.alpha);

        exk.par_iter_mut()
            .zip(eyk.par_iter_mut())
            .zip(ezk.par_iter_mut())
            .enumerate()
            .map(|(idx, ((ex, ey), ez))| {
                // Z-fast, Y-middle, X-slowest
                let izc = idx % nzc;
                let iy = (idx / nzc) % ny;
                let ix = idx / (nzc * ny);

                let kxv = kx[ix];
                let kyv = ky[iy];
                let kzv = kz[izc];

                let k2 = kxv.mul_add(kxv, kyv.mul_add(kyv, kzv * kzv));

                if k2 == 0.0 {
                    *ex = Complex::new(0.0, 0.0);
                    *ey = Complex::new(0.0, 0.0);
                    *ez = Complex::new(0.0, 0.0);
                    return 0.;
                }

                let bmod2 = bx[ix] * by[iy] * bz[izc];
                if bmod2 <= 1e-10 {
                    *ex = Complex::new(0.0, 0.0);
                    *ey = Complex::new(0.0, 0.0);
                    *ez = Complex::new(0.0, 0.0);
                    return 0.;
                }

                const TWO_TAU: f32 = 2. * TAU;

                // φ(k) = G(k) ρ(k) with B-spline deconvolution
                let ghat = (TWO_TAU / vol) * (-k2 / (4.0 * alpha * alpha)).exp() / (k2 * bmod2);
                let phi_k = rho[idx] * ghat;

                // E(k) = i k φ(k)
                *ex = Complex::new(0.0, kxv) * phi_k;
                *ey = Complex::new(0.0, kyv) * phi_k;
                *ez = Complex::new(0.0, kzv) * phi_k;

                // reciprocal-space energy density: (1/2) Re{ ρ*(k) φ(k) }
                // 0.5 * (rho[idx].conj() * phi_k).re as f64
                let energy = (rho[idx].re as f64) * (phi_k.re as f64)
                    + (rho[idx].im as f64) * (phi_k.im as f64);

                0.5 * energy
            })
            .sum()
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

/// |B(k)|^2 for Cardinal B-splines of order 4 (PME deconvolution).
/// Use signed/wrapped index distance to 0 to avoid over-amplifying near the Nyquist frequency.
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

            let eps = 1e-12;
            *val = (s.powi((m as i32) * 2)).max(eps); // |B(ω)|^2 = sinc^(2m)
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

fn wrap(i: isize, n: usize) -> usize {
    let n_isize = n as isize;
    let mut v = i % n_isize;
    if v < 0 {
        v += n_isize;
    }
    v as usize
}

///  Computes the direct, short-range component. Ideally, use a combined GPU kernel with Lennard Jones,
/// or a SIMD variant, instead of this.  We use this for short-range Coulomb forces on the CPU, as part of SPME.
/// `cutoff_dist` is the distance, in Å, at which we no longer apply any force from this component.
/// α controls the blending of short and long-range forces; 0.35Å for α is a good default for a cutoff of 10Å.
///
/// This assumes diff (and dir) is in order tgt - src.
/// Also returns potential energy. `dir` must be a unit vector.
pub fn force_coulomb_short_range(
    dir: Vec3,
    dist: f32,
    // Included in this form to share between this and Lennard Jones.
    inv_dist: f32,
    q_0: f32,
    q_1: f32,
    cutoff_dist: f32,
    α: f32,
) -> (Vec3, f32) {
    if dist >= cutoff_dist {
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

#[cfg(target_arch = "x86_64")]
pub fn force_coulomb_short_range_x8(
    dir: Vec3x8,
    dist: f32x8,
    inv_dist: f32x8,
    q_0: f32x8,
    q_1: f32x8,
    cutoff_dist: f32x8,
    // Alternatively, we could use a normal f32 for this, and splat it in-fn.
    α: f32x8,
) -> (Vec3x8, f32x8) {
    let α_r = α * dist;
    let erfc_term = α_r.erfc();

    let charge_term = q_0 * q_1;

    let energy = charge_term * inv_dist * erfc_term;

    let exp_term = (-α_r * α_r).exp();

    let force_mag = charge_term
        * (erfc_term * inv_dist * inv_dist
            + f32x8::splat(2.) * α * exp_term * f32x8::splat(INV_SQRT_PI) * inv_dist);

    let force = dir * force_mag;

    // This is where we diverge from the syntax of the non-SIMD variant;
    // the outside/inside cutoff.
    // per-lane mask: keep where dist < cutoff_dist, else zero
    unsafe {
        let keep = _mm256_cmp_ps::<{ _CMP_LT_OQ }>(dist.0, cutoff_dist.0);
        let zero = _mm256_set1_ps(0.0);

        let fx = _mm256_blendv_ps(zero, (force.x).0, keep);
        let fy = _mm256_blendv_ps(zero, (force.y).0, keep);
        let fz = _mm256_blendv_ps(zero, (force.z).0, keep);
        let en = _mm256_blendv_ps(zero, energy.0, keep);

        (
            Vec3x8 {
                x: f32x8(fx),
                y: f32x8(fy),
                z: f32x8(fz),
            },
            f32x8(en),
        )
    }
}

#[cfg(target_arch = "x86_64")]
pub fn force_coulomb_short_range_x16(
    dir: Vec3x16,
    dist: f32x16,
    // Included to share between this and Lennard Jones.
    inv_dist: f32x16,
    q_0: f32x16,
    q_1: f32x16,
    cutoff_dist: f32x16,
    // Alternatively, we could use a normal f32 for this, and splat it in-fn.
    α: f32x16,
) -> (Vec3x16, f32x16) {
    let α_r = α * dist;
    let erfc_term = α_r.erfc();

    let charge_term = q_0 * q_1;

    let energy = charge_term * inv_dist * erfc_term;

    let exp_term = (-α_r * α_r).exp();

    let force_mag = charge_term
        * (erfc_term * inv_dist * inv_dist
            + f32x16::splat(2.) * α * exp_term * f32x16::splat(INV_SQRT_PI) * inv_dist);

    let force = dir * force_mag;

    // This is where we diverge from the syntax of the non-SIMD variant;
    // the outside/inside cutoff.
    // per-lane mask: keep where dist < cutoff_dist, else zero
    unsafe {
        use core::arch::x86_64::*;
        let keep: __mmask16 = _mm512_cmp_ps_mask::<{ _CMP_LT_OQ }>(dist.0, cutoff_dist.0);

        let fx = _mm512_maskz_mov_ps(keep, (force.x).0);
        let fy = _mm512_maskz_mov_ps(keep, (force.y).0);
        let fz = _mm512_maskz_mov_ps(keep, (force.z).0);
        let en = _mm512_maskz_mov_ps(keep, energy.0);

        (
            Vec3x16 {
                x: f32x16(fx),
                y: f32x16(fy),
                z: f32x16(fz),
            },
            f32x16(en),
        )
    }
}

/// For flexible molecules, computes the correction term.
/// May be useful for scaling corrections, e.g. bonded scaling and exlusions?
/// todo: This may not be suitable for general use.
pub fn force_correction(dir: Vec3, r: f32, qi: f32, qj: f32, alpha: f32) -> Vec3 {
    // Complement of the real-space Ewald kernel; this is what “belongs” to reciprocal.
    let qfac = qi * qj;
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;

    let ar = alpha * r;
    let fmag = qfac
        * (erf(ar as f64) as f32 * inv_r2 - (alpha * TWO_INV_SQRT_PI) * (-ar * ar).exp() * inv_r);
    // let fmag = qfac * (mul_add(erf(ar as f64) as f32, inv_r2,  -(alpha * TWO_INV_SQRT_PI)) * (-ar * ar).exp() * inv_r);
    dir * fmag
}

/// Used on both CPU and GPU paths
fn self_energy(q: &[f32], alpha: f32) -> f64 {
    -(alpha / SQRT_PI) as f64 * q.iter().map(|&qi| (qi as f64) * (qi as f64)).sum::<f64>()
}

/// Interpolate E back to particles with the same B-spline weights; F = q E
fn gather_forces_to_atoms(
    posits: &[Vec3],
    q: &[f32],
    ex: &[f32],
    ey: &[f32],
    ez: &[f32],
    plan_dims: (usize, usize, usize),
    box_dims: (f32, f32, f32),
) -> Vec<Vec3> {
    let (nx, ny, nz) = plan_dims;
    let (lx, ly, lz) = box_dims;

    posits
        .par_iter()
        .enumerate()
        .map(|(i, &r)| {
            let (ix0, wx) = bspline4_weights(r.x / lx * nx as f32);
            let (iy0, wy) = bspline4_weights(r.y / ly * ny as f32);
            let (iz0, wz) = bspline4_weights(r.z / lz * nz as f32);

            let mut ex_v = 0.0f64;
            let mut ey_v = 0.0f64;
            let mut ez_v = 0.0f64;

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

                        // Z-fast real layout
                        let idx = ix * (ny * nz) + iy * nz + iz;

                        ex_v = w.mul_add(ex[idx] as f64, ex_v);
                        ey_v = w.mul_add(ey[idx] as f64, ey_v);
                        ez_v = w.mul_add(ez[idx] as f64, ez_v);
                    }
                }
            }

            let qi = q[i] as f64;

            // todo: QC the - sign.
            -Vec3 {
                x: (ex_v * qi) as f32,
                y: (ey_v * qi) as f32,
                z: (ez_v * qi) as f32,
            }
        })
        .collect()
}

/// A utility fn. todo: QC this.
fn next_planner_n(mut n: usize) -> usize {
    fn good(mut x: usize) -> bool {
        for p in [2, 3, 5, 7] {
            while x.is_multiple_of(p) {
                x /= p;
            }
        }
        x == 1
    }
    if n < 2 {
        n = 2;
    }
    while !good(n) {
        n += 1;
    }
    n
}

/// A utility function to get the (nx, ny, nz) tuple of plan dimensions based
/// on grid dimensions, and mesh spacing. A mesh spacing of 1Å is a good starting point.
/// Pass this into the `PmeRecip::new()` constructor, or set these values
/// up with some other approach.
pub fn get_grid_n(l: (f32, f32, f32), mesh_spacing: f32) -> (usize, usize, usize) {
    let (lx, ly, lz) = l;

    let nx0 = (lx / mesh_spacing).round().max(SPLINE_ORDER as f32) as usize;
    let ny0 = (ly / mesh_spacing).round().max(SPLINE_ORDER as f32) as usize;
    let nz0 = (lz / mesh_spacing).round().max(SPLINE_ORDER as f32) as usize;

    let nx = next_planner_n(nx0);
    let ny = next_planner_n(ny0);
    let mut nz = next_planner_n(nz0);

    // We use this because we use Z as the fast (contiguous) axis.
    if !nz.is_multiple_of(2) {
        nz = next_planner_n(nz + 1);
    }

    (nx, ny, nz)
}
