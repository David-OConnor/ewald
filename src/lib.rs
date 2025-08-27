#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]
#![allow(clippy::needless_range_loop)]

//! For Smooth-Particle-Mesh Ewald; a standard approximation for Coulomb forces in MD.
//! We use this to handle periodic boundary conditions properly, which we use to take the
//! water molecules into account.

// todo: f32 support / generic floats

// todo: Add CUDA and SIMD support.

#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{
    f64::consts::{PI, TAU},
    time::Instant,
};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaModule, CudaStream, LaunchConfig, PushKernelArg, sys::CUstream, DevicePtr};
// todo: This may be a good candidate for a standalone library.
use lin_alg::f64::Vec3;
#[cfg(target_arch = "x86_64")]
use lin_alg::f64::{Vec3x8, f64x8};
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use statrs::function::erf::{erf, erfc};

const SQRT_PI: f64 = 1.7724538509055159;
const INV_SQRT_PI: f64 = 1. / SQRT_PI;

const SPLINE_ORDER: usize = 4;

pub struct PmeRecip {
    nx: usize,
    ny: usize,
    nz: usize,
    lx: f64,
    ly: f64,
    lz: f64,
    vol: f64,
    /// A tunable variable used in the splitting between short range and long range forces.
    /// A bigger α means more damping, and a smaller real-space contribution. (Cheaper real), but larger
    /// reciprocal load.
    pub alpha: f64,
    // Precomputed k-vectors and B-spline deconvolution |B(k)|^2
    kx: Vec<f64>,
    ky: Vec<f64>,
    kz: Vec<f64>,
    bmod2_x: Vec<f64>,
    bmod2_y: Vec<f64>,
    bmod2_z: Vec<f64>,
    planner: FftPlanner<f64>,
}

impl Default for PmeRecip {
    // todo: Rust needs a beter solution for this.
    fn default() -> Self {
        Self {
            nx: 0,
            ny: 0,
            nz: 0,
            lx: 0.,
            ly: 0.,
            lz: 0.,
            vol: 0.,
            alpha: 0.,
            kx: Vec::new(),
            ky: Vec::new(),
            kz: Vec::new(),
            bmod2_x: Vec::new(),
            bmod2_y: Vec::new(),
            bmod2_z: Vec::new(),
            planner: FftPlanner::new(),
        }
    }
}

impl PmeRecip {
    pub fn new(n: (usize, usize, usize), l: (f64, f64, f64), alpha: f64) -> Self {
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
        }
    }

    /// Helper to reduce DRY between GPU and CPU variants.
    /// Note: Parallelization here doesn't seem to have much effect.
    fn forces_part_b(
        &self,
        rho: &[Complex<f64>],
        n_pts: usize,
    ) -> (Vec<Complex<f64>>, Vec<Complex<f64>>, Vec<Complex<f64>>) {
        // Hoist shared refs so the parallel closure doesn't borrow &mut self
        let (nx, ny, _nz) = (self.nx, self.ny, self.nz);
        let (kx, ky, kz) = (&self.kx, &self.ky, &self.kz);
        let (bx, by, bz) = (&self.bmod2_x, &self.bmod2_y, &self.bmod2_z);
        let (vol, alpha) = (self.vol, self.alpha);

        // Apply influence function to get φ(k) from ρ(k), then make E(k)=i k φ(k)
        let mut exk = vec![Complex::<f64>::new(0.0, 0.0); n_pts];
        let mut eyk = vec![Complex::<f64>::new(0.0, 0.0); n_pts];
        let mut ezk = vec![Complex::<f64>::new(0.0, 0.0); n_pts];

        // let start = Instant::now();
        exk.par_iter_mut()
            .zip(eyk.par_iter_mut())
            .zip(ezk.par_iter_mut())
            .enumerate()
            .for_each(|(idx, ((ex, ey), ez))| {
                let ix = idx % nx;
                let iy = (idx / nx) % ny;
                let iz = idx / (nx * ny);

                let kxv = kx[ix];
                let kyv = ky[iy];
                let kzv = kz[iz];

                let k2 = kxv * kxv + kyv * kyv + kzv * kzv;
                if k2 == 0.0 {
                    *ex = Complex::new(0.0, 0.0);
                    *ey = Complex::new(0.0, 0.0);
                    *ez = Complex::new(0.0, 0.0);
                    return;
                }

                let bmod2 = bx[ix] * by[iy] * bz[iz];
                if bmod2 <= 1e-10 {
                    *ex = Complex::new(0.0, 0.0);
                    *ey = Complex::new(0.0, 0.0);
                    *ez = Complex::new(0.0, 0.0);
                    return;
                }

                // φ(k) = G(k) ρ(k) with B-spline deconvolution
                let ghat = (2.0 * TAU / vol) * (-k2 / (4.0 * alpha * alpha)).exp() / (k2 * bmod2);
                let phi_k = rho[idx] * ghat;

                // E(k) = i k φ(k)
                *ex = Complex::new(0.0, -kxv) * phi_k;
                *ey = Complex::new(0.0, -kyv) * phi_k;
                *ez = Complex::new(0.0, -kzv) * phi_k;
            });

        (exk, eyk, ezk)
    }

    /// Interpolate E back to particles with the same B-spline weights; F = q E
    /// Helper to reduce DRY between GPU and CPU variants.
    /// Note: Parallelization here doesn't seem to have much effect.
    fn forces_part_c(
        &self,
        pos: &[Vec3],
        exk: &[Complex<f64>],
        eyk: &[Complex<f64>],
        ezk: &[Complex<f64>],
        q: &[f64],
    ) -> Vec<Vec3> {
        pos.par_iter()
            .enumerate()
            .map(|(i, &r)| {
                let (ix0, wx) = bspline4_weights(r.x / self.lx * self.nx as f64);
                let (iy0, wy) = bspline4_weights(r.y / self.ly * self.ny as f64);
                let (iz0, wz) = bspline4_weights(r.z / self.lz * self.nz as f64);

                let mut e = Vec3::new_zero();

                for a in 0..4 {
                    let ix = wrap(ix0 + a as isize, self.nx);
                    let wxa = wx[a];

                    for b in 0..4 {
                        let iy = wrap(iy0 + b as isize, self.ny);
                        let wyb = wy[b];
                        let wxy = wxa * wyb;

                        for c in 0..4 {
                            let iz = wrap(iz0 + c as isize, self.nz);
                            let w = wxy * wz[c];
                            let idx = iz * (self.nx * self.ny) + iy * self.nx + ix;

                            e.x += w * exk[idx].re; // after inverse FFT, fields are real (imag ~ 0)
                            e.y += w * eyk[idx].re;
                            e.z += w * ezk[idx].re;
                        }
                    }
                }
                // F = q * E
                e * (q[i])
            })
            .collect()
    }

    /// Compute reciprocal-space forces. Positions must be in the primary box [0,L) per axis.
    pub fn forces(&mut self, pos: &[Vec3], q: &[f64]) -> Vec<Vec3> {
        assert_eq!(pos.len(), q.len());
        let n_pts = self.nx * self.ny * self.nz;
        let mut rho = vec![Complex::<f64>::new(0.0, 0.0); n_pts];
        self.spread_charges(pos, q, &mut rho);

        fft3_inplace(
            &mut rho,
            (self.nx, self.ny, self.nz),
            &mut self.planner,
            true,
        );

        let (mut exk, mut eyk, mut ezk) = self.forces_part_b(&rho, n_pts);
        // let elapsed = start.elapsed();
        // println!("SPME A: {} us", elapsed.as_micros());

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
        // let elapsed = start.elapsed();
        // println!("SPME B: {} us", elapsed.as_micros());

        // let start = Instant::now();
        let result = self.forces_part_c(pos, &exk, &eyk, &ezk, q);

        // let elapsed = start.elapsed();
        // println!("SPME C: {} us", elapsed.as_micros());

        result
    }

    #[cfg(feature = "cuda")]
    /// Compute reciprocal-space forces. Positions must be in the primary box [0,L) per axis.
    /// Note that this only uses GPU for the FFT part, but this is dominates computation time.
    ///
    /// This isn't calling a custom CUDA kernel, to iterate over an array, It leverages host-side
    /// CUDA code, which calls cuFFT.
    pub fn forces_gpu(
        &mut self,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        pos: &[Vec3],
        q: &[f64],
    ) -> Vec<Vec3> {
        assert_eq!(pos.len(), q.len());
        let n_pts = self.nx * self.ny * self.nz;
        let mut rho = vec![Complex::<f64>::new(0.0, 0.0); n_pts];
        self.spread_charges(pos, q, &mut rho);

        let start = Instant::now();
        fft3_inplace(
            &mut rho,
            (self.nx, self.ny, self.nz),
            &mut self.planner,
            true,
        );

        let elapsed = start.elapsed();
        println!("SPME FFT A: {} ms", elapsed.as_millis());

        let (mut exk, mut eyk, mut ezk) = self.forces_part_b(&rho, n_pts);

        let mut exk_test = exk.clone();
        // todo temp. Note: The tests disagree, and the CPU version is reporting 0s?
        fft3_inplace(
            &mut exk_test,
            (self.nx, self.ny, self.nz),
            &mut self.planner,
            false,
        );

        let start = Instant::now();
        // Run the inverse FFTs on the GPU.
        {
            let exk_gpu = stream.memcpy_stod(&flatten_cplx_vec(&exk)).unwrap();
            let eyk_gpu = stream.memcpy_stod(&flatten_cplx_vec(&eyk)).unwrap();
            let ezk_gpu = stream.memcpy_stod(&flatten_cplx_vec(&ezk)).unwrap();

            // Get raw CUdeviceptrs tied to this stream.
            // IMPORTANT: keep the guards alive until after the FFI call returns.
            let (exk_ptr, _exk_guard) = exk_gpu.device_ptr(stream);
            let (eyk_ptr, _eyk_guard) = eyk_gpu.device_ptr(stream);
            let (ezk_ptr, _ezk_guard) = ezk_gpu.device_ptr(stream);

            let nx = exk.len();
            let ny = eyk.len();
            let nz = ezk.len();

            // Call the cuFFT wrapper (still runs on GPU; invoked from host)
            unsafe {
                spme_inverse_ffts_3_c2c(
                    exk_ptr as *mut std::ffi::c_void,
                    eyk_ptr as *mut std::ffi::c_void,
                    ezk_ptr as *mut std::ffi::c_void,
                    self.nx as i32,
                    self.ny as i32,
                    self.nz as i32,
                );
            }

            exk = unflatten_cplx_vec(&stream.memcpy_dtov(&exk_gpu).unwrap());
            eyk = unflatten_cplx_vec(&stream.memcpy_dtov(&eyk_gpu).unwrap());
            ezk = unflatten_cplx_vec(&stream.memcpy_dtov(&ezk_gpu).unwrap());
        }

        // todo temp. And, this test is... showing 0 for the CPU values?
        println!("CPU: {:.4?}", &exk_test[0..10]);
        println!("GPU : {:.4?}", &exk[0..10]);

        let elapsed = start.elapsed();
        println!("SPME FFT B: {} ms", elapsed.as_millis());

        // let start = Instant::now();
        let result = self.forces_part_c(pos, &exk, &eyk, &ezk, q);
        // let elapsed = start.elapsed();
        // println!("SPME recip B: {} us", elapsed.as_millis());

        result
    }

    fn spread_charges(&self, pos: &[Vec3], q: &[f64], rho: &mut [Complex<f64>]) {
        let nxny = self.nx * self.ny;
        for (r, &qi) in pos.iter().zip(q.iter()) {
            // fractional grid coords
            let sx = r.x / self.lx * self.nx as f64;
            let sy = r.y / self.ly * self.ny as f64;
            let sz = r.z / self.lz * self.nz as f64;

            let (ix0, wx) = bspline4_weights(sx);
            let (iy0, wy) = bspline4_weights(sy);
            let (iz0, wz) = bspline4_weights(sz);

            for a in 0..4 {
                let ix = wrap(ix0 + a as isize, self.nx);
                let wxa = wx[a];

                for b in 0..4 {
                    let iy = wrap(iy0 + b as isize, self.ny);
                    let wyb = wy[b];
                    let wxy = wxa * wyb;

                    for c in 0..4 {
                        let iz = wrap(iz0 + c as isize, self.nz);
                        let idx = iz * nxny + iy * self.nx + ix;
                        rho[idx].re += qi * wxy * wz[c];
                    }
                }
            }
        }
    }
}

/// k-array for an orthorhombic cell; FFT index convention → physical wavevector.
fn make_k_array(n: usize, L: f64) -> Vec<f64> {
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
        *out_ = tau_div_l * (fi as f64);
    }
    out
}

/// |B(k)|^2 for B-spline of order m (PME deconvolution).
/// Use signed/wrapped index distance to 0 to avoid over-amplifying near Nyquist.
fn spline_bmod2_1d(n: usize, m: usize) -> Vec<f64> {
    assert!(m >= 1);
    let mut v = vec![0.0; n];

    for (i, val) in v.iter_mut().enumerate() {
        let k = i.min(n - i);

        if k == 0 {
            *val = 1.0; // sinc(0) = 1
        } else {
            let t = PI * (k as f64) / (n as f64); // = |ω|/2 with ω=2πk/n
            let s = t.sin() / t; // sinc(|ω|/2)
            *val = s.powi((m as i32) * 2); // |B(ω)|^2 = sinc^(2m)
        }
    }
    v
}

/// Cubic B-spline weights for 4 neighbors; returns starting index and 4 weights.
/// Input s is in grid units (0..n), arbitrary real; we wrap indices to the grid.
fn bspline4_weights(s: f64) -> (isize, [f64; 4]) {
    let sfloor = s.floor();
    let u = s - sfloor; // fractional part in [0,1)
    let i0 = sfloor as isize - 1; // left-most point of 4-support

    // Cardinal cubic B-spline weights (order 4)
    let u2 = u * u;
    let u3 = u2 * u;

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
    data: &mut [Complex<f64>],
    dims: (usize, usize, usize),
    planner: &mut FftPlanner<f64>,
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
        let mut tmp = vec![Complex::<f64>::new(0.0, 0.0); ny];
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
        let mut tmp = vec![Complex::<f64>::new(0.0, 0.0); nz];
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
        let scale = 1.0 / (len as f64);
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

/// We use this for short-range Coulomb forces, as part of SPME.
/// `cutoff_dist` is the distance, in Å, we switch between short-range, and long-range reciprical
/// forces.
/// This assumes diff (and dir) is in order tgt - src.
/// Also returns potential energy.
pub fn force_coulomb_short_range(
    dir: Vec3,
    dist: f64,
    // Included to share between this and LJ.
    inv_dist: f64,
    inv_dist_sq: f64,
    q_0: f64,
    q_1: f64,
    // lr_switch: (f64, f64),
    cutoff_dist: f64,
    α: f64,
) -> (Vec3, f64) {
    // Outside the taper region; return 0. (All the force is handled in the long-range region.)
    // if r >= lr_switch.1 {
    if dist > cutoff_dist {
        return (Vec3::new_zero(), 0.);
    }

    let α_r = α * dist;
    let erfc_term = erfc(α_r);
    let charge_term = q_0 * q_1;

    let energy = charge_term * inv_dist * erfc_term;

    let exp_term = (-α_r * α_r).exp();

    let force_mag =
        charge_term * (erfc_term * inv_dist_sq + 2.0 * α * exp_term * INV_SQRT_PI * inv_dist);

    (dir * force_mag, energy)

    // Removed taper code.
    // // Inside the taper region, return the short-range force.
    // if r <= lr_switch.0 {
    //     return f;
    // }
    //
    // // Apply switch to the potential; to approximate on the force, multiply by S and add -U*dS/dr*r̂
    // // For brevity, a common practical shortcut is scaling force by S(r):
    // let s = (r - lr_switch.0) / (lr_switch.1 - lr_switch.0);
    // let (S, _dSds) = taper(s);
    // f * S
}

// todo: Update this to reflect your changes to the algo above that apply tapering.
pub fn force_coulomb_ewald_real_x8(
    dir: Vec3x8,
    r: f64x8,
    qi: f64x8,
    qj: f64x8,
    α: f64x8,
) -> Vec3x8 {
    // F = q_i q_j [ erfc(αr)/r² + 2α/√π · e^(−α²r²)/r ]  · 4πϵ0⁻¹  · r̂
    let qfac = qi * qj;
    let inv_r = f64x8::splat(1.) / r;
    let inv_r2 = inv_r * inv_r;

    // let erfc_term = erfc(alpha * r);
    let erfc_term = f64x8::splat(0.); // todo temp: Figure how how to do erfc with SIMD.

    // todo: Figure out how to do exp with SIMD. Probably need powf in lin_alg
    // let exp_term = (-alpha * alpha * r * r).exp();
    // let exp_term = f64x8::splat(E).pow(-alpha * alpha * r * r);
    let exp_term = f64x8::splat(1.); // todo temp

    let force_mag =
        qfac * (erfc_term * inv_r2 + f64x8::splat(2.) * α * exp_term / (f64x8::splat(SQRT_PI) * r));

    dir * force_mag
}

/// Useful for scaling corrections, e.g. 1-4 exclusions in AMBER.
pub fn ewald_comp_force(dir: Vec3, r: f64, qi: f64, qj: f64, alpha: f64) -> Vec3 {
    // Complement of the real-space Ewald kernel; this is what “belongs” to reciprocal.
    let qfac = qi * qj;
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;

    let ar = alpha * r;
    let fmag = qfac * (erf(ar) * inv_r2 - (2.0 * alpha * INV_SQRT_PI) * (-ar * ar).exp() * inv_r);
    dir * fmag
}

/// For CUDA serialization
fn flatten_cplx_vec(v: &[Complex<f64>]) -> Vec<f32> {
    let mut result = Vec::with_capacity(v.len() * 2);

    for v_ in v {
        result.push(v_.re as f32);
        result.push(v_.im as f32);
    }

    result
}

/// For CUDA deserialization
fn unflatten_cplx_vec(v: &[f32]) -> Vec<Complex<f64>> {
    let mut result = Vec::with_capacity(v.len() / 2);

    for i in 0..v.len() / 2 {
        result.push(Complex::new(v[i * 2] as f64, v[i * 2 + 1] as f64));
    }

    result
}

// todo: Experimenting
unsafe extern "C" {
    // The CUDA function name must match this.
    fn spme_inverse_ffts_3_c2c(
        exk: *mut std::ffi::c_void,
        eyk: *mut std::ffi::c_void,
        ezk: *mut std::ffi::c_void,
        nx: i32,
        ny: i32,
        nz: i32,
    );
}
