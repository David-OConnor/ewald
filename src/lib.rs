#![allow(non_snake_case)]
#![allow(mixed_script_confusables)]
#![allow(confusable_idents)]
#![allow(clippy::needless_range_loop)]

//! For Smooth-Particle-Mesh Ewald; a standard approximation for Coulomb forces in MD.
//! We use this to handle periodic boundary conditions properly, which we use to take the
//! water molecules into account.

// todo: f32 support / generic floats

// todo: Add CUDA and SIMD support.

use std::f64::consts::{PI, TAU};

// todo: This may be a good candidate for a standalone library.
use lin_alg::f64::Vec3;
#[cfg(target_arch = "x86_64")]
use lin_alg::f64::{Vec3x8, f64x8};
use rustfft::{FftPlanner, num_complex::Complex};
use statrs::function::erf::{erf, erfc};

const SQRT_PI: f64 = 1.7724538509055159;

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

    /// Compute reciprocal-space forces. Positions must be in the primary box [0,L) per axis.
    pub fn forces(&mut self, pos: &[Vec3], q: &[f64]) -> Vec<Vec3> {
        assert_eq!(pos.len(), q.len());
        let npts = self.nx * self.ny * self.nz;

        let mut rho = vec![Complex::<f64>::new(0.0, 0.0); npts];
        self.spread_charges(pos, q, &mut rho);

        fft3_inplace(
            &mut rho,
            (self.nx, self.ny, self.nz),
            &mut self.planner,
            true,
        );

        // pply influence function to get φ(k) from ρ(k), then make E(k)=i k φ(k)
        let mut exk = vec![Complex::<f64>::new(0.0, 0.0); npts];
        let mut eyk = vec![Complex::<f64>::new(0.0, 0.0); npts];
        let mut ezk = vec![Complex::<f64>::new(0.0, 0.0); npts];

        for iz in 0..self.nz {
            let kz = self.kz[iz];
            let bmod_z = self.bmod2_z[iz];
            for iy in 0..self.ny {
                let ky = self.ky[iy];
                let bmod_y = self.bmod2_y[iy];
                let row = iz * (self.nx * self.ny) + iy * self.nx;
                for ix in 0..self.nx {
                    let idx = row + ix;
                    let kx = self.kx[ix];
                    let bmod_x = self.bmod2_x[ix];

                    // k^2 and B-spline |B(k)|^2 product
                    let k2 = kx * kx + ky * ky + kz * kz;
                    if k2 == 0.0 {
                        // k=0: set field to zero (tin-foil boundary)
                        exk[idx] = Complex::new(0.0, 0.0);
                        eyk[idx] = Complex::new(0.0, 0.0);
                        ezk[idx] = Complex::new(0.0, 0.0);
                        continue;
                    }
                    let bmod2 = bmod_x * bmod_y * bmod_z;
                    // Influence function \hat G(k)
                    let ghat =
                        (2.0 * TAU / self.vol) * (-k2 / (4.0 * self.alpha * self.alpha)).exp() / k2;
                    // Deconvolution by |B(k)|^2 (avoid division by ~0)
                    let ghat = if bmod2 > 1e-10 { ghat / bmod2 } else { 0.0 };

                    // φ(k) = G(k) ρ(k)
                    let phi_k = rho[idx] * ghat;

                    // E(k) = i k φ(k)
                    // i * real -> imag, i * imag -> -real
                    exk[idx] = Complex::new(0.0, -kx) * phi_k;
                    eyk[idx] = Complex::new(0.0, -ky) * phi_k;
                    ezk[idx] = Complex::new(0.0, -kz) * phi_k;
                }
            }
        }

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
        let mut forces = vec![Vec3::new_zero(); pos.len()];
        for (i, &r) in pos.iter().enumerate() {
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
            forces[i] = e * (q[i]); // F = q * E
        }

        forces
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

// Minimal, cache-friendly 3D FFT using rustfft 1D plans along each axis.
// dir=true => forward; dir=false => inverse (and rustfft handles scaling=1).
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
/// This assumes the tapering or cutoff has been applied outside.
fn force_coulomb_short_range_inner(dir: Vec3, r: f64, qi: f64, qj: f64, α: f64) -> Vec3 {
    // F = q_i q_j [ erfc(αr)/r² + 2α/√π · e^(−α²r²)/r ]  · 4πϵ0⁻¹  · r̂
    let qfac = qi * qj;
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;

    let α_r = α * r;
    let erfc_term = erfc(α_r);
    let exp_term = (-α_r * α_r).exp();
    let force_mag = qfac * (erfc_term * inv_r2 + 2.0 * α * exp_term / (SQRT_PI * r));

    dir * force_mag
}

/// We use this for short-range Coulomb forces, as part of SPME.
/// `cutoff_dist` is the distance, in Å, we switch between short-range, and long-range reciprical
/// forces.
/// `α`
pub fn force_coulomb_short_range(
    dir: Vec3,
    r: f64,
    qi: f64,
    qj: f64,
    // lr_switch: (f64, f64),
    cutoff_dist: f64,
    α: f64,
) -> Vec3 {
    // Outside the taper region; return 0. (All the force is handled in the long-range region.)
    // if r >= lr_switch.1 {
    if r > cutoff_dist {
        return Vec3::new_zero();
    }

    force_coulomb_short_range_inner(dir, r, qi, qj, α)

    // let f = force_coulomb_short_range_inner(dir, r, qi, qj, α);
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

    let force_mag = qfac
        * (erfc_term * inv_r2 + f64x8::splat(2.) * α * exp_term / (f64x8::splat(SQRT_PI) * r));

    dir * force_mag
}

/// Useful for scaling corrections, e.g. 1-4 exclusions in AMBER.
pub fn ewald_comp_force(dir: Vec3, r: f64, qi: f64, qj: f64, alpha: f64) -> Vec3 {
    // Complement of the real-space Ewald kernel; this is what “belongs” to reciprocal.
    let qfac = qi * qj;
    let inv_r = 1.0 / r;
    let inv_r2 = inv_r * inv_r;

    let ar = alpha * r;
    let fmag = qfac * (erf(ar) * inv_r2 - (2.0 * alpha / SQRT_PI) * (-ar * ar).exp() * inv_r);
    dir * fmag
}
