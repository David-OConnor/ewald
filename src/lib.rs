//! For Smooth-Particle-Mesh_Ewald; a standard approximation for Coulomb forces in MD.
//! We use this to handle periodic boundary conditions properly, which we use to take the
//! water molecules into account.

// todo: f32 support / generic floats

use std::f64::consts::{FRAC_2_SQRT_PI, PI, TAU};

// todo: This may be a good candidate for a standalone library.
use lin_alg::f64::Vec3;
#[cfg(target_arch = "x86_64")]
use lin_alg::f64::{Vec3x8, f64x8};
use rustfft::{FftDirection, FftPlanner, num_complex::Complex};
use statrs::function::erf::{erf, erfc};

// use crate::dynamics::{AtomDynamics, MdState, ambient::SimBox, non_bonded::SCALE_COUL_14};



const SQRT_PI: f64 = 1.7724538509055159;


// todo: How and where are you setting the cutoff? (e.g. of 10Å, between real and long-range Ewald)

/// We use this to smoothly switch between short-range and long-range Ewald.
fn taper(s: f64) -> (f64, f64) {
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

/// This assumes the tapering has been applied outside.
fn force_coulomb_ewald_real_inner(dir: Vec3, r: f64, qi: f64, qj: f64, α: f64) -> Vec3 {
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
/// `lr_switch` is (start, cutoff).
pub fn force_coulomb_ewald_real(dir: Vec3, r: f64, qi: f64, qj: f64, lr_switch: (f64, f64), α: f64) -> Vec3 {
    // Outside the taper region; return 0. (All the force is handled in the long-range region.)
    if r >= lr_switch.1 {
        return Vec3::new_zero();
    }

    let f = force_coulomb_ewald_real_inner(dir, r, qi, qj, α);

    // Inside the taper region, return the short-range force.
    if r <= lr_switch.0 {
        return f;
    }

    // Apply switch to the potential; to approximate on the force, multiply by S and add -U*dS/dr*r̂
    // For brevity, a common practical shortcut is scaling force by S(r):
    let s = (r - lr_switch.0) / (lr_switch.1 - lr_switch.0);
    let (S, _dSds) = taper(s);
    f * S
}

// todo: Update this to reflectg your changes to the algo above.
pub fn force_coulomb_ewald_real_x8(
    dir: Vec3x8,
    r: f64x8,
    qi: f64x8,
    qj: f64x8,
    alpha: f64x8,
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
        * (erfc_term * inv_r2 + f64x8::splat(2.) * alpha * exp_term / (f64x8::splat(SQRT_PI) * r));

    dir * force_mag
}
