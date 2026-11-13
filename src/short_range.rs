//! This module contains code for the short-range component of the Coulomb force.

use lin_alg::f32::Vec3;
use lin_alg::f32::{Vec3x8, Vec3x16, f32x8, f32x16};
use statrs::function::erf::erfc;

use crate::INV_SQRT_PI;

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

#[inline]
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
    let erfc_term = f32x8::from_array(α_r.to_array().map(|x| erfc(x as f64) as f32));

    let charge_term = q_0 * q_1;

    let energy = charge_term * inv_dist * erfc_term;

    let exp_term = (-α_r * α_r).exp();

    let force_mag = charge_term
        * (erfc_term * inv_dist * inv_dist
            + f32x8::splat(2.) * α * exp_term * f32x8::splat(INV_SQRT_PI) * inv_dist);

    let force = dir * force_mag;

    let keep = dist.lt(cutoff_dist);
    let zero = f32x8::splat(0.);

    (
        Vec3x8 {
            x: keep.select(force.x, zero),
            y: keep.select(force.y, zero),
            z: keep.select(force.z, zero),
        },
        keep.select(energy, zero),
    )
}

#[inline]
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
    let erfc_term = f32x16::from_array(α_r.to_array().map(|x| erfc(x as f64) as f32));

    let charge_term = q_0 * q_1;

    let energy = charge_term * inv_dist * erfc_term;

    let exp_term = (-α_r * α_r).exp();

    let force_mag = charge_term
        * (erfc_term * inv_dist * inv_dist
            + f32x16::splat(2.) * α * exp_term * f32x16::splat(INV_SQRT_PI) * inv_dist);

    let force = dir * force_mag;

    let keep = dist.lt(cutoff_dist);
    let zero = f32x16::splat(0.);

    (
        Vec3x16 {
            x: keep.select(force.x, zero),
            y: keep.select(force.y, zero),
            z: keep.select(force.z, zero),
        },
        keep.select(energy, zero),
    )
}
