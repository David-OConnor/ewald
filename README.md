# Runs the Smooth Particle Ewald Mesh (SPME) algorithm for n-body simulations with periodic boundary conditions

[![Crate](https://img.shields.io/crates/v/ewald.svg)](https://crates.io/crates/ewald)
[![Docs](https://docs.rs/ewald/badge.svg)](https://docs.rs/ewald)

[//]: # ([![DOI]&#40;https://zenodo.org/badge/DOI/10.5281/zenodo.15616833.svg&#41;]&#40;https://doi.org/10.5281/zenodo.15616833&#41;)

This has applications primarily in structural biology. For example, molecular dynamics. Compared to other
n-body approximations for long-range forces, this has utility when periodic bounday conditions are used.
If not using these, for example in cosmology simulations, consider Barnes Hut, or Fast Multipole Methods (FMM)
instead.

Support for SIMD (256-bit and 512-bit), and CUDA (via CUDARC) are planned.

Used by the [Daedalus protein viewer and molecular dynamics program](https://github.com/david-oconnor/daedalus).

Here's an example of use:

```rust
use rayon::prelude::*;
use ewald::{force_coulomb_ewald_real, force_coulomb_ewald_real};

impl System {
    fn apply_forces(&self) {
        pairs
            // todo etc.
            .par_iter()
            .map(|(i_0, i_1)| {
                let atom_0 = &self.atoms[i_0];
                let atom_1 = &self.atoms[i_1];
                let diff = atom_1.pos - atom_0.pos;
                let r = diff.magnitude();
                let dir = diff / r;
    
                let mut f = Vec3::zero();
    
                f += force_coulomb_short_range(
                    dir,
                    r,
                    atom_0.charge,
                    atom_1.charge,
                    // e.g. (8Å, 10Å)
                    lr_switch: (LONG_RANGE_START, LONG_RANGE_CUTOFF),
                    α: f64,
                );
    
                // etc
    
                // todo: Update per the long range PAI
                f += force_coulomb_lr_recip(
                    dir,
                    r,
                    atom_0.charge,
                    atom_1.charge,
                    // e.g. (8Å, 10Å)
                    lr_switch: (LONG_RANGE_START, LONG_RANGE_CUTOFF),
                    α: f64,
                );
    
                atom_0.force += f;
                atom_1.force -= f;
            });
    }
}
```