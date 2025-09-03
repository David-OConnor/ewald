# Runs the Smooth Particle Ewald Mesh (SPME) algorithm for n-body simulations with periodic boundary conditions

[![Crate](https://img.shields.io/crates/v/ewald.svg)](https://crates.io/crates/ewald)
[![Docs](https://docs.rs/ewald/badge.svg)](https://docs.rs/ewald)
[![PyPI](https://img.shields.io/pypi/v/ewald.svg)](https://pypi.org/project/ewald)

[//]: # ([![DOI]&#40;https://zenodo.org/badge/DOI/10.5281/zenodo.15616833.svg&#41;]&#40;https://doi.org/10.5281/zenodo.15616833&#41;)

[Original paper describing the SPME method](https://biomolmd.org/mw/images/e/e0/Spme.pdf)

This library is for Python and Rust.

This has applications primarily in structural biology. For example, molecular dynamics. Compared to other
n-body approximations for long-range forces, this has utility when periodic bounday conditions are used.
If not using these, for example in cosmology simulations, consider Barnes Hut, or Fast Multipole Methods (FMM)
instead.

Uses Rayon to parallelize as thread pools. Support for SIMD (256-bit and 512-bit), and CUDA (via CUDARC) are planned. For now, you may wish to write
custom GPU kernels, using this lib as a reference.

WIP code for using the SPME/recip interaction on GPU.

Used by the [Daedalus protein viewer and molecular dynamics program](https://github.com/david-oconnor/daedalus).

Here's an example of use. The Python API is equivalent.

```rust
use rayon::prelude::*;
use ewald::{force_coulomb_ewald_real, force_coulomb_ewald_real};

const LONG_RANGE_CUTOFF: f64 = 10.0;

// A bigger α means more damping, and a smaller real-space contribution. (Cheaper real), but larger
// reciprocal load.
const EWALD_ALPHA: f64 = 0.35; // Å^-1. 0.35 is good for cutoff = 10.

impl System {
    // Primary application:
    fn apply_forces(&self) {
        pairs
            .par_iter()
            .map(|(i_0, i_1)| {
                let atom_0 = &self.atoms[i_0];
                let atom_1 = &self.atoms[i_1];
                let diff = atom_1.pos - atom_0.pos;
                let r = diff.magnitude();
                let dir = diff / r;

                let mut f = Vec3::zero();

                let (f, energy) = force_coulomb_short_range(
                    dir,
                    r,
                    // We include 1/r as it's likely shared between this and Lennard Jones;
                    // improves efficiency.
                    1./r,
                    atom_0.charge,
                    atom_1.charge,
                    // e.g. (8Å, 10Å)
                    LONG_RANGE_CUTOFF,
                    ALPHA,
                );

                atom_0.force += f;
                atom_1.force -= f;
            });

        let (recip_forces_per_atom, energy_recip) = self.pme_recip.forces(&atom_posits, &[atom_charges]);
    }

    /// Run this at init, and whenever you update the sim box.
    pub fn regen_pme(&mut self) {
        self.pme_recip = PmeRecip::new((SPME_N, SPME_N, SPME_N), self.cell.extent, EWALD_ALPHA);
    }
}
```