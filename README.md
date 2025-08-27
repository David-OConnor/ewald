# Runs the Smooth Particle Ewald Mesh (SPME) algorithm for n-body simulations with periodic boundary conditions

[![Crate](https://img.shields.io/crates/v/ewald.svg)](https://crates.io/crates/ewald)
[![Docs](https://docs.rs/ewald/badge.svg)](https://docs.rs/ewald)

[//]: # ([![DOI]&#40;https://zenodo.org/badge/DOI/10.5281/zenodo.15616833.svg&#41;]&#40;https://doi.org/10.5281/zenodo.15616833&#41;)

##[Original paper describing the SPME method](https://biomolmd.org/mw/images/e/e0/Spme.pdf)

This has applications primarily in structural biology. For example, molecular dynamics. Compared to other
n-body approximations for long-range forces, this has utility when periodic bounday conditions are used.
If not using these, for example in cosmology simulations, consider Barnes Hut, or Fast Multipole Methods (FMM)
instead.

Uses Rayon to parallelize as thread pools. Support for SIMD (256-bit and 512-bit), and CUDA (via CUDARC) are planned. For now, you may wish to write
custom GPU kernels, using this lib as a reference.

WIP code for using the SPME/recip interaction on GPU.

Used by the [Daedalus protein viewer and molecular dynamics program](https://github.com/david-oconnor/daedalus).

Here's an example of use:

```rust
use rayon::prelude::*;
use ewald::{force_coulomb_ewald_real, force_coulomb_ewald_real};

impl System {
    // Primary application:
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
    
    // Helper methods, as required:
    /// Gather all particles that contribute to PME (dyn, water sites, statics).
    /// Returns positions wrapped to the primary box, their charges, and a map telling
    /// us which original DOF each entry corresponds to.
    fn gather_pme_particles_wrapped(&self) -> (Vec<Vec3>, Vec<f64>, Vec<PMEIndex>) {
        let n_dyn = self.atoms.len();
        let n_wat = self.water.len();
        let n_st = self.atoms_static.len();

        // Capacity hint: dyn + 4*water + statics
        let mut pos = Vec::with_capacity(n_dyn + 4 * n_wat + n_st);
        let mut q = Vec::with_capacity(pos.capacity());
        let mut map = Vec::with_capacity(pos.capacity());

        // Dynamic atoms
        for (i, a) in self.atoms.iter().enumerate() {
            pos.push(self.cell.wrap(a.posit));              // [0,L) per axis
            q.push(a.partial_charge);                       // already scaled to Amber units
            map.push(PMEIndex::Dyn(i));
        }

        // Water sites (OPC: O usually has 0 charge; include anyway—cost is negligible)
        for (i, w) in self.water.iter().enumerate() {
            pos.push(self.cell.wrap(w.o.posit));
            q.push(w.o.partial_charge);
            map.push(PMEIndex::WatO(i));
            pos.push(self.cell.wrap(w.m.posit));
            q.push(w.m.partial_charge);
            map.push(PMEIndex::WatM(i));
            pos.push(self.cell.wrap(w.h0.posit));
            q.push(w.h0.partial_charge);
            map.push(PMEIndex::WatH0(i));
            pos.push(self.cell.wrap(w.h1.posit));
            q.push(w.h1.partial_charge);
            map.push(PMEIndex::WatH1(i));
        }

        // Static atoms (contribute to field but you won't update accel)
        for (i, a) in self.atoms_static.iter().enumerate() {
            pos.push(self.cell.wrap(a.posit));
            q.push(a.partial_charge);
            map.push(PMEIndex::Static(i));
        }

        // Optional sanity check (debug only): near-neutral total charge
        #[cfg(debug_assertions)]
        {
            let qsum: f64 = q.iter().sum();
            if qsum.abs() > 1e-6 {
                eprintln!("[PME] Warning: net charge = {qsum:.6e} (PME assumes neutral or a uniform background)");
            }
        }

        (pos, q, map)
    }

    /// Run this at init, and whenever you update the sim box.
    pub fn regen_pme(&mut self) {
        let [lx, ly, lz] = self.cell.extent.to_arr();
        self.pme_recip = PmeRecip::new((SPME_N, SPME_N, SPME_N), (lx, ly, lz), EWALD_ALPHA);
    }
}
```