# Runs the Smooth Particle Ewald Mesh (SPME) algorithm for n-body simulations with periodic boundary conditions

[![Crate](https://img.shields.io/crates/v/ewald.svg)](https://crates.io/crates/ewald)
[![Docs](https://docs.rs/ewald/badge.svg)](https://docs.rs/ewald)
[![PyPI](https://img.shields.io/pypi/v/ewald.svg)](https://pypi.org/project/ewald)

[//]: # ([![DOI]&#40;https://zenodo.org/badge/DOI/10.5281/zenodo.15616833.svg&#41;]&#40;https://doi.org/10.5281/zenodo.15616833&#41;)

[Original paper describing the SPME method, by Essmann et al.](https://biomolmd.org/mw/images/e/e0/Spme.pdf)

Compute Coulomb forces for systems with periodic boundary conditions in an efficient, accurate way. Forces are broken
into three components: 

 - A short-range, *direct force*, 
 - A long-range *reciprical force*
 - A *correction force*, in the case of flexible molecules.

The bulk of the algorithmic complexity is in the *reciprical force*. This spreads charges along a discrete grid, and uses 
Fourier analysis to compute force vectors. The *direct force* is similar to a Coulomb calculation, but uses a fixed distance 
 cutoff, and ewald screening.

This library is for Python and Rust. It supports GPU, and thread-pooled CPU.

**Note: The Python version is currently CPU only**. We would like to fix this, but are having trouble
linking the Cuda FFTs.

This has applications primarily in structural biology and chemistry simulations. For example, molecular dynamics.
It's used to compute Coulomb (or equivalent) forces in systems using an approximation for long-range forces that
scales as $N log(N)$ with respect to system size.

Compared to other n-body approximations for long-range forces, this has utility when periodic bounday conditions are used.
If not using these, for example in cosmology simulations, consider Barnes Hut, or Fast Multipole Methods (FMM)
instead.

The API is split into two main parts: A standalone function to calculate short-range force, and 
a struct with forces method for long-range reciprical forces. The short range computation uses Ewald screening,
and isn't appropriate to use without adding long-range forces to it.

This uses Rayon to parallelize computations in thread pools. Support for SIMD (256-bit and 512-bit), is planned. To use on an nVidia GPU, enable 
either the `cufft` or `vkfft`  feature in `Cargo.toml`. We use these GPU FFT libraries to compute. Note that they both
use the Cuda driver internally; VKFFT is configured with a Cuda backend. **VkFFT is currently broken; don't use it**.

Used by the [Daedalus protein viewer and molecular dynamics program](https://github.com/david-oconnor/daedalus), and
the [Dynamics library](https://github.com/david-oconnor/dynamics).

Uses `f32` for Coulomb interactions. Energy sums are computed as `f64`.

Note: For optimal performance, you may wish to implement short-range SPME integrated with Lennard Jones interactions
in your application as an integrated GPU kernel, instead of using the short-range function here directly. This
library is most effectively used for its reciprical implemention on GPU.

Below is an example of use. The Python API is equivalent.



```rust
use rayon::prelude::*;
use ewald::{force_coulomb_ewald_real, force_coulomb_ewald_real, get_grid_n};

const LONG_RANGE_CUTOFF: f32 = 10.0;

// A bigger α means more damping, and a smaller real-space contribution. (Cheaper real), but larger
// reciprocal load.
const EWALD_ALPHA: f32 = 0.35; // Å^-1. 0.35 is good for cutoff = 10.
const MESH_SPACING: f32 = 1.0;

impl System {
    // Primary application:
    fn apply_forces(&self) {
        // If using with gpu, set up the stream like this. Ideally, store it somewhere instead
        // of generating each time.
        let ctx = cudarc::driver::CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        
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

        // Use `stream` if using with GPU; omit otherwise.
        let (recip_forces_per_atom, energy_recip) = self.pme_recip.forces(&stream, &atom_posits, &[atom_charges]);
    }

    /// Run this at init, and whenever you update the sim box.
    pub fn regen_pme(&mut self) {
        let l = (self.lx, self.ly, self.lz); // As required per your simulation.
        let n = get_grid_n(l, MESH_SPACING);
        
        // Note: You should calculate n using your grid dimensions n, and an appropriate mesh spacing, 
        // or use the utility function we provide.
        self.pme_recip = PmeRecip::new(n, l, EWALD_ALPHA);
        
        // Or on GPU (See how we acquire the Cudarc stream above)
        self.pme_recip = PmeRecip::new(stream, n, l, EWALD_ALPHA);
    }
}
```


## References

 - [A smooth particle mesh Ewald method](https://biomolmd.org/mw/images/e/e0/Spme.pdf)
 - [A comparison of the Spectral Ewald and Smooth Particle Mesh Ewald...](https://arxiv.org/pdf/1712.04718)