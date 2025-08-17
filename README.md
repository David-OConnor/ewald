# Runs the Smooth Particle Ewald Mesh (SPME) algorithm for n-body simulations with periodic boundary conditions

[![Crate](https://img.shields.io/crates/v/ewald.svg)](https://crates.io/crates/ewald)
[![Docs](https://docs.rs/lin_alg/badge.svg)](https://docs.rs/ewald)

[//]: # ([![DOI]&#40;https://zenodo.org/badge/DOI/10.5281/zenodo.15616833.svg&#41;]&#40;https://doi.org/10.5281/zenodo.15616833&#41;)

This has applications primarily in structural biology. For example, molecular dynamics. Compared to other
n-body approximations for long-range forces, this has utility when periodic bounday conditions are used.
If not using these, for example in cosmology simulations, consider Barnes Hut, or Fast Multipole Methods (FMM)
instead.

Used by the [Daedalus protein viewer and molecular dynamics program](https://github.com/david-oconnor/daedalus).

Here's an example of use:

```rust


```