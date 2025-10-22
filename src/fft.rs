//! CPU FFT setup

use realfft::RealFftPlanner;
use rustfft::FftPlanner;
use crate::Complex_;

/// Real to Complex FFT. This approach uses less memory, and is probably faster,
/// than using complex to complex. (Factor of 2 for the memory)
pub(crate) fn fft3_r2c(
    data_r: &mut [f32],
    dims: (usize, usize, usize),
    planner: &mut FftPlanner<f32>,
) -> Vec<Complex_> {
    let (nx, ny, nz) = dims;
    let nxc = nx / 2 + 1;

    let mut rplanner = RealFftPlanner::<f32>::new();
    let r2c_x = rplanner.plan_fft_forward(nx);

    let fft_y = planner.plan_fft_forward(ny);
    let fft_z = planner.plan_fft_forward(nz);

    let mut out = vec![Complex_::new(0.0, 0.0); nxc * ny * nz];

    // X: R2C rows
    // let mut scratch_r = r2c_x.make_scratch_vec();
    for iz in 0..nz {
        for iy in 0..ny {
            let row_r = iz * (nx * ny) + iy * nx;
            let row_k = iz * (nxc * ny) + iy * nxc;
            let in_row = &mut data_r[row_r..row_r + nx];
            let out_row = &mut out[row_k..row_k + nxc];

            r2c_x.process(in_row, out_row).unwrap();
        }
    }

    // Y: C2C columns
    {
        let mut tmp = vec![Complex_::new(0.0, 0.0); ny];
        for iz in 0..nz {
            for ixc in 0..nxc {
                for (j, iy) in (0..ny).enumerate() {
                    tmp[j] = out[iz * (nxc * ny) + iy * nxc + ixc];
                }
                fft_y.process(&mut tmp);
                for (j, iy) in (0..ny).enumerate() {
                    out[iz * (nxc * ny) + iy * nxc + ixc] = tmp[j];
                }
            }
        }
    }

    // Z: C2C columns
    {
        let mut tmp = vec![Complex_::new(0.0, 0.0); nz];
        for iy in 0..ny {
            for ixc in 0..nxc {
                for (k, iz) in (0..nz).enumerate() {
                    tmp[k] = out[iz * (nxc * ny) + iy * nxc + ixc];
                }

                fft_z.process(&mut tmp);
                for (k, iz) in (0..nz).enumerate() {
                    out[iz * (nxc * ny) + iy * nxc + ixc] = tmp[k];
                }
            }
        }
    }

    out
}

/// Minimal 3D C2R using rustfft: inverse C2C along Z and Y, then C2R along X.
pub(crate) fn fft3_c2r(
    data_k: &mut [Complex_],
    dims: (usize, usize, usize),
    planner: &mut FftPlanner<f32>,
) -> Vec<f32> {
    let (nx, ny, nz) = dims;
    let nxc = nx / 2 + 1;

    let mut rplanner = RealFftPlanner::<f32>::new();
    let c2r_x = rplanner.plan_fft_inverse(nx);

    let ifft_y = planner.plan_fft_inverse(ny);
    let ifft_z = planner.plan_fft_inverse(nz);

    // inverse Z: C2C
    {
        let mut tmp = vec![Complex_::new(0.0, 0.0); nz];
        for iy in 0..ny {
            for ixc in 0..nxc {
                for (k, iz) in (0..nz).enumerate() {
                    tmp[k] = data_k[iz * (nxc * ny) + iy * nxc + ixc];
                }
                ifft_z.process(&mut tmp);
                for (k, iz) in (0..nz).enumerate() {
                    data_k[iz * (nxc * ny) + iy * nxc + ixc] = tmp[k];
                }
            }
        }
    }

    // inverse Y: C2C
    {
        let mut tmp = vec![Complex_::new(0.0, 0.0); ny];
        for iz in 0..nz {
            for ixc in 0..nxc {
                for (j, iy) in (0..ny).enumerate() {
                    tmp[j] = data_k[iz * (nxc * ny) + iy * nxc + ixc];
                }
                ifft_y.process(&mut tmp);
                for (j, iy) in (0..ny).enumerate() {
                    data_k[iz * (nxc * ny) + iy * nxc + ixc] = tmp[j];
                }
            }
        }
    }

    // X: C2R rows
    let mut out = vec![0.0f32; nx * ny * nz];
    // let mut scratch_r = c2r_x.make_scratch_vec();
    for iz in 0..nz {
        for iy in 0..ny {
            let row_k = iz * (nxc * ny) + iy * nxc;
            let row_r = iz * (nx * ny) + iy * nx;
            let in_row = &mut data_k[row_k..row_k + nxc];
            let out_row = &mut out[row_r..row_r + nx];

            c2r_x.process(in_row, out_row).unwrap();
        }
    }

    out
}