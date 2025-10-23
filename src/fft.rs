//! CPU FFT setup

use realfft::RealFftPlanner;
use rustfft::FftPlanner;

use crate::Complex_;

/// Real to Complex FFT. This approach uses less memory, and is probably faster,
/// than using complex to complex. (Factor of 2 for the memory).
///
/// Z is the contiguous (fast) dimension; X is the strided (slow) one. This is chosen
/// to be consistnt with cuFFT's `Plan3D`'s conventions.
pub(crate) fn fft3_r2c(
    data_r: &mut [f32],
    dims: (usize, usize, usize),
    planner: &mut FftPlanner<f32>,
) -> Vec<Complex_> {
    let (nx, ny, nz) = dims;

    let nzc = nz / 2 + 1;
    let n_cplx = nx * ny * nzc;

    let mut rplanner = RealFftPlanner::<f32>::new();
    let r2c_z = rplanner.plan_fft_forward(nz);

    let fft_y = planner.plan_fft_forward(ny);
    let fft_x = planner.plan_fft_forward(nx);

    let mut out = vec![Complex_::new(0.0, 0.0); n_cplx];

    // Z: R2C rows (contiguous)
    for ix in 0..nx {
        for iy in 0..ny {
            let row_r = ix * (ny * nz) + iy * nz;
            let row_k = ix * (ny * nzc) + iy * nzc;
            let in_row = &mut data_r[row_r..row_r + nz];
            let out_row = &mut out[row_k..row_k + nzc];
            r2c_z.process(in_row, out_row).unwrap();
        }
    }

    // Y: C2C columns
    {
        let mut tmp = vec![Complex_::new(0.0, 0.0); ny];
        for ix in 0..nx {
            for izc in 0..nzc {
                for (j, iy) in (0..ny).enumerate() {
                    tmp[j] = out[ix * (ny * nzc) + iy * nzc + izc];
                }
                fft_y.process(&mut tmp);
                for (j, iy) in (0..ny).enumerate() {
                    out[ix * (ny * nzc) + iy * nzc + izc] = tmp[j];
                }
            }
        }
    }

    // X: C2C columns (most strided)
    {
        let mut tmp = vec![Complex_::new(0.0, 0.0); nx];
        for iy in 0..ny {
            for izc in 0..nzc {
                for (k, ix) in (0..nx).enumerate() {
                    tmp[k] = out[ix * (ny * nzc) + iy * nzc + izc];
                }
                fft_x.process(&mut tmp);
                for (k, ix) in (0..nx).enumerate() {
                    out[ix * (ny * nzc) + iy * nzc + izc] = tmp[k];
                }
            }
        }
    }

    out
}

// todo: Why is data_k mut, and we return something? Real part mutated ends up as the returned value.
pub(crate) fn fft3_c2r(
    data_k: &mut [Complex_],
    dims: (usize, usize, usize),
    planner: &mut FftPlanner<f32>,
) -> Vec<f32> {
    let (nx, ny, nz) = dims;
    let nzc = nz / 2 + 1;

    let mut rplanner = RealFftPlanner::<f32>::new();
    let c2r_z = rplanner.plan_fft_inverse(nz);

    let ifft_y = planner.plan_fft_inverse(ny);
    let ifft_x = planner.plan_fft_inverse(nx);

    // inverse X: C2C
    {
        let mut tmp = vec![Complex_::new(0.0, 0.0); nx];
        for iy in 0..ny {
            for izc in 0..nzc {
                for (k, ix) in (0..nx).enumerate() {
                    tmp[k] = data_k[ix * (ny * nzc) + iy * nzc + izc];
                }
                ifft_x.process(&mut tmp);
                for (k, ix) in (0..nx).enumerate() {
                    data_k[ix * (ny * nzc) + iy * nzc + izc] = tmp[k];
                }
            }
        }
    }

    // inverse Y: C2C
    {
        let mut tmp = vec![Complex_::new(0.0, 0.0); ny];

        for ix in 0..nx {
            for izc in 0..nzc {
                for (j, iy) in (0..ny).enumerate() {
                    tmp[j] = data_k[ix * (ny * nzc) + iy * nzc + izc];
                }
                ifft_y.process(&mut tmp);
                for (j, iy) in (0..ny).enumerate() {
                    data_k[ix * (ny * nzc) + iy * nzc + izc] = tmp[j];
                }
            }
        }
    }

    // Z: C2R rows (contiguous)
    let mut out = vec![0.; nx * ny * nz];
    for ix in 0..nx {
        for iy in 0..ny {
            let row_k = ix * (ny * nzc) + iy * nzc;
            let row_r = ix * (ny * nz) + iy * nz;
            let in_row = &mut data_k[row_k..row_k + nzc];
            let out_row = &mut out[row_r..row_r + nz];

            // Enforce real-signal constraint at DC / Nyquist along Z
            in_row[0].im = 0.0;
            if nz % 2 == 0 {
                in_row[nzc - 1].im = 0.0;
            }

            c2r_z.process(in_row, out_row).unwrap();
        }
    }

    out
}
