//! Used by both vkFFT and cuFFT pipelines. Computetes long-range reciprical forces on the GPU,
//! using a mix of our kernels, and host-FFI-initiated FFTs.

use std::{ffi::c_void, sync::Arc};

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use lin_alg::f32::{Vec3, vec3s_to_dev};

#[cfg(feature = "cufft")]
use crate::cufft;
#[cfg(feature = "vkfft")]
use crate::vk_fft;
use crate::{PmeRecip, self_energy};

pub(crate) struct Kernels {
    pub kernel_spread: CudaFunction,
    pub kernel_ghat: CudaFunction,
    pub kernel_gather: CudaFunction,
    pub kernel_half_spectrum: CudaFunction,
}

pub(crate) struct GpuTables {
    pub kx: CudaSlice<f32>,
    pub ky: CudaSlice<f32>,
    pub kz: CudaSlice<f32>,
    pub bx: CudaSlice<f32>,
    pub by: CudaSlice<f32>,
    pub bz: CudaSlice<f32>,
}

impl GpuTables {
    pub(crate) fn new(
        k: (&Vec<f32>, &Vec<f32>, &Vec<f32>),
        bmod2: (&Vec<f32>, &Vec<f32>, &Vec<f32>),
        stream: &Arc<CudaStream>,
    ) -> Self {
        Self {
            kx: stream.memcpy_stod(k.0).unwrap(),
            ky: stream.memcpy_stod(k.1).unwrap(),
            kz: stream.memcpy_stod(k.2).unwrap(),
            bx: stream.memcpy_stod(bmod2.0).unwrap(),
            by: stream.memcpy_stod(bmod2.1).unwrap(),
            bz: stream.memcpy_stod(bmod2.2).unwrap(),
        }
    }
}

impl PmeRecip {
    /// Compute reciprocal-space forces on all positions, using the GPU.
    /// Note: We spread charges, and do other procedures on GPU that are already fast on the CPU.
    /// We handle it this way to prevent transfering more info to and from the GPU than required.
    pub fn forces_gpu(
        &mut self,
        stream: &Arc<CudaStream>,
        posits: &[Vec3],
        q: &[f32],
    ) -> (Vec<Vec3>, f32) {
        assert_eq!(posits.len(), q.len());

        let (nx, ny, nz) = self.plan_dims;

        let n_real = nx * ny * nz;
        let nzc = nz / 2 + 1;
        let n_cplx = nx * ny * nzc;

        let complex_len = n_cplx * 2; // (re,im) interleaved

        // ---------- Allocate arrays on the GPU

        // todo: Should we init these once and store, instead of re-allocating at each step?
        // Set up positions, rho, and charge on the GPU once; they'll be used a few times in this function.
        let pos_dev = vec3s_to_dev(stream, posits);
        let q_dev = stream.memcpy_stod(q).unwrap();

        // let mut rho_real_dev: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
        // let mut rho_cplx_dev: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();

        // GPU buffers of complex numbers are flattened.

        // Charge density.
        let mut rho_real_dev: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
        let mut rho_dev: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();

        spread_charges(
            stream,
            &self.kernels.kernel_spread,
            &pos_dev,
            &q_dev,
            &mut rho_real_dev,
            self.plan_dims,
            self.box_dims,
            posits.len() as u32,
        );

        // Convert the spread charges to K space. They will be complex, and in the frequency domain.
        unsafe {
            exec_forward(
                self.planner_gpu,
                cuda_slice_to_ptr_mut(&rho_real_dev, stream),
                cuda_slice_to_ptr_mut(&rho_dev, stream),
            );
        }

        // {
        //     let rho_cpu = stream.memcpy_dtov(&rho_dev).unwrap();
        //     let mut rho_dbg = Vec::new();
        //     for i in 0..complex_len / 2 {
        //         rho_dbg.push(Complex::new(rho_cpu[2 * i], rho_cpu[2 * i + 1]));
        //     }
        //
        //     println!("\n");
        //     for i in 220..230 {
        //         println!("rho GPU post fwd FFT: {:?}", rho_dbg[i])
        //     }
        // }

        // todo: Pre-allocate these instead of every step?
        // Contiguous complex buffer: [exk | eyk | ezk]
        // let ekx_eky_ekz_gpu: CudaSlice<f32> = stream.alloc_zeros(3 * complex_len).unwrap();
        let mut exk_dev: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();
        let mut eyk_dev: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();
        let mut ezk_dev: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();

        // k-space pointers for FFT FFI)
        let ekx_ptr = cuda_slice_to_ptr_mut(&exk_dev, stream);
        let eky_ptr = cuda_slice_to_ptr_mut(&eyk_dev, stream);
        let ekz_ptr = cuda_slice_to_ptr_mut(&ezk_dev, stream);

        // Apply G(k) and gradient to get Exk/Eyk/Ezk
        apply_ghat_and_grad(
            stream,
            &self.kernels.kernel_ghat,
            &rho_dev,
            &mut exk_dev,
            &mut eyk_dev,
            &mut ezk_dev,
            &self.gpu_tables,
            self.plan_dims,
            self.vol,
            self.alpha,
        );

        // {
        //     let exk = stream.memcpy_dtov(&exk_dev).unwrap();
        //     let eyk = stream.memcpy_dtov(&eyk_dev).unwrap();
        //     let mut exk_dbg = Vec::new();
        //     let mut eyk_dbg = Vec::new();
        //
        //     for i in 0..complex_len / 2 {
        //         exk_dbg.push(Complex::new(exk[2 * i], exk[2 * i + 1]));
        //         eyk_dbg.push(Complex::new(eyk[2 * i], eyk[2 * i + 1]));
        //     }
        //     println!("\n");
        //     for i in 220..230 {
        //         println!("exk GPU post GHAT: {:?}", exk_dbg[i]);
        //     }
        //     println!("\n");
        //     // for i in 220..230 {
        //     //     println!("eyk GPU post GHAT: {:?}", eyk_dbg[i]);
        //     // }
        // }

        // todo: Qc this! Not sure what it should be.
        let mut out_partial_gpu: CudaSlice<f64> = stream.alloc_zeros(n_cplx).unwrap();

        energy_half_spectrum(
            stream,
            &self.kernels.kernel_half_spectrum,
            &mut rho_dev,
            &mut out_partial_gpu,
            &self.gpu_tables,
            self.plan_dims,
            self.vol,
            self.alpha,
        );

        let mut energy: f64 = stream
            .memcpy_dtov(&out_partial_gpu)
            .unwrap()
            .into_iter()
            .sum();

        let energy = (energy + self_energy(q, self.alpha)) as f32;

        // println!("\n Energy GPU: {:?}", energy);

        let ex_dev: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
        let ey_dev: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
        let ez_dev: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();

        let ex_ptr = cuda_slice_to_ptr_mut(&ex_dev, stream);
        let ey_ptr = cuda_slice_to_ptr_mut(&ey_dev, stream);
        let ez_ptr = cuda_slice_to_ptr_mut(&ez_dev, stream);

        unsafe {
            exec_inverse(
                self.planner_gpu,
                ekx_ptr,
                eky_ptr,
                ekz_ptr,
                ex_ptr,
                ey_ptr,
                ez_ptr,
            );
        }

        // {
        //     let ex_ = stream.memcpy_dtov(&ex_dev).unwrap();
        //     let ey_ = stream.memcpy_dtov(&ey_dev).unwrap();
        //
        //     println!("\n");
        //     for i in 220..230 {
        //         println!("exk GPU post inv FFT: {:?}", ex_[i]);
        //     }
        //     println!("\n");
        //     for i in 220..230 {
        //         println!("eyk GPU post inv FFT: {:?}", ey_[i]);
        //     }
        // }

        // {
        //     let ex_ = stream.memcpy_dtov(&ex_dev).unwrap();
        //     let ey_ = stream.memcpy_dtov(&ey_dev).unwrap();
        //
        //     println!("\n");
        //     for i in 220..230 {
        //         println!("exk GPU post inv FFT and scale: {:?}", ex_[i]);
        //     }
        //     println!("\n");
        //     for i in 220..230 {
        //         println!("eyk GPU post inv FFT and scale: {:?}", ey_[i]);
        //     }
        // }

        let n_atoms = posits.len();
        let mut out_f_gpu: CudaSlice<f32> = stream.alloc_zeros(3 * n_atoms).unwrap();

        gather_forces_to_atoms(
            stream,
            &self.kernels.kernel_gather,
            &pos_dev,
            &q_dev,
            &ex_dev,
            &ey_dev,
            &ez_dev,
            &mut out_f_gpu,
            self.plan_dims,
            self.box_dims,
        );

        // D2H forces
        let f_host: Vec<f32> = stream.memcpy_dtov(&out_f_gpu).unwrap();

        // todo: QC the - sign?
        let mut f = Vec::with_capacity(posits.len());
        for i in 0..posits.len() {
            f.push(-Vec3 {
                x: f_host[i * 3 + 0],
                y: f_host[i * 3 + 1],
                z: f_host[i * 3 + 2],
            });
        }

        (f, energy)
    }
}

// FFI for GPU FFT functions. These signatures are the same for cuFFT and vkFFT, so we use
// them for both.
unsafe extern "C" {
    pub(crate) fn exec_forward(plan: *mut c_void, rho_real: *mut c_void, rho: *mut c_void);

    pub(crate) fn exec_inverse(
        plan: *mut c_void,
        exk: *mut c_void,
        eyk: *mut c_void,
        ezk: *mut c_void,
        ex: *mut c_void,
        ey: *mut c_void,
        ez: *mut c_void,
    );
}

impl Drop for PmeRecip {
    fn drop(&mut self) {
        unsafe {
            if !self.planner_gpu.is_null() {
                #[cfg(feature = "vkfft")]
                vk_fft::destroy_plan(self.planner_gpu);
                #[cfg(feature = "cufft")]
                cufft::destroy_plan(self.planner_gpu);
                self.planner_gpu = std::ptr::null_mut();
            }
        }
    }
}

pub(crate) fn cuda_slice_to_ptr<T>(buf: &CudaSlice<T>, stream: &Arc<CudaStream>) -> *const c_void {
    let (p, _) = buf.device_ptr(stream);
    p as *const c_void
}

pub(crate) fn cuda_slice_to_ptr_mut<T>(
    buf: &CudaSlice<T>,
    stream: &Arc<CudaStream>,
) -> *mut c_void {
    let (p, _) = buf.device_ptr(stream);
    p as *mut c_void
}

pub(crate) fn split3(
    buf: &CudaSlice<f32>,
    len: usize,
    stream: &Arc<CudaStream>,
) -> (*mut c_void, *mut c_void, *mut c_void) {
    let (base, _) = buf.device_ptr(stream);
    let base = base as usize;

    let stride = len * size_of::<f32>();
    (
        base as *mut c_void,
        (base + stride) as *mut c_void,
        (base + 2 * stride) as *mut c_void,
    )
}

/// Launch the GPU kernel that spreads charges.
/// todo note: Getting the same values as on CPU here.
fn spread_charges(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    // posit, q and rho are passed as CudaSlices, as they're used elsewhere in the flow.
    pos_dev: &CudaSlice<f32>,
    q_dev: &CudaSlice<f32>,
    rho_dev: &mut CudaSlice<f32>, // real only.
    plan_dims: (usize, usize, usize),
    box_dims: (f32, f32, f32),
    n_posits: u32,
) {
    let (nx, ny, nz) = plan_dims;
    let nx_i = nx as i32;
    let ny_i = ny as i32;
    let nz_i = nz as i32;

    let (lx, ly, lz) = box_dims;

    let n_atoms_i = n_posits as i32;

    let cfg = launch_cfg(n_posits, 256);

    let mut launch_args = stream.launch_builder(kernel);

    launch_args.arg(pos_dev);
    launch_args.arg(q_dev);
    launch_args.arg(rho_dev);

    launch_args.arg(&n_atoms_i);

    launch_args.arg(&nx_i);
    launch_args.arg(&ny_i);
    launch_args.arg(&nz_i);
    launch_args.arg(&lx);
    launch_args.arg(&ly);
    launch_args.arg(&lz);

    unsafe { launch_args.launch(cfg) }.unwrap();
}

/// See notes on the CPU equivalent.
fn apply_ghat_and_grad(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    rho_dev: &CudaSlice<f32>, // Cplx
    ekx_dev: &mut CudaSlice<f32>,
    eky_dev: &mut CudaSlice<f32>,
    ekz_dev: &mut CudaSlice<f32>,
    tables: &GpuTables,
    plan_dims: (usize, usize, usize),
    vol: f32,
    alpha: f32,
) {
    let (nx, ny, nz) = plan_dims;
    let nx_i = nx as i32;
    let ny_i = ny as i32;
    let nz_i = nz as i32;

    let n = nx * ny * (nz / 2 + 1);
    let n_real = (nx * ny * nz) as i32;

    // let cfg = LaunchConfig::for_num_elems(n as u32);
    let cfg = launch_cfg(n as u32, 256);
    let mut launch_args = stream.launch_builder(kernel);

    launch_args.arg(rho_dev);

    launch_args.arg(ekx_dev);
    launch_args.arg(eky_dev);
    launch_args.arg(ekz_dev);

    launch_args.arg(&tables.kx);
    launch_args.arg(&tables.ky);
    launch_args.arg(&tables.kz);
    launch_args.arg(&tables.bx);
    launch_args.arg(&tables.by);
    launch_args.arg(&tables.bz);

    launch_args.arg(&nx_i);
    launch_args.arg(&ny_i);
    launch_args.arg(&nz_i);

    launch_args.arg(&vol);
    launch_args.arg(&alpha);

    launch_args.arg(&n_real);

    unsafe { launch_args.launch(cfg) }.unwrap();
}

/// See notes on the CPU equivalent.
fn energy_half_spectrum(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    rho_cplx_gpu: &mut CudaSlice<f32>,
    out_partial_gpu: &mut CudaSlice<f64>,
    tables: &GpuTables,
    plan_dims: (usize, usize, usize),
    vol: f32,
    alpha: f32,
) {
    let (nx, ny, nz) = plan_dims;
    let nx_i = nx as i32;
    let ny_i = ny as i32;
    let nz_i = nz as i32;

    let n = (nx * ny * (nz / 2 + 1)) as i32;

    let block: u32 = 256;
    let grid: u32 = ((n as u32) + block - 1) / block;

    // let cfg = LaunchConfig::for_num_elems(n as u32);
    let cfg = launch_cfg(n as u32, 256);

    let mut launch_args = stream.launch_builder(kernel);

    launch_args.arg(rho_cplx_gpu);

    launch_args.arg(&tables.kx);

    launch_args.arg(&tables.ky);
    launch_args.arg(&tables.kz);
    launch_args.arg(&tables.bx);
    launch_args.arg(&tables.by);
    launch_args.arg(&tables.bz);

    launch_args.arg(&nx_i);
    launch_args.arg(&ny_i);
    launch_args.arg(&nz_i);

    launch_args.arg(&vol);
    launch_args.arg(&alpha);

    launch_args.arg(out_partial_gpu);

    unsafe { launch_args.launch(cfg) }.unwrap();
}

/// See notes on the CPU equivalent.
fn gather_forces_to_atoms(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    pos_gpu: &CudaSlice<f32>,
    q_gpu: &CudaSlice<f32>,
    ex_gpu: &CudaSlice<f32>,
    ey_gpu: &CudaSlice<f32>,
    ez_gpu: &CudaSlice<f32>,
    out_partial_gpu: &mut CudaSlice<f32>,
    plan_dims: (usize, usize, usize),
    box_dims: (f32, f32, f32),
) {
    let (nx, ny, nz) = plan_dims;
    let nx_i = nx as i32;
    let ny_i = ny as i32;
    let nz_i = nz as i32;

    let (lx, ly, lz) = box_dims;

    let n = pos_gpu.len() / 3;
    let n_u32 = n as u32;
    let n_i32 = n as i32;

    // let cfg = LaunchConfig::for_num_elems(n_u32);
    let cfg = launch_cfg(n_u32, 256);
    let mut launch_args = stream.launch_builder(kernel);

    launch_args.arg(pos_gpu);

    launch_args.arg(ex_gpu);
    launch_args.arg(ey_gpu);
    launch_args.arg(ez_gpu);
    launch_args.arg(q_gpu);

    launch_args.arg(out_partial_gpu);

    launch_args.arg(&n_i32);

    launch_args.arg(&nx_i);
    launch_args.arg(&ny_i);
    launch_args.arg(&nz_i);
    launch_args.arg(&lx);
    launch_args.arg(&ly);
    launch_args.arg(&lz);

    unsafe { launch_args.launch(cfg) }.unwrap();
}

/// If we run `LaunchConfig::from_num_elems` for certain kernels, we get the error
/// `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`.
fn launch_cfg(n: u32, block: u32) -> LaunchConfig {
    let grid = (n + block - 1) / block; // ceil_div
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}
