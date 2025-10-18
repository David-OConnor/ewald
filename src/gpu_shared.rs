//! Used by both vkFFT and cuFFT pipelines.

use std::{ffi::c_void, sync::Arc};

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg,
};
use lin_alg::f32::{Vec3, vec3s_to_dev};

use crate::PmeRecip;
#[cfg(feature = "cufft")]
use crate::cufft;
#[cfg(feature = "vkfft")]
use crate::vk_fft;

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
    pub fn forces_gpu(
        &mut self,
        #[cfg(feature = "vkfft")] ctx: &Arc<vk_fft::VkContext>,
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        pos: &[Vec3],
        q: &[f32],
    ) -> (Vec<Vec3>, f32) {
        assert_eq!(pos.len(), q.len());

        // We set these up here instead of at init, so we have the stream, and so
        // init isn't dependent on a stream being passed.
        if self.gpu_tables.is_none() {
            // First run
            let k = (&self.kx, &self.ky, &self.kz);
            let bmod2 = (&self.bmod2_x, &self.bmod2_y, &self.bmod2_z);

            #[cfg(feature = "cufft")]
            {
                self.planner_gpu = cufft::create_gpu_plan(self.plan_dims, stream);
            }
            #[cfg(feature = "vkfft")]
            {
                self.planner_gpu = vk_fft::create_gpu_plan(self.plan_dims, ctx);
            }

            self.gpu_tables = Some(GpuTables::new(k, bmod2, stream));
        }

        let cu_stream = stream.cu_stream() as *mut c_void;
        let tables = self.gpu_tables.as_ref().unwrap();

        let (nx, ny, nz) = self.plan_dims;

        let n_real = nx * ny * nz;
        let n_cplx = nx * ny * (nz / 2 + 1); // half-spectrum length
        let complex_len = n_cplx * 2; // (re,im) interleaved

        let kx_ptr = cuda_slice_to_ptr(&tables.kx, stream);
        let ky_ptr = cuda_slice_to_ptr(&tables.ky, stream);
        let kz_ptr = cuda_slice_to_ptr(&tables.kz, stream);

        let bx_ptr = cuda_slice_to_ptr(&tables.bx, stream);
        let by_ptr = cuda_slice_to_ptr(&tables.by, stream);
        let bz_ptr = cuda_slice_to_ptr(&tables.bz, stream);

        // todo: Can we create this once in the PmeRecip struct, then call it,
        // todo instead of re-allocating each step?

        // todo: Store these eventually. Not too big of a deal to load here though.
        let kernel_spread = module.load_function("spread_charges").unwrap();
        let kernel_ghat = module.load_function("apply_ghat_and_grad").unwrap();
        let kernel_scale = module.load_function("scale_vec").unwrap();
        let kernel_gather = module.load_function("gather_forces_to_atoms").unwrap();
        let kernel_half_spectrum = module.load_function("energy_half_spectrum").unwrap();

        // todo: Should we init these once and store, instead of re-allocating at each step?
        // rho_real on device
        let pos_gpu = vec3s_to_dev(stream, pos);
        let mut rho_real_gpu: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
        let q_gpu = stream.memcpy_stod(q).unwrap();

        // todo: Confirm this gets populated (Probably by the FFT?)
        // rho(k) (half-spectrum) on device
        let mut rho_cplx_gpu: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();

        spread_charges(
            stream,
            &kernel_spread,
            &pos_gpu,
            &q_gpu,
            &mut rho_real_gpu,
            self.plan_dims,
            self.box_dims,
            pos.len() as u32,
        );

        // Forward FFT on GPU: real -> half complex
        unsafe {
            exec_forward_r2c(
                self.planner_gpu,
                cuda_slice_to_ptr_mut(&rho_real_gpu, stream),
                cuda_slice_to_ptr_mut(&rho_cplx_gpu, stream),
            );
        }

        // todo: Can we maintain these in memory instead of re-allocating each time?
        // Contiguous complex buffer: [exk | eyk | ezk]
        // let ekx_eky_ekz_gpu: CudaSlice<f32> = stream.alloc_zeros(3 * complex_len).unwrap();
        let mut ekx_gpu: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();
        let mut eky_gpu: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();
        let mut ekz_gpu: CudaSlice<f32> = stream.alloc_zeros(complex_len).unwrap();

        // k-space pointers for FFT FFI)
        let ekx_ptr = cuda_slice_to_ptr_mut(&ekx_gpu, stream);
        let eky_ptr = cuda_slice_to_ptr_mut(&eky_gpu, stream);
        let ekz_ptr = cuda_slice_to_ptr_mut(&ekz_gpu, stream);

        // // todo: Pre-allocate these instead of every step?
        let mut ex_gpu: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
        let mut ey_gpu: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();
        let mut ez_gpu: CudaSlice<f32> = stream.alloc_zeros(n_real).unwrap();

        let ex_ptr = cuda_slice_to_ptr_mut(&ex_gpu, stream);
        let ey_ptr = cuda_slice_to_ptr_mut(&ey_gpu, stream);
        let ez_ptr = cuda_slice_to_ptr_mut(&ez_gpu, stream);

        // Apply G(k) and gradient to get Exk/Eyk/Ezk
        apply_ghat_and_grad(
            stream,
            &kernel_ghat,
            &mut rho_cplx_gpu,
            &mut ekx_gpu,
            &mut eky_gpu,
            &mut ekz_gpu,
            tables,
            self.plan_dims,
            self.vol,
            self.alpha,
        );

        unsafe {
            // Inverse batched C2R: (exk,eyk,ezk) -> (ex,ey,ez)
            exec_inverse_ExEyEz_c2r(
                self.planner_gpu,
                ekx_ptr,
                eky_ptr,
                ekz_ptr,
                ex_ptr,
                ey_ptr,
                ez_ptr,
            );
        }

        let n_real = nx * ny * nz;
        let inv_n = 1.0f32 / (n_real as f32);

        scale_vec(stream, &kernel_scale, &mut ex_gpu, inv_n);
        scale_vec(stream, &kernel_scale, &mut ey_gpu, inv_n);
        scale_vec(stream, &kernel_scale, &mut ez_gpu, inv_n);

        let n_atoms = pos.len();
        let mut out_f_gpu: CudaSlice<f32> = stream.alloc_zeros(3 * n_atoms).unwrap();

        gather_forces_to_atoms(
            stream,
            &kernel_gather,
            &pos_gpu,
            &q_gpu,
            &ex_gpu,
            &ey_gpu,
            &ez_gpu,
            &mut out_f_gpu,
            self.plan_dims,
            self.box_dims,
        );

        // todo: Qc this! Not sure what it should be.
        let mut out_partial_gpu: CudaSlice<f64> = stream.alloc_zeros(n_cplx).unwrap();

        energy_half_spectrum(
            stream,
            &kernel_half_spectrum,
            &mut rho_cplx_gpu,
            &mut out_partial_gpu,
            tables,
            self.plan_dims,
            self.vol,
            self.alpha,
        );

        let energy = stream
            .memcpy_dtov(&out_partial_gpu)
            .unwrap()
            .into_iter()
            .sum::<f64>() as f32;

        // D2H forces
        let f_host: Vec<f32> = stream.memcpy_dtov(&out_f_gpu).unwrap();
        let mut f = Vec::with_capacity(pos.len());
        for i in 0..pos.len() {
            f.push(Vec3 {
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
    pub(crate) fn exec_forward_r2c(plan: *mut c_void, rho_real: *mut c_void, rho_k: *mut c_void);

    pub(crate) fn exec_inverse_ExEyEz_c2r(
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
                cufft::destroy_plan_r2c_c2r_many(self.planner_gpu);
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
fn spread_charges(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    // posit, q and rho are passed as CudaSlices, as they're used elsewhere in the flow.
    pos_gpu: &CudaSlice<f32>,
    q_gpu: &CudaSlice<f32>,
    rho_real_gpu: &mut CudaSlice<f32>,
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

    let cfg = launch_cfg(n_posits as u32, 256);

    let mut launch_args = stream.launch_builder(kernel);

    launch_args.arg(pos_gpu);
    launch_args.arg(q_gpu);
    launch_args.arg(rho_real_gpu);


    launch_args.arg(&n_atoms_i);

    launch_args.arg(&nx_i);
    launch_args.arg(&ny_i);
    launch_args.arg(&nz_i);
    launch_args.arg(&lx);
    launch_args.arg(&ly);
    launch_args.arg(&lz);

    unsafe { launch_args.launch(cfg) }.unwrap();
}

/// Launch the GPU kernel;  Apply G(k) and gradient to get Exk/Eyk/Ezk
fn apply_ghat_and_grad(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    rho_cplx_gpu: &mut CudaSlice<f32>,
    ekx_gpu: &mut CudaSlice<f32>,
    eky_gpu: &mut CudaSlice<f32>,
    ekz_gpu: &mut CudaSlice<f32>,
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

    // let cfg = LaunchConfig::for_num_elems(n as u32);
    let cfg = launch_cfg(n as u32, 256);
    let mut launch_args = stream.launch_builder(kernel);

    launch_args.arg(rho_cplx_gpu);

    launch_args.arg(ekx_gpu);
    launch_args.arg(eky_gpu);
    launch_args.arg(ekz_gpu);

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

    unsafe { launch_args.launch(cfg) }.unwrap();
}

/// Launch the GPU kernel.
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

/// Launch the GPU kernel.
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


    // todo: QC if this is the right n!
    let n = pos_gpu.len() / 3; // todo: QC!
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


fn scale_vec(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    buf: &mut CudaSlice<f32>,
    s: f32,
) {
    let n_i = buf.len() as i32;

    let cfg = LaunchConfig::for_num_elems(n_i as u32);

    let mut lb = stream.launch_builder(kernel);
    lb.arg(buf);
    lb.arg(&n_i);
    lb.arg(&s);
    unsafe { lb.launch(cfg) }.unwrap();
}

/// If we run `LaunchConfig::from_num_elems`, we get the error `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`.
fn launch_cfg(n: u32, block: u32) -> LaunchConfig {
    let grid = (n + block - 1) / block; // ceil_div
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}