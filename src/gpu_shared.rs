//! Used by both vkFFT and cuFFT pipelines. Computetes long-range reciprical forces on the GPU,
//! using a mix of our kernels, and host-FFI-initiated FFTs.

use std::{ffi::c_void, mem::size_of, sync::Arc};

use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DeviceRepr, HostSlice, LaunchConfig,
    PushKernelArg, SyncOnDrop, result,
};
use lin_alg::f32::Vec3;

use crate::{
    PmeRecip,
    fft::{destroy_plan, exec_forward, exec_inverse},
    self_energy,
};

/// Reusable page-locked host storage with normal CPU cacheability. cudarc's
/// built-in pinned allocation is write-combined, which is unsuitable for the
/// force and energy buffers read back by the CPU.
struct PinnedBuffer<T: DeviceRepr> {
    ptr: *mut T,
    len: usize,
    stream: Arc<CudaStream>,
}

unsafe impl<T: DeviceRepr> Send for PinnedBuffer<T> {}
unsafe impl<T: DeviceRepr> Sync for PinnedBuffer<T> {}

impl<T: DeviceRepr> PinnedBuffer<T> {
    fn new(stream: &Arc<CudaStream>, len: usize) -> Self {
        let ptr = unsafe { result::malloc_host(len * size_of::<T>(), 0).unwrap() } as *mut T;
        assert!(!ptr.is_null());
        Self {
            ptr,
            len,
            stream: stream.clone(),
        }
    }

    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: DeviceRepr> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        let _ = unsafe { result::free_host(self.ptr.cast()) };
    }
}

impl<T: DeviceRepr> HostSlice<T> for PinnedBuffer<T> {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self.as_slice(), SyncOnDrop::Sync(None))
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self.as_mut_slice(), SyncOnDrop::Sync(None))
    }
}

/// Group GPU-specific state, so they can be made an option as a whole, in the case
/// of compiling with GPU support, but no stream is available.
pub(crate) struct GpuData {
    /// FFI to the CPU planner.
    pub planner_gpu: *mut c_void,
    pub gpu_tables: GpuTables,
    pub kernels: Kernels,
    pub workspace: GpuWorkspace,
    // #[cfg(feature = "vkfft")]
    // pub vk_ctx: Arc<vk_fft::VkContext>,
}

pub(crate) struct Kernels {
    pub kernel_spread: CudaFunction,
    pub kernel_ghat: CudaFunction,
    pub kernel_gather: CudaFunction,
}

/// Mesh-sized buffers persist for the lifetime of the PME plan. Atom-sized
/// buffers are replaced only when the number of particles changes.
pub(crate) struct GpuWorkspace {
    rho_real: CudaSlice<f32>,
    rho: CudaSlice<f32>,
    ek: CudaSlice<f32>,
    e: CudaSlice<f32>,
    energy_partial: CudaSlice<f64>,
    energy_host: PinnedBuffer<f64>,
    atoms: Option<GpuAtomWorkspace>,
}

struct GpuAtomWorkspace {
    len: usize,
    pos_dev: CudaSlice<f32>,
    q_dev: CudaSlice<f32>,
    force_dev: CudaSlice<f32>,
    pos_host: PinnedBuffer<f32>,
    q_host: PinnedBuffer<f32>,
    force_host: PinnedBuffer<f32>,
}

impl GpuWorkspace {
    pub(crate) fn new(plan_dims: (usize, usize, usize), stream: &Arc<CudaStream>) -> Self {
        let (nx, ny, nz) = plan_dims;
        let n_real = nx * ny * nz;
        let n_cplx = nx * ny * (nz / 2 + 1);
        let energy_blocks = n_cplx.div_ceil(256);

        Self {
            rho_real: stream.alloc_zeros(n_real).unwrap(),
            rho: stream.alloc_zeros(2 * n_cplx).unwrap(),
            ek: stream.alloc_zeros(3 * 2 * n_cplx).unwrap(),
            e: stream.alloc_zeros(3 * n_real).unwrap(),
            energy_partial: stream.alloc_zeros(energy_blocks).unwrap(),
            energy_host: PinnedBuffer::new(stream, energy_blocks),
            atoms: None,
        }
    }

    fn ensure_atoms(&mut self, n_atoms: usize, stream: &Arc<CudaStream>) {
        if self
            .atoms
            .as_ref()
            .is_some_and(|atoms| atoms.len == n_atoms)
        {
            return;
        }

        self.atoms = Some(GpuAtomWorkspace {
            len: n_atoms,
            pos_dev: stream.alloc_zeros(3 * n_atoms).unwrap(),
            q_dev: stream.alloc_zeros(n_atoms).unwrap(),
            force_dev: stream.alloc_zeros(3 * n_atoms).unwrap(),
            pos_host: PinnedBuffer::new(stream, 3 * n_atoms),
            q_host: PinnedBuffer::new(stream, n_atoms),
            force_host: PinnedBuffer::new(stream, 3 * n_atoms),
        });
    }
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
            kx: stream.clone_htod(k.0).unwrap(),
            ky: stream.clone_htod(k.1).unwrap(),
            kz: stream.clone_htod(k.2).unwrap(),
            bx: stream.clone_htod(bmod2.0).unwrap(),
            by: stream.clone_htod(bmod2.1).unwrap(),
            bz: stream.clone_htod(bmod2.2).unwrap(),
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
        let Some(data) = &mut self.gpu_data else {
            panic!("Error: Computing forces on GPU without having initialized on GPU");
        };

        assert_eq!(posits.len(), q.len());
        if posits.is_empty() {
            return (Vec::new(), 0.0);
        }

        data.workspace.ensure_atoms(posits.len(), stream);
        let GpuWorkspace {
            rho_real,
            rho,
            ek,
            e,
            energy_partial,
            energy_host,
            atoms,
        } = &mut data.workspace;
        let atoms = atoms.as_mut().unwrap();

        // Pinned staging buffers make these copies genuinely asynchronous. Ordinary
        // Vec-backed copies force cudarc to synchronize the stream after each copy.
        {
            let pos_host = atoms.pos_host.as_mut_slice();
            for (dst, posit) in pos_host.chunks_exact_mut(3).zip(posits) {
                dst.copy_from_slice(&posit.to_arr());
            }
            atoms.q_host.as_mut_slice().copy_from_slice(q);
        }
        stream
            .memcpy_htod(&atoms.pos_host, &mut atoms.pos_dev)
            .unwrap();
        stream.memcpy_htod(&atoms.q_host, &mut atoms.q_dev).unwrap();

        // Only the charge grid needs clearing; every other workspace buffer is
        // completely overwritten by its producing kernel or FFT.
        stream.memset_zeros(rho_real).unwrap();

        spread_charges(
            stream,
            &data.kernels.kernel_spread,
            &atoms.pos_dev,
            &atoms.q_dev,
            rho_real,
            posits.len() as u32,
            self.plan_dims,
            self.box_dims,
        );

        // {
        //     let rho_real = stream.memcpy_dtov(&rho_real_dev).unwrap();
        //     println!("\n");
        //     for i in 0..10 {
        //         println!("POSITS: {:?} Q: {:.3}", posits[i], q[i]);
        //         println!("rho real GPU pre fwd FFT: {:?}", rho_real[i])
        //     }
        // }

        // Convert the spread charges to K space. They will be complex, and in the frequency domain.
        unsafe {
            exec_forward(
                data.planner_gpu,
                cuda_slice_to_ptr_mut(rho_real, stream),
                cuda_slice_to_ptr_mut(rho, stream),
            );

            // #[cfg(feature = "cufft")]
            // exec_r2c(data.planner_gpu, &rho_real_dev, &rho_dev);
        }

        // {
        //     let rho_cpu = stream.memcpy_dtov(&rho_dev).unwrap();
        //     let mut rho_dbg = Vec::new();
        //     for i in 0..complex_len / 2 {
        //         rho_dbg.push(Complex::<f32>::new(rho_cpu[2 * i], rho_cpu[2 * i + 1]));
        //     }
        //
        //     println!("\n");
        //     for i in 220..230 {
        //         println!("rho GPU post fwd FFT: {:?}", rho_dbg[i])
        //     }
        // }

        // Apply G(k), compute gradients, and reduce reciprocal energy in one pass.
        apply_ghat_and_grad(
            stream,
            &data.kernels.kernel_ghat,
            rho,
            ek,
            energy_partial,
            &data.gpu_tables,
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

        unsafe {
            exec_inverse(
                data.planner_gpu,
                cuda_slice_to_ptr_mut(ek, stream),
                cuda_slice_to_ptr_mut(e, stream),
            );

            // todo: QC this
            // #[cfg(feature = "cufft")]
            // cudarc::cufft::exec_c2r(data.planner_gpu, &ex_ptr, &ex_ptr);
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

        gather_forces_to_atoms(
            stream,
            &data.kernels.kernel_gather,
            &atoms.pos_dev,
            &atoms.q_dev,
            e,
            &mut atoms.force_dev,
            self.plan_dims,
            self.box_dims,
        );

        // Queue both D2H copies before synchronizing once when the pinned force
        // buffer is read on the host.
        stream.memcpy_dtoh(energy_partial, energy_host).unwrap();
        stream
            .memcpy_dtoh(&atoms.force_dev, &mut atoms.force_host)
            .unwrap();
        stream.synchronize().unwrap();
        let f_host = atoms.force_host.as_slice();
        let reciprocal_energy: f64 = energy_host.as_slice().iter().sum();
        let energy = (reciprocal_energy + self_energy(q, self.alpha)) as f32;

        // todo: QC the - sign?
        let mut f = Vec::with_capacity(atoms.len);
        for i in 0..atoms.len {
            f.push(-Vec3 {
                x: f_host[i * 3 + 0],
                y: f_host[i * 3 + 1],
                z: f_host[i * 3 + 2],
            });
        }

        (f, energy)
    }
}

impl Drop for PmeRecip {
    fn drop(&mut self) {
        let Some(data) = &mut self.gpu_data else {
            return;
        };
        unsafe {
            if !data.planner_gpu.is_null() {
                destroy_plan(data.planner_gpu);
                data.planner_gpu = std::ptr::null_mut();
            }
        }
    }
}

pub(crate) fn _cuda_slice_to_ptr<T>(buf: &CudaSlice<T>, stream: &Arc<CudaStream>) -> *const c_void {
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

/// Launch the GPU kernel that spreads charges.
/// todo note: Getting the same values as on CPU here.
fn spread_charges(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    // posit, q and rho are passed as CudaSlices, as they're used elsewhere in the flow.
    pos_dev: &CudaSlice<f32>,
    q_dev: &CudaSlice<f32>,
    rho_dev: &mut CudaSlice<f32>, // real only.
    n_posits: u32,
    plan_dims: (usize, usize, usize),
    box_dims: (f32, f32, f32),
) {
    let (nx, ny, nz) = plan_dims;
    let nx_i = nx as i32;
    let ny_i = ny as i32;
    let nz_i = nz as i32;

    let (lx, ly, lz) = box_dims;
    let grid_x = nx as f32 / lx;
    let grid_y = ny as f32 / ly;
    let grid_z = nz as f32 / lz;

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
    launch_args.arg(&grid_x);
    launch_args.arg(&grid_y);
    launch_args.arg(&grid_z);

    unsafe { launch_args.launch(cfg) }.unwrap();
}

/// See notes on the CPU equivalent.
fn apply_ghat_and_grad(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    rho_dev: &CudaSlice<f32>, // Cplx
    ek_dev: &mut CudaSlice<f32>,
    energy_partial: &mut CudaSlice<f64>,
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
    let cfg = launch_cfg(n as u32, 256);
    let mut launch_args = stream.launch_builder(kernel);

    launch_args.arg(rho_dev);

    launch_args.arg(ek_dev);

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

    launch_args.arg(energy_partial);

    unsafe { launch_args.launch(cfg) }.unwrap();
}

/// See notes on the CPU equivalent.
fn gather_forces_to_atoms(
    stream: &Arc<CudaStream>,
    kernel: &CudaFunction,
    pos_gpu: &CudaSlice<f32>,
    q_gpu: &CudaSlice<f32>,
    e_gpu: &CudaSlice<f32>,
    out_partial_gpu: &mut CudaSlice<f32>,
    plan_dims: (usize, usize, usize),
    box_dims: (f32, f32, f32),
) {
    let (nx, ny, nz) = plan_dims;
    let nx_i = nx as i32;
    let ny_i = ny as i32;
    let nz_i = nz as i32;

    let (lx, ly, lz) = box_dims;
    let grid_x = nx as f32 / lx;
    let grid_y = ny as f32 / ly;
    let grid_z = nz as f32 / lz;

    let n = pos_gpu.len() / 3;
    let n_u32 = n as u32;
    let n_i32 = n as i32;

    // let cfg = LaunchConfig::for_num_elems(n_u32);
    let cfg = launch_cfg(n_u32, 256);
    let mut launch_args = stream.launch_builder(kernel);

    launch_args.arg(pos_gpu);

    launch_args.arg(e_gpu);
    launch_args.arg(q_gpu);

    launch_args.arg(out_partial_gpu);

    launch_args.arg(&n_i32);

    launch_args.arg(&nx_i);
    launch_args.arg(&ny_i);
    launch_args.arg(&nz_i);
    launch_args.arg(&grid_x);
    launch_args.arg(&grid_y);
    launch_args.arg(&grid_z);

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
