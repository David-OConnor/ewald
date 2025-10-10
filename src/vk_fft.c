#define VK_USE_PLATFORM_WIN32_KHR // or XCB/Xlib/Wayland as needed // todo: What? Why do I want this?
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include "vkFFT.h"
#include "spme_vkfft.h"

// ---- minimal helpers and opaque structs ----
typedef struct {
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice device;
    uint32_t queueFamily;
    VkQueue queue;
    VkCommandPool cmdPool;
    // pipeline cache, descriptor pool, etc., as needed by your compute shaders
} VkContext;

typedef struct {
    VkFFTApplication app_r2c;
    VkFFTApplication app_c2r;
    VkFFTConfiguration cfg_r2c;
    VkFFTConfiguration cfg_c2r;
    VkDevice device;
    VkQueue queue;
    uint64_t Nx, Ny, Nz;
} VkFftPlan;

typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceSize size;
} VkBuf;

// forward decls for tiny Vulkan utils (create instance/device/queue, buffer alloc, cmd submit)
static VkContext* make_default_ctx(void);
static void destroy_ctx(VkContext* ctx);
static VkBuf* alloc_host_visible(VkContext* c, VkDeviceSize nbytes, int zero);
static void  free_buf(VkContext* c, VkBuf* b);
static void  upload(VkContext* c, VkBuf* b, const void* src, VkDeviceSize n);
static void  download(VkContext* c, VkBuf* b, void* dst, VkDeviceSize n);

// ---- exported FFI ----
void* vk_make_context_default(void) { return make_default_ctx(); }
void  vk_destroy_context(void* ctx) { destroy_ctx((VkContext*)ctx); }

void* vk_alloc_and_upload(void* ctx, const void* host_src, uint64_t nbytes) {
    VkContext* c = (VkContext*)ctx;
    VkBuf* b = alloc_host_visible(c, (VkDeviceSize)nbytes, /*zero=*/0);
    upload(c, b, host_src, (VkDeviceSize)nbytes);
    return b;
}
void* vk_alloc_zeroed(void* ctx, uint64_t nbytes) {
    VkContext* c = (VkContext*)ctx;
    VkBuf* b = alloc_host_visible(c, (VkDeviceSize)nbytes, /*zero=*/1);
    return b;
}
void  vk_download(void* ctx, void* dev_buf, void* host_dst, uint64_t nbytes) {
    VkContext* c = (VkContext*)ctx;
    download(c, (VkBuf*)dev_buf, host_dst, (VkDeviceSize)nbytes);
}
void  vk_free(void* ctx, void* dev_buf) {
    VkContext* c = (VkContext*)ctx;
    free_buf(c, (VkBuf*)dev_buf);
}

void* vkfft_make_plan_r2c_c2r_many(void* ctx, int32_t nx, int32_t ny, int32_t nz) {
    VkContext* c = (VkContext*)ctx;
    VkFftPlan* p = (VkFftPlan*)calloc(1, sizeof(VkFftPlan));
    p->device = c->device;
    p->queue = c->queue;
    p->Nx = (uint64_t)nx; p->Ny = (uint64_t)ny; p->Nz = (uint64_t)nz;

    memset(&p->cfg_r2c, 0, sizeof(p->cfg_r2c));
    p->cfg_r2c.FFTdim = 3;
    p->cfg_r2c.size[0] = p->Nx; p->cfg_r2c.size[1] = p->Ny; p->cfg_r2c.size[2] = p->Nz;
    p->cfg_r2c.performR2C = 1;
    p->cfg_r2c.device = c->device;
    p->cfg_r2c.queue = c->queue;
    p->cfg_r2c.physicalDevice = c->phys;
    p->cfg_r2c.commandPool = c->cmdPool;

    memset(&p->cfg_c2r, 0, sizeof(p->cfg_c2r));
    p->cfg_c2r.FFTdim = 3;
    p->cfg_c2r.size[0] = p->Nx; p->cfg_c2r.size[1] = p->Ny; p->cfg_c2r.size[2] = p->Nz;
    p->cfg_c2r.performR2C = 0; // C2R
    p->cfg_c2r.device = c->device;
    p->cfg_c2r.queue = c->queue;
    p->cfg_c2r.physicalDevice = c->phys;
    p->cfg_c2r.commandPool = c->cmdPool;

    if (initializeVkFFT(&p->app_r2c, &p->cfg_r2c) != VKFFT_SUCCESS) { free(p); return NULL; }
    if (initializeVkFFT(&p->app_c2r, &p->cfg_c2r) != VKFFT_SUCCESS) { deleteVkFFT(&p->app_r2c); free(p); return NULL; }
    return p;
}

void  vkfft_destroy_plan(void* plan_) {
    VkFftPlan* p = (VkFftPlan*)plan_;
    if (!p) return;
    deleteVkFFT(&p->app_r2c);
    deleteVkFFT(&p->app_c2r);
    free(p);
}

void  vkfft_exec_forward_r2c(void* plan_, void* real_in_dev, void* complex_out_dev) {
    VkFftPlan* p = (VkFftPlan*)plan_;
    VkBuf* rin = (VkBuf*)real_in_dev;
    VkBuf* kout = (VkBuf*)complex_out_dev;

    VkFFTLaunchParams lp = {0};
    lp.buffer = &rin->buf;
    lp.bufferSize = &rin->size;
    lp.outputBuffer = &kout->buf;
    lp.outputBufferSize = &kout->size;
    performVulkanFFT(&p->app_r2c, &lp, -1, 1); // forward
}

void  vkfft_exec_inverse_c2r(void* plan_, void* complex_in_dev, void* real_out_dev) {
    VkFftPlan* p = (VkFftPlan*)plan_;
    VkBuf* kin = (VkBuf*)complex_in_dev;
    VkBuf* rout = (VkBuf*)real_out_dev;

    VkFFTLaunchParams lp = {0};
    lp.buffer = &kin->buf;
    lp.bufferSize = &kin->size;
    lp.outputBuffer = &rout->buf;
    lp.outputBufferSize = &rout->size;
    performVulkanFFT(&p->app_c2r, &lp, 1, 1); // inverse
}

// ----- SPME compute passes -----
// Implement these with your SPIR-V compute shaders. Each reads/writes buffers by descriptor bindings.
// For brevity, we elide full pipeline creation; in practice, you’ll create pipelines once per ctx.

void  vk_spme_apply_ghat_and_grad(
    void* ctx_, void* rho_k_, void* exk_, void* eyk_, void* ezk_,
    void* kx_, void* ky_, void* kz_,
    void* bx_, void* by_, void* bz_,
    int32_t nx, int32_t ny, int32_t nz, float vol, float alpha)
{
    // Bind rho_k (complex), kx/ky/kz, bx/by/bz, and write exk/eyk/ezk.
    // Dispatch with workgroup covering (nx * ny * (nz/2+1)) complex elements.
    // The kernel computes:
    //   ghat = (2π*2 / vol) * exp(-k^2/(4α^2)) / (k^2 * (bx*by*bz))   // 2π*2 = TAU
    //   phi_k = rho_k * ghat
    //   exk = i*kx*phi_k; eyk = i*ky*phi_k; ezk = i*kz*phi_k
    // (same algebra you used in Rust, but on GPU half-spectrum layout).
    (void)ctx_; (void)rho_k_; (void)exk_; (void)eyk_; (void)ezk_;
    (void)kx_; (void)ky_; (void)kz_; (void)bx_; (void)by_; (void)bz_;
    (void)nx; (void)ny; (void)nz; (void)vol; (void)alpha;
    // TODO: implement compute dispatch
}

void  vk_spme_scale_real_fields(void* ex_, void* ey_, void* ez_, int32_t nx, int32_t ny, int32_t nz) {
    // scale by 1/(nx*ny*nz) in-place; either a tiny compute pass or host-map+scale.
    // For now, do host-map for simplicity (since buffers are HOST_VISIBLE).
    VkBuf* ex = (VkBuf*)ex_; VkBuf* ey = (VkBuf*)ey_; VkBuf* ez = (VkBuf*)ez_;
    const float s = 1.0f / (float)(nx * ny * nz);
    void* p;
    vkMapMemory(/*device*/ ex->mem /*...*/); // elided: you need device handle; store it in VkBuf or pass ctx.
    // For brevity, leave this as TODO; production: do it with a tiny compute shader.
    (void)s; (void)p; (void)ey; (void)ez;
}

double vk_spme_energy_half_spectrum_sum(
    void* ctx_,
    void* rho_k_, void* kx_, void* ky_, void* kz_,
    void* bx_, void* by_, void* bz_,
    int32_t nx, int32_t ny, int32_t nz, float vol, float alpha)
{
    // Compute reduction of 0.5 * Re{ rho*(k) * phi(k) } over half-spectrum.
    // Implement as a compute pass writing partial sums to a buffer, then map and sum on host.
    (void)ctx_; (void)rho_k_; (void)kx_; (void)ky_; (void)kz_;
    (void)bx_; (void)by_; (void)bz_; (void)nx; (void)ny; (void)nz; (void)vol; (void)alpha;
    // TODO: implement compute dispatch + readback. Return 0 for now.
    return 0.0;
}

// --------------- minimal Vulkan utils (sketch) ----------------
// NOTE: These are intentionally terse; fill in proper error handling and instance/device selection.

static VkContext* make_default_ctx(void) {
    VkContext* c = calloc(1, sizeof(VkContext));
    // create instance -> pick physical device with compute -> create device+queue -> command pool
    // (omitted for brevity)
    return c;
}
static void destroy_ctx(VkContext* c) {
    if (!c) return;
    // destroy command pool, device, instance
    free(c);
}
static VkBuf* alloc_host_visible(VkContext* c, VkDeviceSize nbytes, int zero) {
    VkBuf* b = calloc(1, sizeof(VkBuf));
    // vkCreateBuffer -> vkGetBufferMemoryRequirements -> find HOST_VISIBLE|HOST_COHERENT memory ->
    // vkAllocateMemory -> vkBindBufferMemory. If zero!=0, map+memset 0.
    (void)c; (void)nbytes; (void)zero;
    return b;
}
static void free_buf(VkContext* c, VkBuf* b) {
    if (!b) return;
    // vkDestroyBuffer, vkFreeMemory
    (void)c; free(b);
}
static void upload(VkContext* c, VkBuf* b, const void* src, VkDeviceSize n) {
    // map, memcpy, unmap (HOST_VISIBLE & COHERENT)
    (void)c; (void)b; (void)src; (void)n;
}
static void download(VkContext* c, VkBuf* b, void* dst, VkDeviceSize n) {
    // map, memcpy, unmap
    (void)c; (void)b; (void)dst; (void)n;
}
