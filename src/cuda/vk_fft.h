#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

// Context lifecycle (adopt or create a CUDA stream)
//void* vk_make_context_default(void);
//void* vk_make_context_from_stream(void* cu_stream); // returns VkContext*
//void  vk_destroy_context(void* ctx);

// VkFFT plan lifecycle
//void* make_plan(void* ctx, int32_t nx, int32_t ny, int32_t nz, void* cu_stream);
void* make_plan(int32_t nx, int32_t ny, int32_t nz, void* cu_stream);

void destroy_plan(void* plan);

// FFT execs (raw device pointers)
void exec_forward(void* plan, void* real_in, void* complex_out);

void exec_inverse(void* plan, void* complex_in, void* real_out);

#ifdef __cplusplus
} // extern "C"
#endif