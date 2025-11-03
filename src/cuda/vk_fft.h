#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif


// VkFFT plan lifecycle
void* make_plan(int32_t nx, int32_t ny, int32_t nz, void* cu_stream);

void destroy_plan(void* plan);

// FFT execs (raw device pointers)
void exec_forward(void* plan, void* real_in, void* complex_out);

void exec_inverse(void* plan, void* complex_in, void* real_out);

#ifdef __cplusplus
} // extern "C"
#endif