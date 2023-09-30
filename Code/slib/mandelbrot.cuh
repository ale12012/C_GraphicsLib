#ifdef __cplusplus
extern "C" {
#endif
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include\cuda_runtime.h"
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include\device_launch_parameters.h"
__global__ void mandelbrotKernel(int* output, int width, int height, float rmin, float rmax, float imin, float imax, int max_iterations);  
int color(int iterations);
void set_mandelbrot_range(float new_rmin, float new_rmax, float newimin, float new_imax);
void computeMandelbrotGPU(int* h_output, int width, int height);
int initializeCUDA();
void computeMandelbrot(int* pixels, int width, int height);

#ifdef __cplusplus
}
#endif
