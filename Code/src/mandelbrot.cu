#include <complex.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

float rmax = 1.5f;
float rmin = -1.5f;
// make sure the aspect ratio is 16:9
float imin = -1.5f * 9.0f / 16.0f;
float imax = 1.5f * 9.0f / 16.0f;

int max_iterations = 200;

// CUDA Kernel
__global__ void mandelbrotKernel(int* output, int width, int height, float rmin, float rmax, float imin, float imax, int max_iterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    float x0 = (float)x / (float)width * (rmax - rmin) + rmin;
    float y0 = (float)y / (float)height * (imax - imin) + imin;
    float complex z0 = x0 + y0 * I;
    float complex z = z0;
    int iterations = 0;

    while(cabsf(z) < 2.0f && iterations < max_iterations) {
        z = z * z + z0;
        iterations++;
    }
    output[y * width + x] = iterations;
}

// Wrapper function to invoke the CUDA kernel
void computeMandelbrotGPU(int* h_output, int width, int height) {
    int* d_output;

    cudaMalloc(&d_output, width * height * sizeof(int));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    mandelbrotKernel<<<numBlocks, threadsPerBlock>>>(d_output, width, height, rmin, rmax, imin, imax, max_iterations);

    cudaMemcpy(h_output, d_output, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_output);
}

// Set new mandelbrot range
void set_mandelbrot_range(float new_rmin, float new_rmax, float new_imin, float new_imax) {
    rmin = new_rmin;
    rmax = new_rmax;
    imin = new_imin;
    imax = new_imax;
}

// map the number of iterations to a color
int color(int iterations) {
    if(iterations == max_iterations) {
        return 0x00000000;
    } 
    if (iterations > 50) {
        return 0x00FF00FF;
    }
    return 0x00FFFFFF;
}
