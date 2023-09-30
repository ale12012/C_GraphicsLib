#include <cuComplex.h>
#include <stdio.h>
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include\cuda_runtime.h"
#include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include\device_launch_parameters.h"
#include "../slib/mandelbrot.cuh"
#include <math.h>

#define TWO_PI 6.2831853071795864769252867665590057683943

float rmax = 1.5f;
float rmin = -1.5f;
            // make sure the aspect ratio is 16:9
float imin = -1.5f * 9.0f / 16.0f;
float imax = 1.5f * 9.0f / 16.0f;

int max_iterations = 100;

// CUDA Kernel
__global__ void mandelbrotKernel(int* output, int width, int height, float rmin, float rmax, float imin, float imax, int max_iterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= width || y >= height) return;

    double x0 = (double)x / (double)width * (rmax - rmin) + rmin;
    double y0 = (double)y / (double)height * (imax - imin) + imin;
    cuDoubleComplex z0 = make_cuDoubleComplex(x0, y0);
    cuDoubleComplex z = z0;
    int iterations = 0;

    while(cuCabs(z) < 2.0f && iterations < max_iterations) {
        z = cuCadd(cuCmul(z, z), z0);
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

int initializeCUDA() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0) {
        return -1;
    }
    return 0;
}

void computeMandelbrot(int* pixels, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        pixels[i] = i;
    }
    computeMandelbrotGPU(pixels, width, height); 
}

// Set new mandelbrot range
void set_mandelbrot_range(float new_rmin, float new_rmax, float new_imin, float new_imax) {
    rmin = new_rmin;
    rmax = new_rmax;
    imin = new_imin;
    imax = new_imax;
}

// map the number of iterations to a color
int hueToRGB(float hue) {
    float r = sin(hue) * 127.5 + 127.5;
    float g = sin(hue + TWO_PI / 3) * 127.5 + 127.5;
    float b = sin(hue + 2 * TWO_PI / 3) * 127.5 + 127.5;
    return ((int)r << 16) | ((int)g << 8) | (int)b;
}

int fullSpectrumColor(int iterations) {
    float hue = (float)iterations / max_iterations * TWO_PI;
    if (iterations == max_iterations) {
        return 0x000000;
    }
    return hueToRGB(hue);
}



// map the number of iterations to a color
int graidiantColor(int iterations) {
    int rStart = 0xFF, gStart = 0xFF, bStart = 0xFF;  // White
    int rEnd = 0, gEnd = 0, bEnd = 0;                 // Black

    double ratio = (double)iterations / max_iterations;
    int r = (int)(rStart + ratio * (rEnd - rStart));
    int g = (int)(gStart + ratio * (gEnd - gStart));
    int b = (int)(bStart + ratio * (bEnd - bStart));
    if (iterations == max_iterations) {
        r = 0;
        g = 0;
        b = 0;
    }
    return (r << 16) | (g << 8) | b;
}
int defColor(int iterations) {
    int r = (iterations % 8) * 32;  
    int g = (iterations % 16) * 16;
    int b = (iterations % 32) * 8;
    return (r << 16) | (g << 8) | b;
}

int color(int choice, int iterations) {
    if (choice == 1) return graidiantColor(iterations);
    else if (choice == 2) return fullSpectrumColor(iterations);
    else return defColor(iterations);
}
