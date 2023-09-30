#include <complex.h>


float rmax = 1.5f;
float rmin = -1.5f;
// make sure the aspect ratio is 16:9
float imin = -1.5f * 9.0f / 16.0f;
float imax = 1.5f * 9.0f / 16.0f;

int max_iterations = 200;

int mandelbrot(int pixilIndex, int width, int height) {
    int x = pixilIndex % width;
    int y = pixilIndex / width;
    float x0 = (float)x / (float)width * (rmax - rmin) + rmin;
    float y0 = (float)y / (float)height * (imax - imin) + imin;
    float complex z0 = x0 + y0 * I;
    float complex z = z0;
    int iterations = 0;
    while(cabsf(z) < 2.0f && iterations < max_iterations) {
        z = z*z + z0;
        iterations++;
    }
    return iterations;
}

// map the number of iterations to a color
int color(int iterations) {
    if(iterations == max_iterations) {
        return 0x00000000;
    } if (iterations > 50) {
        return 0x00FF00FF;
    }
    else {
        return 0x00FFFFFF;
    }
}