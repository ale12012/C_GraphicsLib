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
    double x0 = (double)x / (double)width * (rmax - rmin) + rmin;
    double y0 = (double)y / (double)height * (imax - imin) + imin;
    double complex z0 = x0 + y0 * I;
    double complex z = z0;
    int iterations = 0;
    while(cabsf(z) < 2.0f && iterations < max_iterations) {
        z = z*z + z0;
        iterations++;
    }
    return iterations;
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
    int ratio = 0xFFFFFF / max_iterations;
    int color = iterations * ratio;
    return color;
}
    