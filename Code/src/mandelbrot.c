#include <complex.h>
#include <math.h>

#define TWO_PI 6.2831853071795864769252867665590057683943

float rmax = 1.5f;
float rmin = -1.5f;
// make sure the aspect ratio is 16:9 
float imin = -1.5f * 9.0f / 16.0f;
float imax = 1.5f * 9.0f / 16.0f;


int max_iterations = 100;

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

int hueToRGB(float hue) {
    float r = sin(hue) * 127.5 + 127.5;
    float g = sin(hue + TWO_PI / 3) * 127.5 + 127.5;
    float b = sin(hue + 2 * TWO_PI / 3) * 127.5 + 127.5;
    return ((int)r << 16) | ((int)g << 8) | (int)b;
}

int fullSpectrumColor(int iterations) {
    float hue = (float)iterations / max_iterations * TWO_PI;
    if (iterations == max_iterations) {
        hue = 1.0f;
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
int color(int iterations) {
    int r = (iterations % 8) * 32;  
    int g = (iterations % 16) * 16;
    int b = (iterations % 32) * 8;
    return (r << 16) | (g << 8) | b;
}