#ifndef MANDELBROT_H
#define MANDELBROT_H

extern float rmax;
extern float rmin;
extern float imin;
extern float imax;

int mandelbrot(int pixilIndex, int width, int height);
int color(int iterations);
void set_mandelbrot_range(float new_rmin, float new_rmax, float newimin, float new_imax);
void computeMandelbrotGPU(int* h_output, int width, int height);

#endif
