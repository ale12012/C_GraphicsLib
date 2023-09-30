#ifndef MANDELBROT_H
#define MANDELBROT_H

extern float rmax;
extern float rmin;
extern float imin;
extern float imax;

int mandelbrot(int pixilIndex, int width, int height);
int color(int iterations);
int fullSpectrumColor(int iterations);
int graidiantColor(int iterations);
int hueToRGB(float hue);
void set_mandelbrot_range(float new_rmin, float new_rmax, float newimin, float new_imax);

#endif
