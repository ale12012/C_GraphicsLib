#ifndef MANDELBROT_H
#define MANDELBROT_H

extern float rmax;
extern float rmin;
extern float imin;
extern float imax;

double mandelbrot_orbit_trap(int pixilIndex, int width, int height);
int mandelbrot_avg_orbit(int pixilIndex, int width, int height);
int map_to_color(double min_distance);
void set_mandelbrot_range(float new_rmin, float new_rmax, float new_imin, float new_imax);

#endif
