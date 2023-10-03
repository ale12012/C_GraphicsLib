#ifndef MANDELBROT_H
#define MANDELBROT_H

extern int mandelbrot_orbit_trap(int pixilIndex, int width, int height);
extern int mandelbrot_avg_orbit(int pixilIndex, int width, int height);
extern void set_mandelbrot_range(double rmin, double rmax, double imin, double imax);

extern double rmin;
extern double rmax;
extern double imin;
extern double imax;

#endif // MANDELBROT_H
