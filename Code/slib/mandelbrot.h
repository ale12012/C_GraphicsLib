#ifndef MANDELBROT_H
#define MANDELBROT_H

extern int mandelbrot(int pixilIndex, int width, int height);
extern void set_mandelbrot_range(double rmin, double rmax, double imin, double imax);
extern void set_aspect_ratio(int width, int height);

extern double rmin;
extern double rmax;
extern double imin;
extern double imax;

#endif // MANDELBROT_H
