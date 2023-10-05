#ifndef NEWTONSFRAC_H
#define NEWTONSFRAC_H

extern void set_range(double rmin, double rmax, double imin, double imax);
extern void set_aspect_ratio(int width, int height);
extern int n_iter_to_zero(int pixilIndex, int maxIterations, int width, int height);

extern double rmin;
extern double rmax;
extern double imin;
extern double imax;

#endif // NEWTONSFRAC_H
