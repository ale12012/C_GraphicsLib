#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>


double rmin = -2.0;
double rmax = 2.0;
double imin = -2.0;
double imax = 2.0;

void set_aspect_ratio(int width, int height){
    imin = (rmax - rmin) * (double)height / (double)width / 2.0;
    imax = -imin;
}


void set_range(double new_rmin, double new_rmax, double new_imin, double new_imax) {
    rmin = new_rmin;
    rmax = new_rmax;
    imin = new_imin;
    imax = new_imax;
}

int map_to_color(int iterations, int maxIterations){
    int color_ratio = 0x00F0FF / maxIterations;
    return iterations * color_ratio;
}

// for function z**3 - 1
void newtons_method(double complex* z) {
    *z -= (cpow(*z, 3.0) - 1.0) / (3.0 * cpow(*z, 2.0));
}
int n_iter_to_zero(int pixilIndex, int maxIterations, int width, int height) {
    int x = pixilIndex % width;
    int y = pixilIndex / width;
    double x0 = (double)x / (double)width * (rmax - rmin) + rmin;
    double y0 = (double)y / (double)height * (imax - imin) + imin;
    double complex z = x0 + y0 * I;
    int i;
    for(i = 0; i < maxIterations; i++) {
        newtons_method(&z);
        if(cabs(cpow(z, 3.0) - 1.0) < 0.0001){
            return map_to_color(i, maxIterations);
        } 
    }
    return map_to_color(i, maxIterations);
}


