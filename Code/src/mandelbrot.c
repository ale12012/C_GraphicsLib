#include <complex.h>
#include <math.h>
#include <stdio.h>

#define TWO_PI 6.2831853071795864769252867665590057683943
#define GOLDEN_RATIO 1.6180339887

float rmax = 1.5f;
float rmin = -1.5f;
// make sure the aspect ratio is 16:9 
float imin = -1.5f * 9.0f / 16.0f;
float imax = 1.5f * 9.0f / 16.0f;


int max_iterations = 150;

void set_mandelbrot_range(float new_rmin, float new_rmax, float new_imin, float new_imax) {
    rmin = new_rmin;
    rmax = new_rmax;
    imin = new_imin;
    imax = new_imax;
}


int map_to_color(double min_distance) {
    // Scale min_distance for better color representation. 
    // You might need to adjust this factor based on the results you see.
    double scaled_distance = min_distance * 10.0;
    scaled_distance = fmin(fmax(scaled_distance, 0.0), 1.0);
    int grayscale_value = (int)(0xFF * (1.0 - scaled_distance));
    return (grayscale_value % 16) | (grayscale_value << 8) | grayscale_value;
}

int map_to_color_avg_orbit(double sum, int iterations) {
    int color = (int)((double)0xFF * sum) | (int)((double)0xFF * sum) << 8 | (int)((double)0xFF * sum) << 16;

    if (iterations < max_iterations) {
        return color;
    } else {
        return 0;
    }
}


double distance_to_line_segment(double complex z) {
    double x = creal(z);
    double y = cimag(z);

    if (y >= -0.5 && y <= 0.5) {  // Point lies within the vertical segment
        return fabs(x);
    } else if (y < -0.5) {  // Point is below the segment
        return hypot(x, y + 0.5);
    } else {  // Point is above the segment
        return hypot(x, y - 0.5);
    }
}

int mandelbrot_avg_orbit(int pixilIndex, int width, int height) {
    int x = pixilIndex % width;
    int y = pixilIndex / width;
    double x0 = (double)x / (double)width * (rmax - rmin) + rmin;
    double y0 = (double)y / (double)height * (imax - imin) + imin;
    double complex z0 = x0 + y0 * I;
    double complex z = z0;
    double complex z_prev = z0;

    double sum = 0.0;

    int iterations = 0;
    while (cabsf(z) < 2.0f && iterations < max_iterations) {
        z_prev = z;
        z = z * z + z0;
        
        double diff = cabsf(z) - cabsf(z_prev);
        // Apply the transformation
        double transformed = 1.0 / (1.0 + diff);
        sum += transformed;

        iterations++;
    }

    if (iterations < max_iterations) {
        sum /= iterations;
    }

    return map_to_color_avg_orbit(sum, iterations);
}


int mandelbrot_orbit_trap(int pixilIndex, int width, int height) {
    int x = pixilIndex % width;
    int y = pixilIndex / width;
    double x0 = (double)x / (double)width * (rmax - rmin) + rmin;
    double y0 = (double)y / (double)height * (imax - imin) + imin;
    double complex z0 = x0 + y0 * I;
    double complex z = z0;
    int iterations = 0;

    double min_distance = 1.0e100;  // A very large number to start

    while (cabsf(z) < 2.0f && iterations < max_iterations) {
        double dist = distance_to_line_segment(z);
        if (dist < min_distance) {
            min_distance = dist;
        }
        z = z * z + z0;
        iterations++;
    }
    return map_to_color(min_distance);
}


