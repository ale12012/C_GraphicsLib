#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#define TWO_PI 6.2831853071795864769252867665590057683943
#define GOLDEN_RATIO 1.6180339887

double rmin = -2.0;
double rmax = 2.0;
double imin = -2.0;
double imax = 2.0;

void set_aspect_ratio(int width, int height){
    imin = (rmax - rmin) * (double)height / (double)width / 2.0;
    imax = -imin;
}

int max_iterations = 500;

void set_mandelbrot_range(double new_rmin, double new_rmax, double new_imin, double new_imax) {
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



int hsl_to_rgb(double h, double s, double l) {
    // Ensure inputs are in expected ranges
    h = fmod(h, 360.0);
    s = fmin(fmax(s, 0.0), 1.0);
    l = fmin(fmax(l, 0.0), 1.0);

    // Convert hue to 0-6 range
    h /= 60.0;

    double c = (1.0 - fabs(2.0 * l - 1.0)) * s;
    double x = c * (1.0 - fabs(fmod(h, 2.0) - 1.0));
    double m = l - 0.5 * c;
    double r, g, b;

    // ... [rest of the function remains the same]

    return ((int)((r + m) * 255) << 16) + ((int)((g + m) * 255) << 8) + (int)((b + m) * 255);
}

int map_to_color_avg_orbit(double normalized_sum, int iterations) {
    if (iterations == max_iterations) {
        return 0x000000; // Black for points inside the Mandelbrot set
    } else {
        // Produce a cyclic color palette by using the fractional part of the normalized sum
        double hue = 57 * abs(acos(sin(normalized_sum))) + 90;  // Needs to be in the range 0 to 360
        double saturation = 1 / exp(normalized_sum*10) ; // Can be adjusted between 0 and 1
        double lightness = tan(normalized_sum);  // Can be adjusted between 0 and 1

        return hsl_to_rgb(hue, saturation, lightness);
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

int mandelbrot_avg_orbit(int pixelIndex, int width, int height) {
    int x = pixelIndex % width;
    int y = pixelIndex / width;
    double x0 = (double)x / (double)width * (rmax - rmin) + rmin;
    double y0 = (double)y / (double)height * (imax - imin) + imin;
    double complex z0 = x0 + y0 * I;
    double complex z = z0;
    double z_prev = z0;
    double sum = 0.0;

    int iterations = 0;
    while (/*cabs(z) < 4.0 &&*/ cabs(z) <= 10e14 && iterations < max_iterations) {  // Adjusted bailout radius for more accurate smoothing
        z = z * z + z0;
        sum += 1.0 / (1.0 + cabs(z - z_prev));
        z_prev = z;
        iterations++;
    }
    sum /= iterations;
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

    double min_distance = 1.0e20;  // A very large number to start

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


