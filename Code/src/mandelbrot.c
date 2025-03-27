#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <float.h>

#define TWO_PI 6.2831853071795864769252867665590057683943
#define GOLDEN_RATIO 1.6180339887
#define max_iterations 100
#define JULIA_SIZE 100

double rmin = -2.0;
double rmax = 2.0;
double imin = -2.0;
double imax = 2.0;

int juliaSpace[JULIA_SIZE][JULIA_SIZE];
static int julia_initialized = 0;

int calculateJulia(double complex z, double complex c) {
    for (int iter = 0; iter < max_iterations; iter++) {
        z = z * z + c;
        if (cabsf(z) > 2.0) {
            return 1;
        }
    }
    return 0;
}

void fillJulia(int juliaSpace[JULIA_SIZE][JULIA_SIZE]) {
    double real_min = -2.0;
    double real_max = 2.0;
    double imag_min = -2.0;
    double imag_max = 2.0;

    for (int i = 0; i < JULIA_SIZE; i++) {
        for (int j = 0; j < JULIA_SIZE; j++) {
            double x = real_min + (double)i / (JULIA_SIZE - 1) * (real_max - real_min);
            double y = imag_min + (double)j / (JULIA_SIZE - 1) * (imag_max - imag_min);
            double complex z = x + y * I;
            juliaSpace[i][j] = calculateJulia(z, 0.35 + 0.35 * I);
        }
    }
}
void ensureJuliaInitialized() {
    if (!julia_initialized) {
        fillJulia(juliaSpace);
        julia_initialized = 1;
    }
}

void set_aspect_ratio(int width, int height){
    double center = (rmax + rmin) / 2.0;
    double real_range = (rmax - rmin);
    double imag_range = real_range * (double)height / (double)width;

    imin = -imag_range / 2.0;
    imax = imag_range / 2.0;

}

void set_mandelbrot_range(double new_rmin, double new_rmax, double new_imin, double new_imax) {
    rmin = new_rmin;
    rmax = new_rmax;
    imin = new_imin;
    imax = new_imax;
}

int hsl_to_rgb(double h, double s, double l) {
    // Ensure inputs are in expected ranges
    h = fmod(h, 360.0);
    if (h < 0) h += 360.0;  // Ensure positive hue
    s = fmin(fmax(s, 0.0), 1.0);
    l = fmin(fmax(l, 0.0), 1.0);

    // Convert hue to 0-6 range
    h /= 60.0;

    double c = (1.0 - fabs(2.0 * l - 1.0)) * s;
    double x = c * (1.0 - fabs(fmod(h, 2.0) - 1.0));
    double m = l - 0.5 * c;
    double r = 0, g = 0, b = 0;

    // Assign r, g, b based on hue sector
    if (0 <= h && h < 1) {
        r = c, g = x, b = 0;
    } else if (1 <= h && h < 2) {
        r = x, g = c, b = 0;
    } else if (2 <= h && h < 3) {
        r = 0, g = c, b = x;
    } else if (3 <= h && h < 4) {
        r = 0, g = x, b = c;
    } else if (4 <= h && h < 5) {
        r = x, g = 0, b = c;
    } else if (5 <= h && h < 6) {
        r = c, g = 0, b = x;
    }

    // Convert to 0-255 and pack into an integer
    int red = (int)((r + m) * 255);
    int green = (int)((g + m) * 255);
    int blue = (int)((b + m) * 255);

    return (red << 16) | (green << 8) | blue;
}

int iterations_to_color(int iter){
    return (int)((double)0xFFFFFF * iter/max_iterations);
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

double distance_to_julia(double complex z) {
    double real_min = -2.0;
    double real_max = 2.0;
    double imag_min = -2.0;
    double imag_max = 2.0;

    double pixel_width = (real_max - real_min) / JULIA_SIZE;
    double pixel_height = (imag_max - imag_min) / JULIA_SIZE;
    double min_dist = DBL_MAX;

    for (int y = 1; y < JULIA_SIZE - 1; y++) {
        for (int x = 1; x < JULIA_SIZE - 1; x++) {
            int current = juliaSpace[y][x];

            if (juliaSpace[y][x+1] != current ||
                juliaSpace[y][x-1] != current ||
                juliaSpace[y+1][x] != current ||
                juliaSpace[y-1][x] != current) {

                double real = real_min + x * pixel_width;
                double imag = imag_min + y * pixel_height;
                double complex p = real + imag * I;

                double dist = cabs(z - p);
                if (dist < min_dist) min_dist = dist;
            }
        }
    }

    return min_dist;
}

int mandelbrot(int pixelIndex, int width, int height) {
    ensureJuliaInitialized();  // safe to call every time

    int x = pixelIndex % width;
    int y = pixelIndex / width;
    double x0 = (double)x / (double)width * (rmax - rmin) + rmin;
    double y0 = (double)y / (double)height * (imax - imin) + imin;
    double complex z0 = x0 + y0 * I;
    double complex z = z0;

    double min_dist = DBL_MAX;

    for (int i = 0; i < max_iterations && cabsf(z) < 4.0f; i++) {
        double dist = distance_to_julia(z);
        if (dist < min_dist) min_dist = dist;

        z = z * z + z0;
    }

    double brightness = exp(-min_dist * 10.0);
    int gray = (int)(brightness * 255.0);
    return (gray << 16) | (gray << 8) | gray;
}
