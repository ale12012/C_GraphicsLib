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

int max_iterations = 1000;

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


int map_to_color(double min_distance, int iter) {
    // Scale min_distance for better color representation. 
    // You might need to adjust this factor based on the results you see.
    double scaled_distance = min_distance * 100.0f;
    scaled_distance = fmin(fmax(scaled_distance, 0.0), 1.0);
    //int grayscale_value = (int)(0xFF * (1.0 - scaled_distance));
    //return (grayscale_value % 16) | (grayscale_value << 8) | grayscale_value;
    double h, s, l;
    h = (scaled_distance * 360.0f + fmod(pow((double)iter / (double)max_iterations * 360.0, 1.50), 360.0)) / 2.0f;
    s = (1.0f - scaled_distance + (double)iter / (double)max_iterations) / 2.0f;
    l = (1.0f - scaled_distance + (double)iter / (double)max_iterations) / 2.0f;
    return hsl_to_rgb(h, s, l);
}


int iterations_to_color(int iter) {
    double h = fmod(pow((double)iter / (double)max_iterations * 360.0, 1.50), 360.0);
    double s =  (double)iter / (double)max_iterations; 
    double l = (double)iter / (double)max_iterations;
    return hsl_to_rgb(h, s, l);
}
    

int map_to_color_avg_orbit(double normalized_sum, int iterations) {
    if (iterations == max_iterations) {
        return 0x000000; // Black for points inside the Mandelbrot set
    } else {
        //double hue = fmod(pow(iterations / max_iterations * 360.0f, 1.5), 360);
        double hue = fmod(normalized_sum, 360.0f);  // Needs to be in the range 0 to 360
        double saturation = (double)iterations/(double)max_iterations ; // Can be adjusted between 0 and 1
        double lightness = fmod(normalized_sum, 0.9f);  // Can be adjusted between 0 and 1

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
    //probably needs interpolation
    int x = pixelIndex % width;
    int y = pixelIndex / width;
    double x0 = (double)x / (double)width * (rmax - rmin) + rmin;
    double y0 = (double)y / (double)height * (imax - imin) + imin;
    double complex z0 = x0 + y0 * I;
    double complex z = z0;
    double complex z_prev = z0;
    double complex dz = 1 + 0 * I;
    double complex dz_sum = 0 + 0 * I;
    double dbail = 1e19;
    double sum = 0.0;
    

    int iterations = 0;
    while (cabsf(dz_sum) < dbail && iterations < max_iterations) {
        z = z*z + z0;
        dz = 2*dz*z + 1;
        dz_sum += dz;
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
    double complex dz = 1 + 0 * I;
    double complex dz_sum = 0 + 0 * I;
    double dbail = 1e19;
    
    int iterations = 0;

    double min_distance = 1.0e20;  // A very large number to start

    while (cabsf(dz_sum) < dbail && iterations < max_iterations) {
        double dist = distance_to_line_segment(z);
        if (dist < min_distance) {
            min_distance = dist;
        }
        z = z * z + z0;
        dz = 2*dz*z + 1;
        dz_sum += dz;
        iterations++;
    }
    return map_to_color(min_distance, iterations);
}

int dbail_mandelbrot(int pixilIndex, int width, int height)
{
    int x = pixilIndex % width;
    int y = pixilIndex / width;
    double x0 = (double)x / (double)width * (rmax - rmin) + rmin;
    double y0 = (double)y / (double)height * (imax - imin) + imin;
    double complex z0 = x0 + y0 * I;
    double complex z = z0;
    int iterations = 0;
    double complex dz = 1 + 0 * I;
    double complex dz_sum = 0 + 0 * I;
    double dbail = 1e6;

    while(iterations <= max_iterations && cabsf(dz_sum) < dbail)
    {
        z = z*z + z0;
        dz = 2*dz*z + 1;
        dz_sum += dz;
        iterations++;
    }
    if (cabs(dz_sum) >= dbail){
        return iterations_to_color(iterations);
    }
    return 0;
}


