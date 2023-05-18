#include <stdio.h>
#include <stdlib.h>

void bin_ssa(double* t, int n, int* x1, int* x2, int* x3, double dt, double T, int* out_t_bins, int* out_x1_binned, int* out_x2_binned, int* out_x3_binned) {
    // allocate memory for first difference of species counts
    int* dx1 = malloc(n * sizeof(int));
    int* dx2 = malloc(n * sizeof(int));
    int* dx3 = malloc(n * sizeof(int));

    // compute the first difference of the species counts
    dx1[0] = x1[0];
    dx2[0] = x2[0];
    dx3[0] = x3[0];
    for (int i = 1; i < n; i++) {
        dx1[i] = x1[i] - x1[i-1];
        dx2[i] = x2[i] - x2[i-1];
        dx3[i] = x3[i] - x3[i-1];
    }

    // define the time domain for the binned data
    int num_bins = (int)(T / dt);
    double* t_bins = malloc(num_bins * sizeof(double));
    int* x1_binned = calloc(num_bins, sizeof(int));
    int* x2_binned = calloc(num_bins, sizeof(int));
    int* x3_binned = calloc(num_bins, sizeof(int));
    for (int i = 0; i < num_bins; i++) {
        t_bins[i] = i * dt;
    }

    // assign reaction times to time bins
    for (int i = 0; i < n; i++) {
        int bin_index = (int)(t[i] / dt);
        x1_binned[bin_index] += dx1[i];
        x2_binned[bin_index] += dx2[i];
        x3_binned[bin_index] += dx3[i];
    }

    // accumulate species counts over time
    for (int i = 1; i < num_bins; i++) {
        x1_binned[i] += x1_binned[i-1];
        x2_binned[i] += x2_binned[i-1];
        x3_binned[i] += x3_binned[i-1];
    }

    // return binned data
    for (int i = 0; i < num_bins; i++) {
        out_t_bins[i] = t_bins[i];
        out_x1_binned[i] = x1_binned[i];
        out_x2_binned[i] = x2_binned[i];
        out_x3_binned[i] = x3_binned[i];
    }

    // free memory
    free(dx1);
    free(dx2);
    free(dx3);
    free(t_bins);
    free(x1_binned);
    free(x2_binned);
    free(x3_binned);
}

int main() {
    // example usage
    int n = 1000000;
    double* t = malloc(n * sizeof(double));
    int* x1 = malloc(n * sizeof(int));
    int* x2 = malloc(n * sizeof(int));
    int* x3 = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) {
        t[i] = (double)i / 1000.0;
        x1[i] = i % 10;
        x2[i] = i % 20
