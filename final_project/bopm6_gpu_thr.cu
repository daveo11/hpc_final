/**
 * Final Project - Binomial Lattice model(for option pricing)
 * 
 * For this assignment you will write an MPI matrix multiplication.
 * 
 * To compile:
 *     gcc -Wall -O3 -march=native *.c -o bopm_serial -lm
 * 
 * To run on your local machine:
 *    ./bopm_serial
 * 
 * this will run with default values
 *
 * int n = 1000; // Number of steps
 * double S = 100; // Initial stock price
 * double K = 100; // Strike price
 * double r = 0.05; // Risk-free rate
 * double v = 0.2; // Volatility
 * double T = 1; // Time to maturity
 * int PC = 1; // 0 for call, 1 for put
 * int AM=1;
 * 
 * to change values run with these params:
 * 
 * usage: ./bopm6 [-n num-steps] [-s initial-stock-price] [-k strike-price] [-r risk-free-rate] [-v volatility]  [-t time-to-maturity] [-p put-or-call] -a [american 1 european 0]input output
 *  
 * american put ./bopm_serial -n 1000 -s 85 -k 90 -r 0.03 -v 0.2 -t 1 -p 1 -a 1
 * american call ./bopm_serial -n 1000 -s 85 -k 90 -r 0.03 -v 0.2 -t 1 -p 0 -a 1
 * european put ./bopm_serial -n 1000 -s 85 -k 90 -r 0.03 -v 0.2 -t 1 -p 1 -a 0
 * european call ./bopm_serial -n 1000 -s 85 -k 90 -r 0.03 -v 0.2 -t 1 -p 0 -a 0
 * 
 * https://github.com/padraic00/Binomial-Options-Pricing-Model/tree/master (python)
 * https://en.wikipedia.org/wiki/Binomial_options_pricing_model
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <inttypes.h>

#include "matrix.h"
#include "util.h"
#include <omp.h>


// make this work
__global__
void backward_induction_device(Matrix* O, size_t n, double S, double K, double r, double v, double T, int PC, int AM) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    // Backward induction for option price
    if (AM) {
        double immediateExercise;
        if (PC == 1) { // Put
            immediateExercise = fmax(K - tmp[j][i], 0);
        } else { // Call
            immediateExercise = fmax(tmp[j][i] - K, 0);
        }
        MATRIX_AT(O,i,j) = fmax(exp(-r * dt) * (p * MATRIX_AT(O,i,j + 1) + (1 - p) * MATRIX_AT(O,i + 1,j + 1)), immediateExercise);
    } else {
        MATRIX_AT(O,i,j) = exp(-r * dt) * (p * MATRIX_AT(O,i,j + 1) + (1 - p) * MATRIX_AT(O,i + 1,j + 1));
    }
}


bool OptionsVal(Matrix* O, size_t n,double S, double K, double r, double v, double T, int PC,int AM) {
   //const size_t n = C->rows;

    double dt = T / n;
    double u = exp(v * sqrt(dt));
    double d = 1 / u;
    double p = (exp(r * dt) - d) / (u - d);

    matrix_fill_zeros(O);
    #omp parallel shared(O, n, S, K, r, v, T, PC, AM, u, d, p, dt) private(tmp)
    {
        // Memory allocation for Pm and Cm
        #pragma omp for
        double **tmp = calloc(n , sizeof(double *));
        for (size_t i = 0; i <= n; i++) {
            tmp[i] = calloc(n , sizeof(double));
        }

        // Calculate stock price at each node
        #pragma omp for
        for (size_t j = 0; j <= n; j++) {
            tmp[j][n] = S * pow(u, n - j) * pow(d, j);
        }

        // Initialize values at maturity
        #pragma omp for
        for (size_t j = 0; j <= n; j++) {
            if (PC == 1) { // Put
                MATRIX_AT(O,j,n) = fmax(K - tmp[j][n], 0);

            } else { // Call
                MATRIX_AT(O,j,n) = fmax(tmp[j][n] - K, 0);
            }
        }
    }
    //send to device
    cudaMemcpy(O->data_device, O->data, O->rows * O->cols * sizeof(double), cudaMemcpyHostToDevice);

    // Backward induction for option price
    int grid_size = (n+1)/32 + 1; // Needs fixing
    backward_induction_device<<<grid_size, 32>>>(O, n, S, K, r, v, T, PC, AM);
    cudasynchronize();

    // send results back to host
    cudaMemcpy(O->data, O->data_device, O->rows * O->cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory
    free(tmp);
 
    return true;
}

int main(int argc, char* const argv[]) {
 int n = 1000; // Number of steps
    double S = 100; // Initial stock price
    double K = 100; // Strike price
    double r = 0.05; // Risk-free rate
    double v = 0.2; // Volatility
    double T = 1; // Time to maturity
    int PC = 0; // 0 for call, 1 for put
    int AM=1;
 
    int opt;

    while ((opt = getopt(argc, argv, "n:s:k:r:v:t:p:a:x")) != -1) {
        char* end;
        switch (opt) {
            case 'n': n = strtoumax(optarg, &end, 10); break;
            case 's': S = atof(optarg); break;
            case 'k': K = atof(optarg); break;
            case 'r': r = atof(optarg); break;
            case 'v': v = atof(optarg); break;
            case 't': T = atof(optarg); break;
            case 'p': PC = atoi(optarg);break;
            case 'a': AM = atoi(optarg);break;
        }
    }

    if ((optind + 7 != argc )&&(optind + 0 != argc)){
        fprintf(stderr, "usage: %s [-n num-steps] [-s initial-stock-price] [-k strike-price] [-r risk-free-rate] [-v volatility]  [-t time-to-maturity] [-p put-or-call] input output\n", argv[0]);
        return 1;
    }

    Matrix *O = matrix_create_raw(n+1, n+1);
    if (!OptionsVal(O,n, S, K, r, v, T, PC,AM)) { fprintf(stderr, "Failed to perform Put Options Price\n"); return 1; }


    // Result extraction
    if (PC==0)
        printf("Call options: %.2f\n",MATRIX_AT(O,0,0) );
    else
        printf("Put options: %.2f\n",MATRIX_AT(O,0,0) );

    return 0;
}