/**
 * Final Project - Binomial Lattice model(for option pricing)
 * 
 * This will produce values for both American and European options 
 * when using the corresponding argument (-a 1 for american -a 0 for european)
 * 
 * To compile:
 *      gcc -Wall -fopenmp -O3 -march=native bopm-openmp.c matrix.c util.c  -o bopm-openmp -lm
 * 
 * To run on your local machine:
 *    ./bopm_openmp
 * 
 * this will run with default values
 *
 int n = 1000; // Number of steps
 * double S = 100; // Initial stock price
 * double K = 100; // Strike price
 * double r = 0.05; // Risk-free rate (for 5%)
 * double q = 0.03; //dividend yield (for 3%)
 * double v = 0.2; // Volatility (for 2%)
 * double T = 1; // Time to maturity (1=1 year)
 * int PC = 1; // 0 for call, 1 for put
 * int AM=1; //American = 1 European=0
 * 
 * threads defaults to 8... can be changed with -c parameter
 * 
 * american put with 16 threads ./bopm-openmp -n 1000 -s 85 -k 90 -q 0.01 -r 0.03 -v 0.2 -t 1 -p 1 -a 1 -c 16
 * 
 * to change values run with these params:
 * 
 * usage: ./bopm-openmp [-n num-steps] [-s initial-stock-price] [-q dividend-yield] [-k strike-price] [-r risk-free-rate] [-v volatility]  [-t time-to-maturity] [-p put-or-call(1 or 0)] -a [american 1 european 0] -c[threads] input output
 *  
 * american put ./bopm-openmp -n 1000 -s 85 -k 90 -q 0.01 -r 0.03 -v 0.2 -t 1 -p 1 -a 1 
 * american call ./bopm-openmp -n 1000 -s 85 -k 90 -q 0.01 -r 0.03 -v 0.2 -t 1 -p 0 -a 1 
 * european put ./bopm-openmp -n 1000 -s 85 -k 90 -q 0.01 -r 0.03 -v 0.2 -t 1 -p 1 -a 0 
 * european call ./bopm-openmp -n 1000 -s 85 -k 90 -q 0.01 -r 0.03 -v 0.2 -t 1 -p 0 -a 0 
 * 
 * https://en.wikipedia.org/wiki/Binomial_options_pricing_model
 * https://www.unisalento.it/documents/20152/615419/Option+Pricing+-+A+Simplified+Approach.pdf
 * https://github.com/padraic00/Binomial-Options-Pricing-Model/tree/master
 * 
 * verified calculations are corrrect based on:
 * https://math.columbia.edu/~smirnov/options13.html
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <inttypes.h>
#include <omp.h> // Include OpenMP header


#include "matrix.h"
#include "util.h"

// number of time the timing loops are repeated to get better accuracy
#ifndef NUM_RUNS
#define NUM_RUNS 2 // we take the minimum across this number of runs
#endif
#ifndef NUM_ITER_PER_RUN
#define NUM_ITER_PER_RUN 2 // we take the average across this number of iterations for each run
#endif


typedef bool (*options_val_func)(Matrix* O, size_t n, double S, double q, double K, double r, double v, double T, int PC, int AM,int threads);

double time_options_val_func(const char* label, options_val_func func,
                             Matrix* O, size_t n, double S, double q, double K, double r, double v, double T, int PC, int AM,int threads) {
    struct timespec start, end;
    printf("%s", label);
    fflush(stdout);
    func(O, n, S, q, K, r, v, T, PC, AM,threads); // run the function once just to make sure the CPU is "warmed up"
    // We take the minimum across several runs
    double best_time = 0;
    for (int run = 0; run < NUM_RUNS; run++) {
        // We take the average across a few iterations
        clock_gettime(CLOCK_MONOTONIC, &start); // get the start time
        for (int iter = 0; iter < NUM_ITER_PER_RUN; iter++) {
            func(O, n, S, q, K, r, v, T, PC, AM,threads); // code that gets timed
        }
        clock_gettime(CLOCK_MONOTONIC, &end); // get the end time
        double time = get_time_diff(&start, &end) / NUM_ITER_PER_RUN; // compute average difference
        if (run == 0 || time < best_time) {
            best_time = time;
        }
    }
    print_time(best_time);
    printf("\n");
    return best_time;
}

bool OptionsVal(Matrix* O, size_t n, double S, double q, double K, double r, double v, double T, int PC, int AM,int threads) {
    double dt = T / n;
    double u = exp((r - q) * dt + v * sqrt(dt));
    double d = exp((r - q) * dt - v * sqrt(dt));
    double p = (exp((r - q) * dt) - d) / (u - d);

    matrix_fill_zeros(O);

    // Allocate memory for tmp array
    double **tmp = (double **)malloc((n + 1) * sizeof(double *));
    if (tmp == NULL) {
        fprintf(stderr,"Error:Memory allocation failed for tmp array.\n");
        return false;
    }
    for (size_t i = 0; i <= n; i++) {
        tmp[i] = (double *)malloc((n + 1) * sizeof(double));
        if (tmp[i] == NULL) {
            fprintf(stderr,"Error:Memory allocation failed for tmp array.\n");
             
            for(size_t j=0;j<i;j++)
            {
                free(tmp[j]);
            }
            free(tmp);
            return false;
        }
    }

   // printf("%d",omp_get_max_threads());
  #pragma omp parallel for shared(tmp,n,S,u,d) default(none) num_threads(threads)
//change threads to test times
    // Initialize stock price array
    for (size_t i = 0; i <= n; i++) {

        for (size_t j = 0; j <= i; j++) {
            tmp[j][i] = S * pow(u, (i - j)) * pow(d, j);
        }
    }

    //num_threads(8)
    // Calculate option values at maturity
     //#pragma omp parallel for default(none) shared(tmp, n, S, u, d, O, PC, AM, p, dt,K,r) num_threads(8)
        // num_threads(omp_get_max_threads())

   //{
   // #pragma omp for
   // #pragma omp parallel for shared(O,tmp,n,PC) default(none) num_threads(16)

    for (size_t j = 0; j <= n; j++) {
        if (PC == 1) { // Put
            MATRIX_AT(O, j, n) = fmax(0, K - tmp[j][n]); // Put option value
        } else { // Call
            MATRIX_AT(O, j, n) = fmax(0, tmp[j][n] - K); // Call option value
        }
    }
//}
//#pragma omp for
    // Backward induction to calculate option values at earlier time steps
   //leave int for neg
  //  #pragma omp parallel for default(none) schedule(static) shared(tmp, n, S, u, d, O, PC, AM, p, dt,K,r) num_threads(16)
//#pragma omp for
   // #pragma omp parallel for shared(O,tmp,n,PC,AM,p,dt,K,r) default(none) num_threads(16)

    for (int i = n - 1; i >= 0; i--) {

        for (size_t j = 0; j <= i; j++) {
            double immediate_exercise;
            if (AM) {
                // American option: consider immediate exercise
                if (PC == 1) { // Put
                    immediate_exercise = fmax(0, K - tmp[j][i]); // Immediate exercise value
                } else { // Call
                    immediate_exercise = fmax(0, tmp[j][i] - K); // Immediate exercise value
                }
                // Compare immediate exercise with continuation value
                MATRIX_AT(O, j, i) = fmax(immediate_exercise, exp(-r * dt) * (p * MATRIX_AT(O, j, i + 1) + (1 - p) * MATRIX_AT(O, j + 1, i + 1)));
            } else {
                // European option: only consider continuation value
                MATRIX_AT(O, j, i) = exp(-r * dt) * (p * MATRIX_AT(O, j, i + 1) + (1 - p) * MATRIX_AT(O, j + 1, i + 1));
            }
        }
    }
    
  // }
    // Free allocated memory
      //  #pragma omp parallel for shared(tmp) default(none) 

    for (size_t i = 0; i <= n; i++) {
        free(tmp[i]);
    }
    free(tmp);

    return true;
}

void parse_arguments(int argc, char* const argv[], int n, double  S, double K, double q, double r, double v, double T, int PC, int AM) {
    int n_flag = 0, s_flag = 0, q_flag = 0, k_flag = 0, r_flag = 0, v_flag = 0, t_flag = 0, p_flag = 0, a_flag = 0;

    int opt;
    while ((opt = getopt(argc, argv, "n:s:q:k:r:v:t:p:a:")) != -1) {
        char* end;
        switch (opt) {
        case 'n': {
            long val = strtol(optarg, &end, 10);
            if (*end != '\0') {
                fprintf(stderr, "Error: Non-numeric value provided for -n\n");
                exit(1);
            }
            n = val;
            n_flag = 1;
            break;
        }
        case 's': {
            char* end;
            double val = strtod(optarg, &end);
            if (*end != '\0') {
                fprintf(stderr, "Error: Non-numeric value provided for -s\n");
                exit(1);
            }
            S = val;
            s_flag = 1;
            break;
        }
        case 'q': {
            char* end;
            double val = strtod(optarg, &end);
            if (*end != '\0') {
                fprintf(stderr, "Error: Non-numeric value provided for -q\n");
                exit(1);
            }
            q = val;
            q_flag = 1;
            break;
        }
        case 'k': {
            char* end;
            double val = strtod(optarg, &end);
            if (*end != '\0') {
                fprintf(stderr, "Error: Non-numeric value provided for -k\n");
                exit(1);
            }
            K = val;
            k_flag = 1;
            break;
        }
        case 'r': {
            char* end;
            double val = strtod(optarg, &end);
            if (*end != '\0') {
                fprintf(stderr, "Error: Non-numeric value provided for -r\n");
                exit(1);
            }
            r = val;
            r_flag = 1;
            break;
        }
        case 'v': {
            char* end;
            double val = strtod(optarg, &end);
            if (*end != '\0') {
                fprintf(stderr, "Error: Non-numeric value provided for -v\n");
                exit(1);
            }
            v = val;
            v_flag = 1;
            break;
        }
        case 't': {
            char* end;
            double val = strtod(optarg, &end);
            if (*end != '\0') {
                fprintf(stderr, "Error: Non-numeric value provided for -t\n");
                exit(1);
            }
            T = val;
            t_flag = 1;
            break;
        }
        case 'p': {

         char* end;
            double val = strtod(optarg, &end);
            if (*end != '\0') {
                fprintf(stderr, "Error: Non-numeric value provided for -p\n");
                exit(1);
            }
           // int val2 = atoi(optarg);

            if (val != 0 && val != 1) {
                fprintf(stderr, "Error: Invalid value provided for -p (must be 0 or 1)\n");
                exit(1);
            }
            PC = val;
            p_flag = 1;
            break;
        }
        case 'a': {

         char* end;
            double val = strtod(optarg, &end);
            if (*end != '\0') {
                fprintf(stderr, "Error: Non-numeric value provided for -a\n");
                exit(1);
            }
   
            //int val2 = atoi(optarg);
            if (val != 0 && val != 1) {
                fprintf(stderr, "Error: Invalid value provided for -a (must be 0 or 1)\n");
                exit(1);
            }
            AM = val;
            a_flag = 1;
            break;
        }
        default:
            fprintf(stderr, "Unknown option: %c\n", opt);
            exit(1);
        }
    }

    if (!(n_flag && s_flag && q_flag && k_flag && r_flag && v_flag && t_flag && p_flag && a_flag)) {
        fprintf(stderr, "Error: Not all required arguments are provided.\n");
        fprintf(stderr, "usage: %s [-n num-steps] [-s initial-stock-price] [-k strike-price] [-q dividend-yield] [-r risk-free-rate] [-v volatility]  [-t time-to-maturity] [-p put-or-call] [-a American=1 European=0(not 1)] input output\n", argv[0]);
        exit(1);
    }
}

int main(int argc, char* const argv[]) {
   int n = 1000; // Number of steps
   double S = 100; // Initial stock price
   double K = 100; // Strike price
   double q = 0.03; //dividend yield
   double r = 0.05; // Risk-free rate
   double v = 0.2; // Volatility
   double T = 1; // Time to maturity
   int PC = 1; // 0 for call, 1 for put
   int AM=1;//American = 1 European=0
   
       parse_arguments(argc, argv, n, S, K, q, r, v, T, PC, AM);


   //int threads=omp_get_max_threads();

   int threads=8;

    const char* optionType;
    const char* exerciseType;

    optionType=(PC==0)?"Call":"Put";
    exerciseType=(AM==1)?"American":"European";

 // Displaying each argument one line for each
    printf("Threads: %d\n", threads);
 
    printf("Number of steps: %d\n", n);
    printf("Initial stock price: %.2f\n", S);
    printf("Strike price: %.2f\n", K);
    printf("Dividend yield: %.2f\n", q);
    printf("Risk-free rate: %.2f\n", r);
    printf("Volatility: %.2f\n", v);
    printf("Time to maturity: %.2f\n", T);
    printf("Calculating %s %s options\n", exerciseType, optionType);

    Matrix *O = matrix_create_raw(n+1, n+1);


   // struct timespec start, end;
   // clock_gettime(CLOCK_MONOTONIC,&start);
        // Calculate option price

     if (!OptionsVal(O,n, S,q, K, r, v, T, PC,AM,threads)) { fprintf(stderr, "Failed to perform Options Value\n"); return 1; }
   // clock_gettime(CLOCK_MONOTONIC,&end);
   // double time =get_time_diff(&start,&end);
   // print_time(time);
   // printf("\n");

  // time_options_val_func("bopm-openmp: ", OptionsVal, O,n, S,q, K, r, v, T, PC,AM);
   // printf("bopm-openmp speed is %.2f\n",time_mpi);

//clock_t start_time, end_time;
  //  double elapsed_time_ms;

   // start_time = clock(); // Record the start time

   // //if (!OptionsVal(O,n, S,q, K, r, v, T, PC,AM)) { fprintf(stderr, "Failed to perform Options Value\n"); return 1; }

   // end_time = clock(); // Record the end time

   // elapsed_time_ms = (double)(end_time - start_time) / (CLOCKS_PER_SEC / 1000); // Calculate the elapsed time in milliseconds

   // printf("Execution time: %.2f milliseconds\n", elapsed_time_ms);




   
printf("\n");
    printf("%s %s options: %.2f\n", exerciseType, optionType, MATRIX_AT(O, 0, 0));
printf("\n");



   time_options_val_func("bopm-openmp: ", OptionsVal, O,n, S,q, K, r, v, T, PC,AM,threads);



    return 0;
}
