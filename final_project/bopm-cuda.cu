/**
 * Final Project - Binomial Lattice model(for option pricing)
 * 
 * This will produce values for both American and European options 
 * when using the corresponding argument (-a 1 for american -a 0 for european)
 *  
 * To compile:
 *     nvcc -arch=sm_86 -O3 --compiler-options -march=native --expt-relaxed-constexpr bopm-cuda.cu -o bopm-cuda -lm
 *
 * Remember that you cannot compile or run this code on your laptop. You cannot
 * run or compile this code on the head node. You must compile and run this code
 * on a gpu-shared node. You can use the following command to get a node:
 *    srun --pty -p gpu-shared --exclusive /bin/bash
 * 
 * To run:
 *    ./bopm_cuda
 * 
 * this will run with default values
 *
 * int n = 1000; // Number of steps
 * double S = 100; // Initial stock price
 * double K = 100; // Strike price
 * double r = 0.05; // Risk-free rate (for 5%)
 * double q = 0.03; //dividend yield (for 3%)
 * double v = 0.2; // Volatility (for 2%)
 * double T = 1; // Time to maturity (1=1 year)
 * int PC = 1; // 0 for call, 1 for put
 * int AM=1; //American = 1 European=0(not 1)
 * 
 * to change values run with these params:
 * 
 * usage: ./bopm-cuda [-n num-steps] [-s initial-stock-price] [-q dividend-yield] [-k strike-price] [-r risk-free-rate] [-v volatility]  [-t time-to-maturity] [-p put-or-call(1 or 0)] -a [american 1 european 0]input output
 *  
 * american put ./bopm-cuda -n 1000 -s 85 -k 90 -q 0.01 -r 0.03 -v 0.2 -t 1 -p 1 -a 1 
 * american call ./bopm-cuda -n 1000 -s 85 -k 90 -q 0.01 -r 0.03 -v 0.2 -t 1 -p 0 -a 1 
 * european put ./bopm-cuda -n 1000 -s 85 -k 90 -q 0.01 -r 0.03 -v 0.2 -t 1 -p 1 -a 0 
 * european call ./bopm-cuda -n 1000 -s 85 -k 90 -q 0.01 -r 0.03 -v 0.2 -t 1 -p 0 -a 0 
 * 
 * https://en.wikipedia.org/wiki/Binomial_options_pricing_model
 * https://www.unisalento.it/documents/20152/615419/Option+Pricing+-+A+Simplified+Approach.pdf
 * https://github.com/padraic00/Binomial-Options-Pricing-Model/tree/master
 * 
 * verified calculations are corrrect based on:
 * https://math.columbia.edu/~smirnov/options13.html
 * 
 * 
 */


#include "matrix.hpp"

#include <iostream>
#include <iomanip>
#include <array>
#include <time.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include <cmath>

// number of time the timing loops are repeated to get better accuracy
#ifndef NUM_RUNS
#define NUM_RUNS 2 // we take the minimum across this number of runs
#endif
#ifndef NUM_ITER_PER_RUN
#define NUM_ITER_PER_RUN 2 // we take the average across this number of iterations for each run
#endif



template<typename T>
__global__ void options_val_cuda_kernel(T *O, size_t n, double S, double q, double K, double r, double v, double TM, int PC, int AM, int level) {
    double dt = TM / n;
    double u = exp((r - q) * dt + v * sqrt(dt));
    double d = exp((r - q) * dt - v * sqrt(dt));
    double p = (exp((r - q) * dt) - d) / (u - d);

    int i = n - level; // Calculate the current level

    // Calculate stock price at each node for this thread
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        double Pm = S * pow(u, i - j) * pow(d, j);
        if (PC == 1) { // Put
            O[j] = fmax(0, K - Pm); // American Put option value at maturity
        } else { // Call
            O[j] = fmax(0, Pm - K); // American Call option value at maturity
        }
    }

    __syncthreads();

    // Backward induction for option price
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        double immediateExercise;
        if (AM) {
            // American option: consider immediate exercise
            if (PC == 1) { // Put
                immediateExercise = fmax(0, K - S * pow(u, i - j) * pow(d, j)); // Immediate exercise value
            } else { // Call
                immediateExercise = fmax(0, S * pow(u, i - j) * pow(d, j) - K); // Immediate exercise value
            }
            // Compare immediate exercise with continuation value
            double continuationValue = exp(-r * dt) * (p * O[j] + (1 - p) * O[j + 1]);
            O[j] = fmax(immediateExercise, continuationValue);
        } else {
            // European option: only consider continuation value
            O[j] = exp(-r * dt) * (p * O[j] + (1 - p) * O[j + 1]);
        }
    }
}
template<typename T>
void options_val_cuda(Matrix<T>& O, size_t n, double S, double q, double K, double r, double v, double TM, int PC, int AM) {
    T *d_O; // Pointer for GPU memory
    T* O_data = O.data;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_O, sizeof(T) * (n + 1));

    // Copy data from CPU to GPU
    cudaMemcpy(d_O, O_data, sizeof(T) * (n + 1), cudaMemcpyHostToDevice);

    // Define block size (number of threads per block)
    constexpr int blockSize = 128; // Adjust as needed

    // Calculate grid size based on the total number of elements (n + 1)
    int gridSize = (n + blockSize - 1) / blockSize;

    // Invoke kernel
   // options_val_cuda_kernel<<<gridSize, blockSize>>>(d_O, n, S, q, K, r, v, TM, PC, AM);

  for (int level = n - 1; level >= 0; --level) {
    options_val_cuda_kernel<<<gridSize, blockSize>>>(d_O, n, S, q, K, r, v, TM, PC, AM,level);
        cudaDeviceSynchronize(); // Wait for the kernel to finish
      }



    // Copy results from GPU to CPU
    cudaMemcpy(O_data, d_O, sizeof(T) * (n + 1), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_O);
}

template<typename T>
void options_val_cuda_memonly(Matrix<T>& O, size_t n, double S, double q, double K, double r, double v, double TM, int PC, int AM) {
    T *d_O; // Pointer for GPU memory
    T* O_data = O.data;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_O, sizeof(T) * (n + 1));

    // Copy data from CPU to GPU
    cudaMemcpy(d_O, O_data, sizeof(T) * (n + 1), cudaMemcpyHostToDevice);

    // Define block size (number of threads per block)
    //constexpr int blockSize = 128; // Adjust as needed

    // Calculate grid size based on the total number of elements (n + 1)
   // int gridSize = (n + blockSize - 1) / blockSize;

    //removed kernel call

    // Copy results from GPU to CPU
    cudaMemcpy(O_data, d_O, sizeof(T) * (n + 1), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_O);
}



/**
 * Get the difference between two times.
 */
double get_time_diff(struct timespec* start, struct timespec* end) {
    double diff = end->tv_sec - start->tv_sec;
    diff += (end->tv_nsec - start->tv_nsec) / 1000000000.0;
    return diff;
}

/**
 * Prints a positive number with the given number of sigfigs and a unit. The
 * value is scaled to the correct unit (which are mult apart - 1000 for SI and
 * 1024 for digit prefixes).
 */
template<size_t n_units>
std::string format_with_unit(double val, int sigfigs, int mult,
                             const std::array<std::string, n_units> units) {
    size_t i_unit = 0;
    while (i_unit < n_units && val >= mult) { val /= mult; i_unit++; }
    if (i_unit == 0) { sigfigs = 0; }
    else if (val < 10) { sigfigs -= 1; }
    else if (val < 100) { sigfigs -= 2; }
    else { sigfigs -= 3; }

    std::ostringstream out;
    out << std::fixed << std::setprecision(sigfigs) << val << " " << units[i_unit];
    return out.str();
}

/**
 * Prints a number of bytes after converting to a nicer unit.
 */
std::string format_bytes(size_t n) {
    static const std::array<std::string, 4> units{"bytes", "KiB", "MiB", "GiB"};
    return format_with_unit(n, 3, 1024, units);
}

/**
 * Print the time (in seconds) with the right units and 3 significant digits.
 */
std::string format_time(double seconds) {
    static const std::array<std::string, 4> units{"ns", "us", "ms", "s"};
    return format_with_unit(seconds * 1000000000.0, 3, 1000, units);
}

/**
 * Time a options function.
 */


template<typename T>
double time_options_val_cuda_func(Matrix<T>& O, size_t n, double S, double q, double K, double r, double v, double TM, int PC, int AM) {
    struct timespec start, end;
    std::cout << "options_val_cuda: " << std::flush;
    options_val_cuda(O, n, S,q, K, r, v, TM, PC, AM); // run the function once just to make sure the CPU is "warmed up"
    // We take the minimum across several runs
    double best_time = 0;
    for (int run = 0; run < NUM_RUNS; run++) {
        // We take the average across a few iterations
        clock_gettime(CLOCK_MONOTONIC, &start); // get the start time
        for (int iter = 0; iter < NUM_ITER_PER_RUN; iter++) {
           options_val_cuda(O, n, S,q, K, r, v, TM, PC, AM); // code that gets timed
        }
        clock_gettime(CLOCK_MONOTONIC, &end); // get the end time
        double time = get_time_diff(&start, &end) / NUM_ITER_PER_RUN; // compute average difference
        if (run == 0 || time < best_time) {
            best_time = time;
        }
    }
    std::cout << format_time(best_time) << std::endl;
    return best_time;
}

template<typename T>
double time_options_val_cuda_func_mem(Matrix<T>& O, size_t n, double S, double q, double K, double r, double v, double TM, int PC, int AM) {
    struct timespec start, end;
    std::cout << "options_val_cuda_memonly: " << std::flush;
    options_val_cuda_memonly(O, n, S,q, K, r, v, TM, PC, AM); // run the function once just to make sure the CPU is "warmed up"
    // We take the minimum across several runs
    double best_time = 0;
    for (int run = 0; run < NUM_RUNS; run++) {
        // We take the average across a few iterations
        clock_gettime(CLOCK_MONOTONIC, &start); // get the start time
        for (int iter = 0; iter < NUM_ITER_PER_RUN; iter++) {
            options_val_cuda_memonly(O, n, S,q, K, r, v, TM, PC, AM); // run the function once just to make sure the CPU is "warmed up"
        }
        clock_gettime(CLOCK_MONOTONIC, &end); // get the end time
        double time = get_time_diff(&start, &end) / NUM_ITER_PER_RUN; // compute average difference
        if (run == 0 || time < best_time) {
            best_time = time;
        }
    }
    std::cout << format_time(best_time) << std::endl;
    return best_time;
}


template<typename T>
int run(size_t n, double S,double q, double K, double r, double v, double TM, int PC, int AM) {

    
    Matrix<T> O(n + 1, n + 1);
   
    try {
       options_val_cuda(O, n, S,q, K, r, v, TM, PC, AM); 
    std::string optionType = (PC == 0) ? "Call" : "Put";
    std::string exerciseType = (AM == 1) ? "American" : "European"; 

    std::cout << "Number of steps: " << n << std::endl;
    std::cout << "Initial stock price: " << std::fixed << std::setprecision(2) << S << std::endl;
    std::cout << "Strike price: " << std::fixed << std::setprecision(2) << K << std::endl;
    std::cout << "Dividend yield: " << std::fixed << std::setprecision(2) << q << std::endl;
    std::cout << "Risk-free rate: " << std::fixed << std::setprecision(2) << r << std::endl;
    std::cout << "Volatility: " << std::fixed << std::setprecision(2) << v << std::endl;
    std::cout << "Time to maturity: " << std::fixed << std::setprecision(2) << TM << std::endl;
    std::cout << "Calculating " << exerciseType << " " << optionType << " options" << std::endl;
    std::cout <<  exerciseType << " " << optionType << " options: ";

    std::cout << std::fixed << std::setprecision(2) << O(0, 0) << std::endl;

    double time_cuda = time_options_val_cuda_func(O, n, S,q, K, r, v, TM, PC, AM);
    double time_cuda_mem = time_options_val_cuda_func_mem(O, n, S,q, K, r, v, TM, PC, AM);
    double time_cuda_no_mem = time_cuda - time_cuda_mem;
    std::cout << "cuda no mem: " << format_time(time_cuda_no_mem) << std::endl;





    } catch (const std::exception& e) {
        std::cerr << "Failed to perform CUDA Option Pricing: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

int main(int argn, const char* argv[]) {
    int n = 1000; // Number of steps
    double S = 100; // Initial stock price
    double K = 100; // Strike price
    double q = 0.03; //dividend yield

    double r = 0.05; // Risk-free rate
    double v = 0.2; // Volatility
    double TM = 1; // Time to maturity
    int PC = 0; // 0 for call, 1 for put
    int AM = 1; //American = 1 European=0(not 1)

    int n_flag = 0;
    int s_flag = 0;
    int q_flag = 0;
    int k_flag = 0;
    int r_flag = 0;
    int v_flag = 0;
    int t_flag = 0;
    int p_flag = 0;
    int a_flag = 0;


    int opt;

    
   while ((opt = getopt(argn, const_cast<char* const*>(argv), "n:s:q:k:r:v:t:p:a:")) != -1) {
        switch (opt) {
            case 'n': n = std::strtoull(optarg, nullptr, 10); n_flag=1; break;
            case 's': S = std::atof(optarg); s_flag=1; break;
            case 'q': q = std::atof(optarg); q_flag=1; break;

            case 'k': K = std::atof(optarg); k_flag=1; break;
            case 'r': r = std::atof(optarg); r_flag=1; break;
            case 'v': v = std::atof(optarg); v_flag=1; break;
            case 't': TM = std::atof(optarg); t_flag=1; break;
            case 'p': PC = std::atoi(optarg); p_flag=1; break;
            case 'a': AM = std::atoi(optarg); a_flag=1; break;
            default:
                std::cerr << "Invalid option\n";
                return 1;
        }
    }

   if (argn > 1)
    {
    if (!(n_flag && s_flag && q_flag && k_flag && r_flag && v_flag && t_flag && p_flag && a_flag)) {
        std::cerr << "usage: " << argv[0] << " [-n num-steps] [-s initial-stock-price] [-q dividend-yield] [-k strike-price] [-r risk-free-rate] [-v volatility]  [-t time-to-maturity] [-p put-or-call] [-a American=1 European=0(not 1)] input output\n";
        return 1;
    }
   }
    int retval = run<double>(n,S,q,K,r,v,TM,PC,AM);


    // finalize using CUDA
    cudaDeviceReset();

    return retval;
}
