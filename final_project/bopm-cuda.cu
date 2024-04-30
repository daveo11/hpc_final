/**
 * Final Project - Binomial Lattice model(for option pricing)
 * 
 * This will produce values for both American and European options 
 * when using the corresponding argument (-a 1 for american -a 0 for european)
 *  
 * To compile:
 *     expanse: nvcc -arch=sm_70 -O3 --compiler-options -march=native --expt-relaxed-constexpr bopm-cuda.cu -o bopm-cuda -lm
 *     mucluster:  nvcc -arch=sm_86 -O3 --compiler-options -march=native --expt-relaxed-constexpr bopm-cuda.cu -o bopm-cuda -lm
 * 
 * Remember that you cannot compile or run this code on your laptop. You cannot
 * run or compile this code on the head node. You must compile and run this code
 * on a gpu-shared node. You can use the following command to get a node:
 * 
 *   expanse: srun -p gpu-shared --gpus 1 --pty -A mor101 -n 1 -N 1 -t 00:30:00 --wait=0 --export=ALL /bin/bash
 *   mucluster: srun --pty -p gpu-shared --exclusive /bin/bash
 * 
 * To run with default values:
 *    ./bopm-cuda
 *
 * int n = 1000; // Number of steps
 * double S = 100; // Initial stock price
 * double K = 100; // Strike price
 * double r = 0.05; // Risk-free rate (for 5%)
 * double q = 0.03; //dividend yield (for 3%)
 * double v = 0.2; // Volatility (for 2%)
 * double T = 1; // Time to maturity (1=1 year)
 * int PC = 1; // 0 for call, 1 for put
 * int AM=1; //American = 1 European=0
 * 
 * to change values run with these parameters:
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
 * https://www.codearmo.com/python-tutorial/options-trading-binomial-pricing-model
 * https://leimao.github.io/blog/Proper-CUDA-Error-Checking/ 
 * https://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf
 * 
 * verified calculations are corrrect based on:
 * https://math.columbia.edu/~smirnov/options13.html
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

// Function to check for CUDA errors
#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(-1); \
    }

/**
 * Binomial lattice model for option pricing function.
 * The option value will be in the matrix at (0,0) when
 * all processing completes.This code runs on the GPU.
 * Each thread executes the same kernel code, but they can 
 * be differentiated by the threadIdx and blockDim 
 * 
 */
template<typename T>
    __global__ void options_val_cuda_kernel(T *O, size_t n, double S, double q, double K, double r, double v, double TM, int PC, int AM) {
    double dt = TM / n;
    double u = exp((r - q) * dt + v * sqrt(dt));
    double d = exp((r - q) * dt - v * sqrt(dt));
    double p = (exp((r - q) * dt) - d) / (u - d);

    // Calculate stock price at each node for this thread
    for (int j = threadIdx.x; j <= n; j += blockDim.x) {
        double Pm = S * pow(u, n - j) * pow(d, j);
        if (PC == 1) { // Put
            O[j] = fmax(0, K - Pm); // American Put option value at maturity
        } else { // Call
            O[j] = fmax(0, Pm - K); // American Call option value at maturity
        }
    }

    __syncthreads();

    // Backward induction for option price
    for (int i = n - 1; i >= 0; i--) {
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
        __syncthreads(); // Synchronize here to ensure all threads have updated their values
    }
}

/**
 * This function sets up all the GPU memory allocation and calls the kernel function on the GPU.
 * NOTES: The Matrix objects use host memory. They are  
 * matrices of type T, which could be either floats or doubles,however we are using double 
 * because it it a financial application and it is much more accurate to use double
 */
template<typename T>
void options_val_cuda(Matrix<T>& O, size_t n, double S, double q, double K, double r, double v, double TM, int PC, int AM) {
    T *d_O; // Pointer for GPU memory
    T* O_data = O.data;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_O, sizeof(T) * (n + 1));
    CUDA_CHECK_ERROR(cudaGetLastError());

    // Copy data from CPU to GPU
    cudaMemcpy(d_O, O_data, sizeof(T) * (n + 1), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(cudaGetLastError());

    // Define block size (number of threads per block)
    constexpr int blockSize = 128; // Adjust as needed

    // Calculate grid size based on the total number of elements (n + 1)
    int gridSize = (n + blockSize - 1) / blockSize;

    // Invoke kernel
    options_val_cuda_kernel<<<gridSize, blockSize>>>(d_O, n, S, q, K, r, v, TM, PC, AM);

    // Copy results from GPU to CPU
    cudaMemcpy(O_data, d_O, sizeof(T) * (n + 1), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(cudaGetLastError());

    // Free GPU memory
    cudaFree(d_O);
}

/**
 * Benchmark function for determining how long memory copies take. This is
 * used to determine how much time just memory operations take.
 */
template<typename T>
void options_val_cuda_memonly(Matrix<T>& O, size_t n, double S, double q, double K, double r, double v, double TM, int PC, int AM) {
    T *d_O; // Pointer for GPU memory
    T* O_data = O.data;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_O, sizeof(T) * (n + 1));
    CUDA_CHECK_ERROR(cudaGetLastError());

    // Copy data from CPU to GPU
    cudaMemcpy(d_O, O_data, sizeof(T) * (n + 1), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(cudaGetLastError());

    // Copy results from GPU to CPU
    cudaMemcpy(O_data, d_O, sizeof(T) * (n + 1), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(cudaGetLastError());

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
double time_options_val_cuda_func(const char* label, void (*func)(Matrix<T>&, size_t, double, double, double, double, double, double, int, int),
                                  Matrix<T>& O, size_t n, double S, double q ,double K, double r, double v, double TM, int PC, int AM) {
    struct timespec start, end;
    std::cout << label << std::flush;
    func(O, n, S, q, K, r, v, TM, PC, AM); // run the function once just to make sure the GPU is "warmed up"
    // We take the minimum across several runs
    double best_time = 0;
    for (int run = 0; run < NUM_RUNS; run++) {
        // We take the average across a few iterations
        clock_gettime(CLOCK_MONOTONIC, &start); // get the start time
        for (int iter = 0; iter < NUM_ITER_PER_RUN; iter++) {
            func(O, n, S, q, K, r, v, TM, PC, AM); // code that gets timed
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

/**
 * function calls the options_val_cuda and displays arguments used and does the benchmark timings
 */

template<typename T>
int run(size_t n, double S,double q, double K, double r, double v, double TM, int PC, int AM) {

    Matrix<T> O(n + 1, n + 1);
   
    try {
       options_val_cuda(O, n, S,q, K, r, v, TM, PC, AM); 
       std::string optionType = (PC == 0) ? "Call" : "Put";
       std::string exerciseType = (AM == 1) ? "American" : "European"; 

       // Displaying each argument one line for each
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

       double time_cuda = time_options_val_cuda_func("Cuda:      ",options_val_cuda,O, n, S,q, K, r, v, TM, PC, AM);
       double time_cuda_mem = time_options_val_cuda_func("Cuda mem:      ",options_val_cuda_memonly,O, n, S,q, K, r, v, TM, PC, AM);
       double time_cuda_no_mem = time_cuda - time_cuda_mem;
       std::cout << "cuda no mem: " << format_time(time_cuda_no_mem) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to perform CUDA Option Pricing: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

int main(int argn, const char* argv[]) {

    //set default values
    int n = 1000; // Number of steps
    double S = 100; // Initial stock price
    double K = 100; // Strike price
    double q = 0.03; //dividend yield
    double r = 0.05; // Risk-free rate
    double v = 0.2; // Volatility
    double TM = 1; // Time to maturity
    int PC = 0; // 0 for call, 1 for put
    int AM = 1; //American = 1 European=0(not 1)

   //flags to identify if all arguments are added 
    int n_flag = 0;
    int s_flag = 0;
    int q_flag = 0;
    int k_flag = 0;
    int r_flag = 0;
    int v_flag = 0;
    int t_flag = 0;
    int p_flag = 0;
    int a_flag = 0;

    //parse input args
    //needed const_cast<char* const*> for getopt to prevent warning with c++ compile
    int opt;
    while ((opt = getopt(argn,  const_cast<char* const*>(argv), "n:s:q:k:r:v:t:p:a:")) != -1) {
        char* end;
        switch (opt) {
            case 'n': {
                long val = std::strtol(optarg, &end, 10);
                if (*end != '\0') {
                    std::cerr << "Error: Non-numeric value provided for -n" << std::endl;
                    exit(1);
                }
                n = val;
                n_flag = 1;
                break;
            }
            case 's': {
                double val = std::strtod(optarg, &end);
                if (*end != '\0') {
                    std::cerr << "Error: Non-numeric value provided for -s" << std::endl;
                    exit(1);
                }
                S = val;
                s_flag = 1;
                break;
            }
            case 'q': {
                double val = std::strtod(optarg, &end);
                if (*end != '\0') {
                    std::cerr << "Error: Non-numeric value provided for -q" << std::endl;
                    exit(1);
                }
                q = val;
                q_flag = 1;
                break;
            }
            case 'k': {
                double val = std::strtod(optarg, &end);
                if (*end != '\0') {
                    std::cerr << "Error: Non-numeric value provided for -k" << std::endl;
                    exit(1);
                }
                K = val;
                k_flag = 1;
                break;
            }
            case 'r': {
                double val = std::strtod(optarg, &end);
                if (*end != '\0') {
                    std::cerr << "Error: Non-numeric value provided for -r" << std::endl;
                    exit(1);
                }
                r = val;
                r_flag = 1;
                break;
            }
            case 'v': {
                double val = std::strtod(optarg, &end);
                if (*end != '\0') {
                    std::cerr << "Error: Non-numeric value provided for -v" << std::endl;
                    exit(1);
                }
                v = val;
                v_flag = 1;
                break;
            }
            case 't': {
                double val = std::strtod(optarg, &end);
                if (*end != '\0') {
                    std::cerr << "Error: Non-numeric value provided for -t" << std::endl;
                    exit(1);
                }
                TM = val;
                t_flag = 1;
                break;
            }
            case 'p': {
                double val = std::strtod(optarg, &end);
                if (*end != '\0') {
                    std::cerr << "Error: Non-numeric value provided for -p" << std::endl;
                    exit(1);
                }
                if (val != 0 && val != 1) {
                    std::cerr << "Error: Invalid value provided for -p (must be 0 or 1)" << std::endl;
                    exit(1);
                }
                PC = static_cast<int>(val);
                p_flag = 1;
                break;
            }
            case 'a': {
                double val = std::strtod(optarg, &end);
                if (*end != '\0') {
                    std::cerr << "Error: Non-numeric value provided for -a" << std::endl;
                    exit(1);
                }
                if (val != 0 && val != 1) {
                    std::cerr << "Error: Invalid value provided for -a (must be 0 or 1)" << std::endl;
                    exit(1);
                }
                AM = static_cast<int>(val);
                a_flag = 1;
                break;
            }
            default:
                std::cerr << "Unknown option: " << static_cast<char>(opt) << std::endl;
                exit(1);
        }
    }

   //check if any arguments are specified. if any are specified, then you need to enter all. They are "optional" because you can run with just
   //the defaults

    if (argn > 1){
        if (!(n_flag && s_flag && q_flag && k_flag && r_flag && v_flag && t_flag && p_flag && a_flag)) {
            std::cerr << "Error: Not all required arguments are provided." << std::endl;
            std::cerr << "usage: " << argv[0] << " [-n num-steps] [-s initial-stock-price] [-k strike-price] [-q dividend-yield] [-r risk-free-rate] [-v volatility]  [-t time-to-maturity] [-p put-or-call] [-a American=1 European=0(not 1)] input output" << std::endl;
            exit(1);
    }
   }
    //start the process
    int retval = run<double>(n,S,q,K,r,v,TM,PC,AM);

    // finalize using CUDA
    cudaDeviceReset();

    return retval;
}
