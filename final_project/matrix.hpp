/** Defines a C++ matrix class. */
#pragma once

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/mman.h>  // for mmap() and munmap()
#include <unistd.h>  // for sysconf() and _SC_PAGE_SIZE
#include <string.h>  // for memcpy()


// forward declarations
template <typename T>
static inline size_t __read_csv_first_line(std::string& line, T** out);
template <typename T>
static inline void __read_csv_line(std::string& line, T* out, size_t count);
template <typename T>
static inline size_t __npy_read_header(std::istream& file, size_t* shape);


#define DATA_MALLOCED   1 // data is from malloc(), needs free()
#define DATA_MEMMAPPED  2 // data is from mmap(), needs munmap()
#define DATA_BORROWED   3 // data is from elsewhere and should not be freed


template <typename T>
class Matrix {
private:
    char _data_source; // one of DATA_MALLOCED, DATA_MEMMAPPED, or DATA_BORROWED
    size_t _rows, _cols, _size;
    T* _data;
    void _free() {
        if (_data_source == DATA_MEMMAPPED) {
            size_t addr = (((size_t)_data) & ~(sysconf(_SC_PAGE_SIZE)-1));
            munmap((void*)addr, _size*sizeof(T));
        } else if (_data_source == DATA_MALLOCED) { delete[] _data; }
    }

    Matrix(size_t rows, size_t cols, T* data, char data_source) :
        _rows(rows), _cols(cols), _size(rows * cols), _data(data), _data_source(data_source),
        rows(_rows), cols(_cols), size(_size), data(_data) { }

public:
    // Publicly accessible members
    const size_t& rows;
    const size_t& cols;
    const size_t& size; // size is simply rows*cols, but it comes up a lot
    T* const & data;

    Matrix() : Matrix(0, 0, NULL, DATA_BORROWED) { }
    Matrix(size_t rows, size_t cols) : Matrix(rows, cols, new T[rows*cols], DATA_MALLOCED) { }
    Matrix(const Matrix<T>& M) : Matrix(M._rows, M._cols, new T[M._size], DATA_MALLOCED) { memcpy(_data, M._data, _size*sizeof(T)); }
    Matrix(Matrix<T>&& M) : _rows(M._rows), _cols(M._cols), _size(M._size), _data(M._data), _data_source(M._data_source),
        rows(_rows), cols(_cols), size(_size), data(_data) { M._rows = 0; M._cols = 0; M._size = 0; M._data = NULL; M._data_source = DATA_BORROWED; }
    ~Matrix() { _free(); }
    Matrix<T>& operator=(const Matrix<T>& M) {
        if (_size != M._size || _data_source != DATA_MALLOCED) {
            _free();
            _data = new T[M._size];
            _data_source = DATA_MALLOCED;
        }
        _rows = M._rows;
        _cols = M._cols;
        _size = M._size;
        memcpy(_data, M._data, _size*sizeof(T));
        return *this;
    }
    Matrix<T>& operator=(Matrix<T>&& M) {
        std::swap(_rows, M._rows);
        std::swap(_cols, M._cols);
        std::swap(_size, M._size);
        std::swap(_data, M._data);
        std::swap(_data_source, M._data_source);
        return *this;
    }

    // Data type conversions
    template <typename T2>
    Matrix(const Matrix<T2>& M) : Matrix(M.rows, M.cols, new T[M.size], DATA_MALLOCED) {
        for (size_t i = 0; i < _size; i++) { _data[i] = static_cast<T>(M.data[i]); }
    }
    template <typename T2>
    Matrix<T>& operator=(const Matrix<T2>& M) {
        if (_size != M.size || _data_source != DATA_MALLOCED) {
            _free();
            _data = new T[M.size];
            _data_source = DATA_MALLOCED;
        }
        _rows = M.rows;
        _cols = M.cols;
        _size = M.size;
        for (size_t i = 0; i < _size; i++) { _data[i] = static_cast<T>(M.data[i]); }
        return *this;
    }

    /**
     * Creates a new matrix by loading the data from the given CSV file. It is
     * assumed that every row in the file has the same number of values. If any
     * row has more values than the first row, those extra values are ignored. If
     * any row has less than the first row, the missing data is filled with 0s.
     */
    static Matrix<T> from_csv(std::istream& file) {
        // Get the first line from the file
        std::string line;
        std::getline(file, line);

        // Parse the first line of the file, getting the number of columns
        T* data = NULL;
        size_t rows = 1, cols = __read_csv_first_line(line, &data);
        if (cols == 0) { free(data); throw std::invalid_argument("No data in CSV file.");}

        // Get remaining rows of the file
        size_t rows_alloc = cols; // assume square, adjust later if needed
        data = (T*)realloc(data, cols*rows_alloc*sizeof(T));
        while (file.good()) {
            std::getline(file, line);
            __read_csv_line(line, &data[rows*cols], cols);
            if (++rows == rows_alloc) {
                rows_alloc += rows_alloc > 128 ? 128 : rows_alloc;
                data = (T*)realloc(data, rows_alloc*cols*sizeof(T));
            }
        }

        // Make the final matrix
        return Matrix(rows, cols, (T*)realloc(data, rows*cols*sizeof(T)), DATA_MALLOCED);
    }

    /**
     * Same as matrix_from_csv() but takes a file path instead.
     */
    static Matrix<T> from_csv(const char* path) {
        auto file = std::ifstream(path);
        return from_csv(file);
    }

    /** Saves a matrix to a CSV file. */
    void to_csv(std::ostream& file) const {
        if (_rows < 1 || _cols < 1) { return; }
        for (size_t i = 0, ind = 0; i < _rows; i++, ind++) {
            for (size_t j = 0; j < _cols-1; j++, ind++) { file << _data[ind] << ','; }
            file << _data[ind] << '\n';
        }
    }

    /** Same as matrix_to_csv() but takes a file path instead. */
    void to_csv(const char* path) const {
        auto file = std::ofstream(path);
        to_csv(file);
    }

private:
    static const char* get_npy_dtype() {
        const char* descr =
            std::is_same<T, float>::value ? "<f4" : std::is_same<T, double>::value ? "<f8" :
            std::is_same<T, int8_t>::value ? "<i1" : std::is_same<T, int16_t>::value ? "<i2" : std::is_same<T, int32_t>::value ? "<i4" : std::is_same<T, int64_t>::value ? "<i8" :
            std::is_same<T, uint8_t>::value ? "<u1" : std::is_same<T, uint16_t>::value ? "<u2" : std::is_same<T, uint32_t>::value ? "<u4" : std::is_same<T, uint64_t>::value ? "<u8" : NULL;
        if (descr == NULL) { throw std::domain_error("Matrix<T> unsupported to_npy()"); }
        return descr;
    }

public:
    /**
     * Creates a new matrix by loading the data from the given NPY file. This is
     * a file format used by the numpy library. This function only supports arrays
     * that are little-endian doubles, c-contiguous, and 1 or 2 dimensional. The
     * file is loaded as memory-mapped so it is backed by the file and loaded
     * on-demand. The file should be opened for reading or reading and writing.
     */
    static Matrix<T> from_npy(std::istream& file) {
        // Read the header, check it, and get the shape of the matrix
        size_t shape[2] = { 0, 0 }, offset = __npy_read_header<T>(file, shape);

        // TODO: Get the memory mapped data
        //char* data = (char*)mmap(NULL, shape[0]*shape[1]*sizeof(T) + offset,
        //                         PROT_READ|PROT_WRITE, MAP_SHARED, fileno(file), 0);
        //if (data == MAP_FAILED) { throw std::runtime_error("Failed to memory map file."); }
        //data += offset
        T* data = new T[shape[0]*shape[1]];
        if (!file.seekg(offset).read((char*)data, shape[0]*shape[1]*sizeof(T))) {
            delete[] data;
            throw std::runtime_error("Failed to read NPY data.");
        }

        // Make the matrix itself
        return Matrix(shape[0], shape[1], (T*)data, DATA_MEMMAPPED);
    }

    /** Same as matrix_from_npy() but takes a file path instead. */
    static Matrix<T> from_npy(const char* path) {
        auto file = std::ifstream(path, std::ios_base::in | std::ios_base::binary);
        return from_npy(file);
    }

    /**
     * Saves a matrix to a NPY file. This is a file format used by the numpy
     * library. This will return false if the data cannot be written.
     */
    void to_npy(std::ostream& file) const {
        // create the header
        char header[128];
        int len = snprintf(header, sizeof(header), "\x93NUMPY\x01   "
            "{'descr': '%s', 'fortran_order': False, 'shape': (%zu, %zu), }",
            get_npy_dtype(), _rows, _cols);
        if (len < 0) { throw std::runtime_error("Failed to write NPY header."); }
        header[7] = 0; // have to after the string is written
        *(unsigned short*)&header[8] = sizeof(header) - 10;
        memset(header + len, ' ', sizeof(header)-len-1);
        header[sizeof(header)-1] = '\n';

        // write the header and the data
        if (!file.write(header, sizeof(header)).write(_data, sizeof(T)*_size)) { throw std::runtime_error("Failed to write NPY data.");}
    }

    /** Same as matrix_to_npy() but takes a file path instead. */
    void to_npy(const char* path) const {
        auto file = std::ofstream(path, std::ios_base::out | std::ios_base::binary);
        to_npy(file);
    }


    /** Fills a matrix with all 0s. */
    inline Matrix<T>& fill_zeros() { std::fill(_data, _data + _size, (T)0); return *this; }
    /** Fills a matrix with all the same value. */
    inline Matrix<T>& fill(T value) { std::fill(_data, _data + _size, value); return *this; }
    /** The data is all set to zeroes except the main diagonal which is set to ones. */
    Matrix<T>& identity() {
        fill_zeros();
        size_t size = std::min(_rows, _cols);
        for (size_t i = 0, index = 0; i < _size; index += _cols+1) { _data[index] = (T)1; }
        return *this;
    }
    /** The data is filled in random values in [0.0, 1.0]. */
    Matrix<T>& random() {
        std::random_device dev;
        std::mt19937_64 rng(dev());

        using dist_t = typename std::conditional<
            std::is_integral<T>::value, 
            typename std::conditional<
                std::is_signed<T>::value,
                std::uniform_int_distribution<long long>,
                std::uniform_int_distribution<unsigned long long>>::type,
            std::uniform_real_distribution<T>>::type;

        if (std::is_integral<T>::value) {
            dist_t distribution(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
            for (size_t i = 0; i < _size; i++) { _data[i] = (T)distribution(rng); }
        } else if (std::is_floating_point<T>::value) {
            dist_t distribution((T)0, (T)1);
            for (size_t i = 0; i < _size; i++) { _data[i] = (T)distribution(rng); }
        } else {
            throw std::domain_error("Matrix<T> unsupported random()");
        }

        return *this;
    }

    Matrix<T> extract(size_t row_start, size_t row_end, size_t col_start, size_t col_end) const {
        if (row_start >= _rows || row_end >= _rows || col_start >= _cols || col_end >= _cols) {
            throw std::invalid_argument("Row or column index out of bounds.");
        }
        size_t new_rows = row_end - row_start + 1, new_cols = col_end - col_start + 1;
        Matrix<T> out(new_rows, new_cols);
        for (size_t i = row_start, i2 = 0; i <= row_end; i++, i2++) {
            for (size_t j = col_start, j2 = 0; j <= col_end; j++, j2++) {
                out._data[i2*new_cols+j2] = _data[i*_cols+j];
            }
        }
        return out;
    }

    inline T& operator()(size_t i, size_t j) { return _data[i*_cols+j]; }
    inline const T& operator()(size_t i, size_t j) const { return _data[i*_cols+j]; }
    inline T& operator()(size_t index) { return _data[index]; }
    inline const T& operator()(size_t index) const { return _data[index]; }
    inline T& operator[](size_t index) { return _data[index]; }
    inline const T& operator[](size_t index) const { return _data[index]; }

    /**
     * Check if the two matrices have the same rows and columns and that all values
     * in two matrices are exactly equal. Note that due to floating-point math
     * inaccuracies, this will frequently be false even when performing the same
     * exact math. Use allclose() instead.
     */
    inline bool operator==(const Matrix<T>& M) const {
        return _rows == M._rows && _cols == M._cols &&
            memcmp(_data, M._data, _size*sizeof(T)) == 0;
    }

    /**
     * Check if all values in two matrices are within a relative and absolute
     * tolerance of each other. This returns true if, for each pair of elements in
     * A and B, abs(a-b) <= (atol+rtol*abs(b)) is true. Good values for rtol and
     * atol are 1e-05 and 1e-08. If any value in either matrix is nan, this will
     * always compare as false.
     */
    bool allclose(const Matrix<T>& M, double rtol = 1e-05, double atol = 1e-08) const {
        if (_rows != M._rows || _cols != M._cols) { return false; }
        for (size_t i = 0; i < _size; i++) {
            double diff = std::fabs(_data[i] - M._data[i]);
            if (diff > atol + rtol * std::fabs(M._data[i])) {
                return false;
            }
        }
        return true;
    }

    typedef T (*unary_func)(T);
    typedef T (*binary_func)(T, T);

    /**
     * Apply a unary function to every element of a matrix replacing the value with
     * the return value of the function. This operates in-place.
     * 
     * For example, the unary function could be one of the built-in functions like
     * sin, cos, fabs, exp, log, sqrt, floor, or ceil. It will also work with any
     * custom functions that have a single double parameter and return a double.
     */
    void apply(unary_func func) { for (size_t i = 0; i < _size; i++) { _data[i] = func(_data[i]); } }

    /**
     * Apply a binary function to every pair of elements from two matrices,
     * replacing the value in the first matrix with the return value of the
     * function. This operates in-place.
     * 
     * This returns false if the two matrices are not the same shape.
     * 
     * For example, the binary function could be one of the built-in functions like
     * pow, fmod, fmax, fmin, hypot. It will also work with any custom functions that
     * have two double paramters and return a double value.
     */
    bool apply(const Matrix<T>& M, binary_func func) {
        if (_rows != M._rows || _cols != M._cols) { return false; }
        for (size_t i = 0; i < _size; i++) { _data[i] = func(_data[i], M._data[i]); }
        return true;
    }

    /**
     * Apply a binary function to every element in the matrix with the scalar
     * (i.e. fixed value) as the second argument, replacing the value in the matrix
     * with the return value of the function. This operates in-place.
     * 
     * For example, the binary function could be one of the built-in functions like
     * pow, fmod, fmax, fmin, hypot. It will also work with any custom functions that
     * have two double paramters and return a double value.
     */
    void apply(const T value, binary_func func) {
        for (size_t i = 0; i < _size; i++) { _data[i] = func(_data[i], value); }
    }

    /**
     * Apply a binary function to every element in the matrix with the scalar
     * (i.e. fixed value) as the first argument, replacing the value in the matrix
     * with the return value of the function. This operates in-place.
     * 
     * For example, the binary function could be one of the built-in functions like
     * pow, fmod, fmax, fmin, hypot. It will also work with any custom functions that
     * have two double paramters and return a double value.
     */
    void apply_alt(const T value, binary_func func) {
        for (size_t i = 0; i < _size; i++) { _data[i] = func(value, _data[i]); }
    }

    /**
     * Apply a unary function to every element of a matrix and save the value to a
     * new matrix.
     * 
     * For example, the unary function could be one of the built-in functions like
     * sin, cos, fabs, exp, log, sqrt, floor, or ceil. It will also work with any
     * custom functions that have a single double parameter and return a double.
     */
    Matrix<T> map(unary_func func) const {
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = func(_data[i]); }
        return out;
    }

    /**
     * Apply a binary function to every pair of elements from two matrices, and
     * save the return value to a new matrix.
     */
    Matrix<T> map(const Matrix<T>& M, binary_func func) const {
        if (_rows != M._rows || _cols != M._cols) { throw std::invalid_argument("Matrix dimensions do not match"); }
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = func(_data[i], M._data[i]); }
        return out;
    }

    /**
     * Apply a binary function to every pair of elements from a matrix and scalar,
     * and save the return value to a new matrix.
     */
    Matrix<T> map(const T& value, binary_func func) const {
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = func(_data[i], value); }
        return out;
    }

    /**
     * Apply a binary function to every element in the matrix with the scalar
     * (i.e. fixed value) as the first argument, saving the return value to a new
     * matrix.
     */
    Matrix<T> map_alt(const T value, binary_func func) {
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = func(value, _data[i]); }
        return out;
    }

    /**
     * Apply a binary function to every elements and the previous return value,
     * reducing down to a single value. Useful for things like fmax().
     */
    T reduce(binary_func func) {
        T val = _data[0];
        for (size_t i = 1; i < _size; i++) { val = func(val, _data[i]); }
        return val;
    }


    ////////// Operator overloads //////////
    Matrix<T> operator +() const {
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = +_data[i]; }
        return out;
    }
    Matrix<T> operator -() const {
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = -_data[i]; }
        return out;
    }

    Matrix<T> operator +(const Matrix<T>& M) const {
        if (_rows != M._rows || _cols != M._cols) { throw std::invalid_argument("Matrix dimensions do not match"); }
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = _data[i] + M._data[i]; }
        return out;
    }
    Matrix<T> operator -(const Matrix<T>& M) const {
        if (_rows != M._rows || _cols != M._cols) { throw std::invalid_argument("Matrix dimensions do not match"); }
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = _data[i] - M._data[i]; }
        return out;
    }
    Matrix<T> operator *(const Matrix<T>& M) const {
        if (_rows != M._rows || _cols != M._cols) { throw std::invalid_argument("Matrix dimensions do not match"); }
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = _data[i] * M._data[i]; }
        return out;
    }
    Matrix<T> operator /(const Matrix<T>& M) const {
        if (_rows != M._rows || _cols != M._cols) { throw std::invalid_argument("Matrix dimensions do not match"); }
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = _data[i] / M._data[i]; }
        return out;
    }

    Matrix<T> operator +(const T value) const {
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = _data[i] + value; }
        return out;
    }
    Matrix<T> operator -(const T value) const {
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = _data[i] - value; }
        return out;
    }
    Matrix<T> operator *(const T value) const {
        Matrix<T> out(_rows, _cols);
        for (size_t i = 0; i < _size; i++) { out._data[i] = _data[i] * value; }
        return out;
    }
    Matrix<T> operator /(const T value) const {
        Matrix<T> out(_rows, _cols);
        T reciprocal = 1 / value;
        for (size_t i = 0; i < _size; i++) { out._data[i] = _data[i] * reciprocal; }
        return out;
    }

    // Compound assignment operators
    Matrix<T> operator +=(const Matrix<T>& M) {
        if (_rows != M._rows || _cols != M._cols) { throw std::invalid_argument("Matrix dimensions do not match"); }
        for (size_t i = 0; i < _size; i++) { _data[i] += M._data[i]; }
        return *this;
    }
    Matrix<T> operator -=(const Matrix<T>& M) {
        if (_rows != M._rows || _cols != M._cols) { throw std::invalid_argument("Matrix dimensions do not match"); }
        for (size_t i = 0; i < _size; i++) { _data[i] -= M._data[i]; }
        return *this;
    }
    Matrix<T> operator *=(const Matrix<T>& M) {
        if (_rows != M._rows || _cols != M._cols) { throw std::invalid_argument("Matrix dimensions do not match"); }
        for (size_t i = 0; i < _size; i++) { _data[i] *= M._data[i]; }
        return *this;
    }
    Matrix<T> operator /=(const Matrix<T>& M) {
        if (_rows != M._rows || _cols != M._cols) { throw std::invalid_argument("Matrix dimensions do not match"); }
        for (size_t i = 0; i < _size; i++) { _data[i] /= M._data[i]; }
        return *this;
    }

    Matrix<T> operator +=(const T value) {
        for (size_t i = 0; i < _size; i++) { _data[i] += value; }
        return *this;
    }
    Matrix<T> operator -=(const T value) { // TODO: how to do this in reverse?
        for (size_t i = 0; i < _size; i++) { _data[i] -= value; }
        return *this;
    }
    Matrix<T> operator *=(const T value) {
        for (size_t i = 0; i < _size; i++) { _data[i] *= value; }
        return *this;
    }
    Matrix<T> operator /=(const T value) { // TODO: how to do this in reverse?
        T reciprocal = 1 / value;
        for (size_t i = 0; i < _size; i++) { _data[i] *= reciprocal; }
        return *this;
    }
};

template<typename T>
Matrix<T> operator +(const T value, const Matrix<T>& M) {
    Matrix<T> out(M.rows, M.cols);
    for (size_t i = 0; i < M.size; i++) { out[i] = value + M[i]; }
    return out;
}
template<typename T>
Matrix<T> operator -(const T value, const Matrix<T>& M) {
    Matrix<T> out(M.rows, M.cols);
    for (size_t i = 0; i < M.size; i++) { out[i] = value - M[i]; }
    return out;
}
template<typename T>
Matrix<T> operator *(const T value, const Matrix<T>& M) {
    Matrix<T> out(M.rows, M.cols);
    for (size_t i = 0; i < M.size; i++) { out[i] = value * M[i]; }
    return out;
}
template<typename T>
Matrix<T> operator /(const T value, const Matrix<T>& M) {
    Matrix<T> out(M.rows, M.cols);
    for (size_t i = 0; i < M.size; i++) { out[i] = value / M[i]; }
    return out;
}


//////////////////// Matrix Multiplication ////////////////////
/**
 * Original basic version of matrix multiplication. This is intollerably slow for 2048x2048.
 */
template<typename T>
void matrix_multiplication_orig(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const size_t m = A.rows, n = A.cols, p = B.cols;
    if (n != B.rows || m != C.rows || p != C.cols) { throw std::invalid_argument("Matrix dimensions do not match for multiplication."); }

    const T* A_data = A.data;
    const T* B_data = B.data;
    T* C_data = C.data;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < p; j++) {
            T value = 0;
            for (size_t k = 0; k < n; k++) { value += A_data[i*n+k] * B_data[k*p+j]; }
            C_data[i*p+j] = value;
        }
    }
}

/**
 * Blocked version of matrix multiplication. This is much faster than the basic version.
 * It also has the advantage of being more cache-friendly.
 */
template<typename T>
void matrix_multiplication_block(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C) {
    const size_t m = A.rows, n = A.cols, p = B.cols;
    if (n != B.rows || m != C.rows || p != C.cols) { throw std::invalid_argument("Matrix dimensions do not match for multiplication."); }
    C.fill_zeros();

    static const int block_size = 32;
    const T* __restrict__ A_data = A.data;
    const T* __restrict__ B_data = B.data;
    T* __restrict__ C_data = C.data;

    for (size_t i0 = 0; i0 < m; i0 += block_size) {
        const size_t i_blk_size = std::min(i0+block_size, m) - i0;
        for (size_t j0 = 0; j0 < p; j0 += block_size) {
            const size_t j_blk_size = std::min(j0+block_size, p) - j0;
            for (size_t k0 = 0; k0 < n; k0 += block_size) {
                const size_t k_blk_size = std::min(k0+block_size, n) - k0;

                // Get pointers to just the blocks at i0, j0, k0
                const T * __restrict__ A_block = &A_data[i0*n+k0];
                const T * __restrict__ B_block = &B_data[k0*p+j0];
                T * __restrict__ C_block = &C_data[i0*p+j0];

                // Working on block at i0, j0, k0
                for (size_t i = 0; i < i_blk_size; i++) {
                    const T* A_row = &A_block[i*n]; 
                    T* __restrict__ C_out = &C_block[i*p];
                    for (size_t k = 0; k < k_blk_size; k++) {
                        T Aik = *A_row++;
                        for (size_t j = 0; j < j_blk_size; j++) { // j loop on the inside is fastest
                            C_out[j] += Aik * B_block[j+k*p];
                        }
                    }
                }
            }
        }
    }
}



//////////////////// IO Helpers ////////////////////
template <typename T>
static inline T __read_csv_val(const std::string& token) {
    T val;
    std::stringstream ss(token);
    if (!ss >> val || !ss.eof()) {
        std::cerr << "Not a number in CSV file, using 0.0:" << token << std::endl;
        return 0.0;
    }
    return val;
}

template <typename T>
static inline size_t __read_csv_first_line(std::string& line, T** out) {
    size_t pos = 0, start = 0;
    size_t i = 0, size = 16;
    T* vals = (T*)malloc(16*sizeof(T));
    while ((pos = line.find(',', start)) != std::string::npos) {
        std::string token = line.substr(start, pos-start);
        vals[i] = __read_csv_val<T>(token);
        if (++i == size) {
            size += size > 512 ? 512 : size;
            vals = (T*)realloc(vals, size*sizeof(T));
        }
        start = pos + 1;
    }
    *out = vals;
    return i;
}

template <typename T>
static inline void __read_csv_line(std::string& line, T* out, size_t count) {
    size_t pos = 0, start = 0;
    size_t i = 0;
    while ((pos = line.find(',', start)) != std::string::npos) {
        std::string token = line.substr(start, pos-start);
        out[i] = __read_csv_val<T>(token);
        if (++i == count) { return; } // stop reading
        start = pos + 1;
    }
    for (size_t j = i; j < count; j++) { out[j] = (T)0; } // zero-fill remainder
}

static inline size_t __py_dict_value(const std::string& dict, const std::string& key) {
    size_t pos = dict.find(key);
    if (pos == std::string::npos || pos == 0 || dict[pos-1] != dict[pos+key.length()] ||
        (dict[pos-1] != '\'' && dict[pos-1] != '"')) { return std::string::npos; }
    pos += key.length() + 1;
    while (isspace(dict[pos])) { pos++; }
    if (dict[pos] != ':') { return std::string::npos; }
    pos++;
    while (isspace(dict[pos])) { pos++; }
    return pos;
}

static inline std::string __py_dict_value_str(const std::string& dict, const std::string& key, const std::string& default_val) {
    size_t pos = __py_dict_value(dict, key), end;
    if (pos == std::string::npos) { return default_val; }
    char c = dict[pos];
    if ((c != '\'' && c != '"') || ((end = dict.find(c, ++pos)) == std::string::npos)) { return default_val; }
    return dict.substr(pos, end-pos);
}

static inline bool __py_dict_value_bool(const std::string& dict, const std::string& key, const bool default_val) {
    size_t pos = __py_dict_value(dict, key);
    if (pos == std::string::npos) { return default_val; }
    if (dict.compare(pos, 5, "False") == 0) { return false; }
    if (dict.compare(pos, 4, "True") == 0) { return true; }
    return default_val;
}

static inline bool __py_dict_value_tuple(const std::string& dict, const std::string& key, size_t* val) {
    size_t pos = __py_dict_value(dict, key);
    if (pos == std::string::npos || dict[pos++] != '(') { return false; }
    val[0] = val[1] = 1;
    std::string values = dict.substr(pos, dict.find(')', pos)-pos+1);
    char* s = (char*)values.c_str();
    char c = 0;
    return ((sscanf(s, " %c", &c) == 1 && c == ')') ||
        (sscanf(s, " %zu %c", &val[0], &c) == 2 && c == ')') ||
        (sscanf(s, " %zu , %c", &val[0], &c) == 2 && c == ')') ||
        (sscanf(s, " %zu , %zu %c", &val[0], &val[1], &c) == 3 && c == ')') ||
        (sscanf(s, " %zu , %zu , %c", &val[0], &val[1], &c) == 3 && c == ')'));
}

template <typename T>
static inline size_t __npy_read_header(std::istream& file, size_t* shape) {
    char header[10];
    if (!file.read(header, sizeof(header))) { throw std::runtime_error("Failed to read NPY header"); }
    if (memcmp(header, "\x93NUMPY", 6) != 0) { throw std::runtime_error("Invalid NPY header"); }
    // header[6] is major file version
    // header[7] is minor file version
    int len = *(unsigned short*)(header+8); // assumes running on little-endian

    // Read the rest of the header (a Python dict)
    char* dict = (char*)malloc(len+1);
    if (!file.read(dict, len) || dict[0] != '{') { free(dict); throw std::runtime_error("Failed to read NPY header dict."); }
    dict[len] = 0;

    // only allowed descr: 'float64', 'f8', or '<f8'  // TODO: base on type T
    std::string descr = __py_dict_value_str(dict, "descr", "");
    if (descr != "float64" && descr != "f8" && descr != "<f8") { free(dict); throw std::invalid_argument("Invalid data type in NPY."); }

    // only allowed fortran_order is false
    if (__py_dict_value_bool(dict, "fortran_order", false)) { free(dict); throw std::invalid_argument("Fortran order not allowed in NPY."); }

    // only allowed to be 0d, 1d, or 2d, but this is checked elsewhere
    if (!__py_dict_value_tuple(dict, "shape", shape) || shape[0] < 1 || shape[1] < 1) { free(dict); throw std::invalid_argument("Invalid shape in NPY."); }
    free(dict);

    return sizeof(header) + len;
}
