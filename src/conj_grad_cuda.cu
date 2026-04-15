/*
    nvcc -arch compute_60 -code sm_60 -O1 -Xcompiler -Wall conj_grad_cuda.cu -o conj_grad_cuda
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
            cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define TOL 1e-6

/* ================= GPU KERNELS ================= */

// Dot product: result = <a, b>
__global__ void dot_product_kernel(int n, double* a, double* b, double* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(result, a[i] * b[i]);
    }
}

// Vector copy: y = x
__global__ void vec_copy_kernel(int n, double* x, double* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i];
    }
}

// Matrix-vector multiplication: result = A * x
__global__ void mat_vec_mul_kernel(int n, double* A, double* x, double* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i*n + j] * x[j];
        }
        result[i] = sum;
    }
}

// Vector multiply and add: z = a * x + y
__global__ void vec_mul_add_kernel(int n, double a, double* x, double* y, double* z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = a * x[i] + y[i];
    }
}

/* ================= CPU Reference ================= */

// Helper to compute dot product of two vectors
double dot_product(int n, double *a, double *b) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

// Conjugate Gradient CPU reference
void conj_grad_cpu(int n, double* A, double* b, double* x) {
    double r[n], p[n], Ap[n];
    double rsold, rsnew, alpha, beta;

    // Initial r = b - Ax (assuming initial x is zeros)
    for (int i = 0; i < n; i++) {
        // Need to use following if x != 0:
        // double Ax0 = 0;
        // for (int j = 0; j < n; j++) Ax0 += A[i][j] * x[j];
        // r[i] = b[i] - Ax0;
        r[i] = b[i];
        p[i] = r[i]; // Initial search direction is the residual
    }

    rsold = dot_product(n, r, r);

    for (int i = 0; i < n; i++) {
        // Compute Ap = A * p
        for (int row = 0; row < n; row++) {
            Ap[row] = 0;
            for (int col = 0; col < n; col++) Ap[row] += A[row][col] * p[col];
        }

        // alpha = rsold / (p' * A * p)
        alpha = rsold / dot_product(n, p, Ap);

        // Update x and r
        for (int j = 0; j < n; j++) {
            x[j] = x[j] + alpha * p[j];
            r[j] = r[j] - alpha * Ap[j];
        }

        rsnew = dot_product(n, r, r);
        if (sqrt(rsnew) < TOL) break; // Check convergence

        // beta = rsnew / rsold
        beta = rsnew / rsold;

        // Update search direction: p = r + beta * p
        for (int j = 0; j < n; j++) p[j] = r[j] + beta * p[j];

        rsold = rsnew;
    }
}

/* ================= CG WRAPPER ================= */

// Conjugate gradient GPU wrapper function
void conj_grad_gpu(int n, double *h_A, double *h_b, double *h_x)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed;

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Allocate device memory
    double *Ad, *bd, *xd, *rd, *pd, *Apd, *temp_result;
    CUDA_SAFE_CALL(cudaMalloc((void**)&Ad, n * n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&bd, n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&xd, n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&rd, n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&pd, n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Apd, n * sizeof(double)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&temp_result, sizeof(double)));

    // Record start event (end-to-end)
    cudaEventRecord(start);

    // Copy data to device
    CUDA_SAFE_CALL(cudaMemcpy(Ad, h_A, n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(bd, h_b, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(xd, h_x, n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Initial r = b - Ax (assuming initial x is zeros)
    // r = b
    vec_copy_kernel<<<gridSize, blockSize>>>(n, bd, rd);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    // p = r (initial search direction)
    vec_copy_kernel<<<gridSize, blockSize>>>(n, rd, pd);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    // rsold = r · r
    CUDA_SAFE_CALL(cudaMemset(temp_result, 0, sizeof(double)));
    dot_product_kernel<<<gridSize, blockSize>>>(n, rd, rd, temp_result);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    double rsold;
    CUDA_SAFE_CALL(cudaMemcpy(&rsold, temp_result, sizeof(double), cudaMemcpyDeviceToHost));
    
    // CG iterations
    for (int iter = 0; iter < n; iter++) {
        // Compute Ap = A * p
        mat_vec_mul_kernel<<<gridSize, blockSize>>>(n, Ad, pd, Apd);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        // alpha = rsold / (p · Ap)
        CUDA_SAFE_CALL(cudaMemset(temp_result, 0, sizeof(double)));
        dot_product_kernel<<<gridSize, blockSize>>>(n, pd, Apd, temp_result);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        double pAp;
        CUDA_SAFE_CALL(cudaMemcpy(&pAp, temp_result, sizeof(double), cudaMemcpyDeviceToHost));
        
        double alpha = rsold / pAp;
        
        // x = x + alpha * p
        vec_mul_add_kernel<<<gridSize, blockSize>>>(n, alpha, pd, xd, xd);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        // r = r - alpha * Ap
        vec_mul_add_kernel<<<gridSize, blockSize>>>(n, -alpha, Apd, rd, rd);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        // rsnew = r · r
        CUDA_SAFE_CALL(cudaMemset(temp_result, 0, sizeof(double)));
        dot_product_kernel<<<gridSize, blockSize>>>(n, rd, rd, temp_result);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        double rsnew;
        CUDA_SAFE_CALL(cudaMemcpy(&rsnew, temp_result, sizeof(double), cudaMemcpyDeviceToHost));
        
        // Check convergence
        if (sqrt(rsnew) < TOL) break;
        
        // beta = rsnew / rsold
        double beta = rsnew / rsold;
        
        // p = r + beta * p
        vec_mul_add_kernel<<<gridSize, blockSize>>>(n, beta, pd, rd, pd);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        rsold = rsnew;
    }

    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(h_x, xd, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Record stop event and synchronize
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    cudaEventElapsedTime(&elapsed, start, stop);

    printf("End-to-end time: %f ms\n", elapsed);
    
    // Free device memory
    CUDA_SAFE_CALL(cudaFree(Ad));
    CUDA_SAFE_CALL(cudaFree(bd));
    CUDA_SAFE_CALL(cudaFree(xd));
    CUDA_SAFE_CALL(cudaFree(rd));
    CUDA_SAFE_CALL(cudaFree(pd));
    CUDA_SAFE_CALL(cudaFree(Apd));
    CUDA_SAFE_CALL(cudaFree(temp_result));
}

/* ================= MAIN ================= */

int main()
{
    CUDA_SAFE_CALL(cudaSetDevice(0));

    int sizes[2] = {1024, 2048};

    for (int s = 0; s < 2; s++) {
        
        int width = sizes[s];
        size_t sizeMat = width * width * sizeof(double);
        size_t sizeVec = width * sizeof(double);

        printf("\n===== Testing %dx%d =====\n", width, width);

        // Allocate host memory
        double *h_A = (double*) malloc(sizeMat);
        double *h_b = (double*) malloc(sizeVec);
        double *h_x = (double*) malloc(sizeVec);
        double *h_gold = (double*) malloc(sizeVec);

        // Initialize host matrix (symmetric, positive-definite, real)
        srand(1234);
        for (int i = 0; i < width*width; i++) {
            h_A[i] = (double)rand()/RAND_MAX * 10.0;
        }
        // Symmetrize in-place (upper triangle only)
        for (int i = 0; i < width; i++) {
            for (int j = i + 1; j < width; j++) {
                double sym = 0.5 * (h_A[i * width + j] + h_A[j * width + i]);
                h_A[i * width + j] = sym;
                h_A[j * width + i] = sym;
            }
        }
        // Add width * I to ensure positive-definiteness
        for (int i = 0; i < width; i++) {
            h_A[i * width + i] += width;
        }

        // Initialize host vectors
        for (int i = 0; i < width; i++) {
            h_b[i] = (double)rand()/RAND_MAX * 10.0;
            h_x[i] = 0.0; // Initial guess
            h_gold[i] = 0.0; // To store CPU result
        }

        // CPU reference
        conj_grad_cpu(width, h_A, h_b, h_x);

        // GPU computation
        conj_grad_gpu(width, h_A, h_b, h_x);

        // Verify correctness
        double max_err = 0.0;
        for (int i = 0; i < width; i++) {
            double diff = fabs(h_x[i] - h_gold[i]);
            if (diff > max_err) max_err = diff;
        }
        printf("Max error: %.10f\n", max_err);

        // Free host memory
        free(h_A);
        free(h_b);
        free(h_x);
        free(h_gold);
    }

    return 0;
}