/*
    nvcc -arch compute_60 -code sm_60 -O1 -Xcompiler -Wall conj_grad_cuda.cu -o conj_grad_cuda
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
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
typedef double data_t;

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:
 
        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

/* ================= GPU KERNELS ================= */

// Dot product: result = <a, b>
__global__ void dot_product_kernel(int n, data_t* a, data_t* b, data_t* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(result, a[i] * b[i]);
    }
}

// Vector copy: y = x
__global__ void vec_copy_kernel(int n, data_t* x, data_t* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i];
    }
}

// Matrix-vector multiplication: result = A * x
__global__ void mat_vec_mul_kernel(int n, data_t* A, data_t* x, data_t* result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data_t sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A[i*n + j] * x[j];
        }
        result[i] = sum;
    }
}

// Vector multiply and add: z = a * x + y
__global__ void vec_mul_add_kernel(int n, data_t a, data_t* x, data_t* y, data_t* z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = a * x[i] + y[i];
    }
}

/* ================= CPU Reference ================= */

// Helper to compute dot product of two vectors
data_t dot_product(int n, data_t *a, data_t *b) {
    data_t sum = 0.0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

// Conjugate Gradient CPU reference
void conj_grad_cpu(int n, data_t* A, data_t* b, data_t* x) {
    data_t r[n], p[n], Ap[n];
    data_t rsold, rsnew, alpha, beta;

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
            for (int col = 0; col < n; col++) Ap[row] += A[row * n + col] * p[col];
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
void conj_grad_gpu(int n, data_t *h_A, data_t *h_b, data_t *h_x)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed;

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Allocate device memory
    data_t *Ad, *bd, *xd, *rd, *pd, *Apd, *temp_result;
    CUDA_SAFE_CALL(cudaMalloc((void**)&Ad, n * n * sizeof(data_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&bd, n * sizeof(data_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&xd, n * sizeof(data_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&rd, n * sizeof(data_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&pd, n * sizeof(data_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&Apd, n * sizeof(data_t)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&temp_result, sizeof(data_t)));

    // Record start event (end-to-end)
    cudaEventRecord(start);

    // Copy data to device
    CUDA_SAFE_CALL(cudaMemcpy(Ad, h_A, n * n * sizeof(data_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(bd, h_b, n * sizeof(data_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(xd, h_x, n * sizeof(data_t), cudaMemcpyHostToDevice));
    
    // Initial r = b - Ax (assuming initial x is zeros)
    // r = b
    vec_copy_kernel<<<gridSize, blockSize>>>(n, bd, rd);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    // p = r (initial search direction)
    vec_copy_kernel<<<gridSize, blockSize>>>(n, rd, pd);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    // rsold = r · r
    CUDA_SAFE_CALL(cudaMemset(temp_result, 0, sizeof(data_t)));
    dot_product_kernel<<<gridSize, blockSize>>>(n, rd, rd, temp_result);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    
    data_t rsold;
    CUDA_SAFE_CALL(cudaMemcpy(&rsold, temp_result, sizeof(data_t), cudaMemcpyDeviceToHost));
    
    // CG iterations
    for (int iter = 0; iter < n; iter++) {
        // Compute Ap = A * p
        mat_vec_mul_kernel<<<gridSize, blockSize>>>(n, Ad, pd, Apd);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        // alpha = rsold / (p · Ap)
        CUDA_SAFE_CALL(cudaMemset(temp_result, 0, sizeof(data_t)));
        dot_product_kernel<<<gridSize, blockSize>>>(n, pd, Apd, temp_result);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        data_t pAp;
        CUDA_SAFE_CALL(cudaMemcpy(&pAp, temp_result, sizeof(data_t), cudaMemcpyDeviceToHost));
        
        data_t alpha = rsold / pAp;
        
        // x = x + alpha * p
        vec_mul_add_kernel<<<gridSize, blockSize>>>(n, alpha, pd, xd, xd);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        // r = r - alpha * Ap
        vec_mul_add_kernel<<<gridSize, blockSize>>>(n, -alpha, Apd, rd, rd);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        // rsnew = r · r
        CUDA_SAFE_CALL(cudaMemset(temp_result, 0, sizeof(data_t)));
        dot_product_kernel<<<gridSize, blockSize>>>(n, rd, rd, temp_result);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        data_t rsnew;
        CUDA_SAFE_CALL(cudaMemcpy(&rsnew, temp_result, sizeof(data_t), cudaMemcpyDeviceToHost));
        
        // Check convergence
        if (sqrt(rsnew) < TOL) break;
        
        // beta = rsnew / rsold
        data_t beta = rsnew / rsold;
        
        // p = r + beta * p
        vec_mul_add_kernel<<<gridSize, blockSize>>>(n, beta, pd, rd, pd);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        rsold = rsnew;
    }

    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(h_x, xd, n * sizeof(data_t), cudaMemcpyDeviceToHost));

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

    struct timespec time_start, time_stop;

    int sizes[2] = {1024, 2048};

    for (int s = 0; s < 2; s++) {
        
        int width = sizes[s];
        size_t sizeMat = width * width * sizeof(data_t);
        size_t sizeVec = width * sizeof(data_t);

        printf("\n===== Testing %dx%d =====\n", width, width);

        // Allocate host memory
        data_t *h_A = (data_t*) malloc(sizeMat);
        data_t *h_b = (data_t*) malloc(sizeVec);
        data_t *h_x = (data_t*) malloc(sizeVec);
        data_t *h_gold = (data_t*) malloc(sizeVec);

        // Initialize host matrix (symmetric, positive-definite, real)
        srand(1234);
        for (int i = 0; i < width*width; i++) {
            h_A[i] = (data_t)rand()/RAND_MAX * 10.0;
        }
        // Symmetrize in-place (upper triangle only)
        for (int i = 0; i < width; i++) {
            for (int j = i + 1; j < width; j++) {
                data_t sym = 0.5 * (h_A[i * width + j] + h_A[j * width + i]);
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
            h_b[i] = (data_t)rand()/RAND_MAX * 10.0;
            h_x[i] = 0.0; // Initial guess
            h_gold[i] = 0.0; // To store CPU result
        }

        // CPU reference
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
        conj_grad_cpu(width, h_A, h_b, h_x);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
        double timestamp = interval(time_start, time_stop);
        printf("CPU time: %f ms\n", timestamp * 1000.0);

        // GPU computation
        conj_grad_gpu(width, h_A, h_b, h_x);

        // Verify correctness
        data_t max_err = 0.0;
        for (int i = 0; i < width; i++) {
            data_t diff = fabs(h_x[i] - h_gold[i]);
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