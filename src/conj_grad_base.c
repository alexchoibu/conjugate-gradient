#include <stdio.h>
#include <math.h>

// Helper to compute dot product of two vectors
double dot_product(int n, double *a, double *b) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

// Naive Conjugate Gradient Solver
void conjugate_gradient(int n, double A[n][n], double *b, double *x, double tol) {
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
        if (sqrt(rsnew) < tol) break; // Check convergence

        // beta = rsnew / rsold
        beta = rsnew / rsold;

        // Update search direction: p = r + beta * p
        for (int j = 0; j < n; j++) p[j] = r[j] + beta * p[j];

        rsold = rsnew;
    }
}

// Example usage
int main() {
    int n = 2;
    double A[2][2] = {{4, 1}, {1, 3}};
    double b[2] = {1, 2};
    double x[2] = {0}; // Initial guess
    double tol = 1e-6;

    conjugate_gradient(n, A, b, x, tol);

    printf("Solution: ");
    for (int i = 0; i < n; i++) printf("%f ", x[i]);
    printf("\n");

    return 0;
}