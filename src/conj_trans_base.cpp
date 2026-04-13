#include <stdio.h>
#include <complex.h>

// Function to perform conjugate transpose
void conjugateTranspose(int rows, int cols, double complex A[rows][cols], double complex B[cols][rows]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Conj takes the complex conjugate: a+bi -> a-bi
            B[j][i] = conj(A[i][j]); 
        }
    }
}

int main() {
    int rows = 2, cols = 3;
    // Example: 2x3 Matrix
    double complex A[2][3] = {
        {1.0 + 1.0*I, 2.0 + 2.0*I, 3.0 + 3.0*I},
        {4.0 + 4.0*I, 5.0 + 5.0*I, 6.0 + 6.0*I}
    };
    double complex B[3][2]; // Transposed: 3x2

    conjugateTranspose(rows, cols, A, B);

    // Print result
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            printf("%.1f%+.1fi ", creal(B[i][j]), cimag(B[i][j]));
        }
        printf("\n");
    }
    return 0;
}
