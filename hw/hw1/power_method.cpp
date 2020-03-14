#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <random>
#include <array>
#include <tuple>

using namespace std;

// FORTRAN adds _ after all the function names
// and all variables are called by reference
extern "C"{
    double dsymv_(
        const char * UPLO, const int * N, 
        const double * ALPHA, double * A,
        const int * LDA, double * X, const int * INCX,
        const double * BETA, double * Y, const int * INCY
    );

    double dnrm2_(
        const int * N, double * X, const int * INCX
    );

    double daxpy_(
        const int * N, double DA, double * DX, const int * INCX,
        double * DY, const int * INCY
    );

    double ddot_(
        const int * N, double * DX, const int * INCX, double * DY, const int * INCY
    );

    double dscal_(
        const int * N, double * DA, double * DX, const int * INCX
    );

    double dswap_(
        const int * N, double * DX, const int * INCX, double * DY, const int * INCY
    );
}

void read_matrix(ifstream& file, int n, double * A) {
  for (int i = 0; i < n; i++ ) {
    for (int j = 0; j < n; j++) {
      // Write in column major order
      file >> A[j*n+i];
    }
  }
}



auto power_method_dsym(int matrix_size, double * matrix, int max_iter=1000){
    
    int k = 1;
    float lambda_old = 0;
    float lambda = 1;
    float eps = 1e-6;
    random_device rd;
    mt19937 gen(rd());
    std::uniform_real_distribution<double> unif(-1., 1.);
    
    auto x = new double[matrix_size];
    auto y = new double[matrix_size];

    for(int i = 0; i < matrix_size; i++){
        x[i] = unif(gen);
    
    for(int i = 0; i < matrix_size; i++){
        y[i] = unif(gen);
    }}

    const char UPLO = 'u';
    const int INCX = 1;
    const int INCY = 1;
    const int N = matrix_size;
    const double ALPHA = 1;
    const double BETA = 0;
    double euc_norm = 0;
    
    while (abs(lambda - lambda_old) > eps && k < max_iter){
        
        lambda_old = lambda;

        dsymv_(
            &UPLO, &N, &ALPHA, matrix, &N, x, &INCX, &BETA, y, &INCY
        );

        lambda = ddot_(&N, y, &INCX, x, &INCY) / dnrm2_(&N, x, &INCX);

        euc_norm = 1 / dnrm2_(&N, y, &INCX);

        dscal_(&N, &euc_norm, y, &INCX);

        dswap_(&N, x, &INCX, y, &INCY);
        
        k++;

    }

    return make_tuple(lambda, x);
}

int main( int argc, char** argv ){
    
  if (argc != 2) {
    printf("usage: power_method matrix\n");
    printf("  matrix should be text file of form:\n");
    printf("    n\n");
    printf("    n lines of n space-delimited numbers\n");

    return 0;
  }

  // Get matrix
  ifstream mat_file(argv[1]);
  if (!mat_file) {
    printf("Could not open file.");
    return 1;
  }
  int n;
  mat_file >> n;
  
  auto A = new double[n*n];

  read_matrix(mat_file, n, A);

  double lambda; 
  double * x;
  
  tie(lambda, x) = power_method_dsym(n, A);

  // Print results
  printf("Eigenvalue : %f\n", lambda);
  printf("Eigenvector:\n");
  for (int i = 0; i < n; i++)
    printf("%f\n", x[i]);;

  return 0;
};
