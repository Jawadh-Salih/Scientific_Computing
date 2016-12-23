
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>
// #include <math.h>

float ** generate_matrix_sp(float **M, unsigned long N){
  M = (float **)malloc(sizeof(float *)*N);
  for(long i=0; i < N; i++) M[i] = malloc(sizeof(float) * N);
  for(long i=0;i<N;i++){
    for(long j=0;j<N;j++){
      M[i][j] =(float)rand()/(float)RAND_MAX +1;
      // printf("%f ",M[i][j]);
    }
    // printf("\n" );

    //printf("%f\n",V[i] );
  }
  return M;
}
double ** generate_matrix_dp(double **M, unsigned long N){
  M = (double **)malloc(sizeof(double *)*N);
  for(long i=0; i < N; i++) M[i] = malloc(sizeof(double) * N);

  for(long i=0;i<N;i++){
    for(long j=0;j<N;j++){
      M[i][j] = (double)rand()/(double)RAND_MAX +1;
      // printf("%lf ",M[i][j]);
    }
  }
    // printf("\n" );  }
  return M;
}


float** matrix_mul_sp(float **M1,float **M2, unsigned long N){
  float ** C = (float **)malloc(sizeof(float *)*N);
  for(long i=0; i < N; i++) C[i] = malloc(sizeof(float) * N);
  for(long i=0;i<N;i++){
    for(long j=0;j<N;j++){
        for(long k=0;k<N;k++){
          C[i][j] += M1[i][k]*M2[k][j];
        }
    }
  }
  // for(long i=0;i<N;i++){
  //   for(long j=0;j<N;j++){
  //     printf("%f ",C[i][j]);
  //   }
  //   printf("\n" );
  // }

  return C;
}
double ** matrix_mul_dp(double **V1,double **V2, unsigned long N){
  double ** C = (double **)malloc(sizeof(double *)*N);
  for(long i=0; i < N; i++) C[i] = malloc(sizeof(double) * N);

  for(long i=0;i<N;i++){
    for(long j=0;j<N;j++){
        for(long k=0;k<N;k++){
          C[i][j] += V1[i][k]*V2[k][j];
        }
    }
  }
  for(long i=0;i<N;i++){
    for(long j=0;j<N;j++){
      printf("%lf ",C[i][j]);
    }
    printf("\n" );
  }
  return C;
}

int main(){

  float **Asp,**Bsp; // Single precision
  double **Adp,**Bdp; // double precision
  unsigned long N=100; // Dimension of a vector.
  int num_threads = 8;

  double time_serial=0.0;
  double time_parallel=0.0;
  double time_cuda=0.0;
  printf("========================================================================\n");
  printf("====================Matrix Multiplication calculation===================\n");
  printf("========================================================================\n");

  printf("==============================Single Precision==========================\n");

  // randome seed as a Pseudo Random Number generator.
  unsigned seed = (time(NULL));
  srand(seed);
  clock_t start, stop;

  Asp = generate_matrix_sp(Asp,N);
  Bsp = generate_matrix_sp(Bsp,N);

  //Start of CPU serial computation
  start  = clock();

  float ** C_sp = matrix_mul_sp(Asp,Bsp,N);

  stop = clock();
  //End of CPU Sserial computation.
  time_serial = 1000.0 * (stop-start)/(double)CLOCKS_PER_SEC;

  float ** C_sp_par = (float **)malloc(sizeof(float *)*N);
  for(long i=0; i < N; i++) C_sp_par[i] = malloc(sizeof(float) * N);

  //Start of CPU parallel computation in OMP.
  start= clock();

  float dot_product_sp=0.0;
  #pragma omp parallel num_threads(num_threads)private(i,j,k) shared(C_sp_par,Asp,Bsp)
  {
    #pragma omp for schedule(static) reduction(+:dot_product_sp)
    for(long i=0;i<N;i++){
      for(long j=0;j<N;j++){
          dot_product_sp = 0;
          for(long k=0;k<N;k++){
            dot_product_sp += Asp[i][k]*Bsp[k][j];
          }
          C_sp_par[i][j] = dot_product_sp;
      }
    }
  }
  for(long i=0;i<N;i++){
    for(long j=0;j<N;j++){
      printf("%lf ",C_sp_par[i][j]);
    }
    printf("\n" );
  }
  stop = clock();
  // End of CPU parallel Computation
  time_parallel = 1000.0* (stop-start)/(double)CLOCKS_PER_SEC;

  //Start of GPU computation.

  //End of GPU computation.

  printf("Time taken to execute Serial program : %g ms \n", time_serial );
  printf("Time taken to execute parallel program : %g ms\n", 1000* (stop_cpu_par-start_cpu_par)/(float)CLOCKS_PER_SEC);

  Adp = generate_matrix_dp(Adp,N);
  Bdp = generate_matrix_dp(Bdp,N);
  start_cpu_serial = clock();
  double **C_dp =matrix_mul_dp(Adp,Bdp,N);
  stop_cpu_serial = clock();
  double ** C_dp_par = (double **)malloc(sizeof(double *)*N);
  for(long i=0; i < N; i++) C_dp_par[i] = malloc(sizeof(double) * N);
  // float dotP_sp1 = dotP_dp;


  // printf("%.20f\n",dotP_dp );
  printf("Double Precision\n");
  // printf("CPU serial dot product is %lf\n",dotP_dp );
  printf("Time taken to execute Serial program : %f ms \n", 1000*(stop_cpu_serial-start_cpu_serial)/(float)CLOCKS_PER_SEC );
  printf("Time taken to execute parallel program : %f ms\n", 1000* (stop_cpu_par-start_cpu_par)/(float)CLOCKS_PER_SEC);

  //parallel computation in OMP

  double dot_product_dp = 0.0;

  start_cpu_par = clock();
  int i=0,j=0,k=0;
  #pragma omp parallel num_threads(num_threads)private(i,j,k) shared(C_dp_par,Adp,Bdp)
  {
    #pragma omp for schedule(static) reduction(+:dot_product_dp)
    for(long i=0;i<N;i++){
      for(long j=0;j<N;j++){
          dot_product_dp = 0.0;
          for(long k=0;k<N;k++){
            dot_product_dp += Adp[i][k]*Bdp[k][j];
          }
          C_dp_par[i][j] = dot_product_dp;
      }
    }
  stop_cpu_par = clock();

  // GPU computation in CUDA
  printf("Time taken to execute Parallel program  : %f ms \n", 1000*(stop_cpu_par-start_cpu_par)/(float)CLOCKS_PER_SEC );

  }

    //  free(&V1sp);free(&V2sp);free(&V1dp);free(&V2dp);


  // V1dp = generate_vector_dp(V1dp,N);
  // V2dp = generate_vector_dp(V2dp,N);
  //


}
