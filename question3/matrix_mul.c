
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
// #include <cuda.h>
// #include <math.h>
#define TILE_N 10

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

// Single Precision Implementation
float** matrix_mul_sp(float **M1,float **M2, unsigned long N){

  // for(long i=0;i<N;i++){
  //   for(long j=0;j<N;j++){
  //     printf("%f ",M1[i][j]);
  //   }
  //   printf("\n" );
  // }
  //   printf("\n" );
  // for(long i=0;i<N;i++){
  //   for(long j=0;j<N;j++){
  //     printf("%f ",M2[i][j]);
  //   }
  //   printf("\n" );
  // }
  //   printf("\n" );
  float ** C = (float **)malloc(sizeof(float *)*N);
  for(long i=0; i < N; i++) C[i] = malloc(sizeof(float) * N);
  for(long i=0;i<N;i++){
    for(long j=0;j<N;j++){
        C[i][j]=0.0;
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

// Double Precision Implementation
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
  // for(long i=0;i<N;i++){
  //   for(long j=0;j<N;j++){
  //     printf("%lf ",C[i][j]);
  //   }
  //   printf("\n" );
  // }
  return C;
}

__global__ matrix_mul_sp_kernel(float *A_dev,float *B_dev,float * C_dev, unsigned long N){

  // to store the element of matrix

  int row = blockIdx.y*blockDim.y+threadIdx.y; // row index for the matrices
  int col = blockIdx.x*blockDim.x+threadIdx.x; // column index for the matrices

  if ((row < N) && (col <N)){
    float C_value = 0.0;
    for(int k=0;k<N;k++){
        C_value += A_dev[row*N+i]*B_dev[k*N+col];
        C_dev[row*N+col] = C_value;
    }

  }
}
int main(){

  float **Asp,**Bsp; // Single precision
  double **Adp,**Bdp; // double precision
  unsigned long N=1000; // Dimension of a vector.
  int threads = 8;

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
  // for(long i=0;i<N;i++){
  //   for(long j=0;j<N;j++){
  //     printf("%f ",C_sp[i][j]);
  //   }
  //   printf("\n" );
  // }
  stop = clock();
  //End of CPU Sserial computation.
  time_serial = 1000.0 * (stop-start)/(double)CLOCKS_PER_SEC;

  float ** C_sp_par = (float **)malloc(sizeof(float *)*N);
  for(long i=0; i < N; i++) C_sp_par[i] = malloc(sizeof(float) * N);

  //Start of CPU parallel computation in OMP.
  start= clock();

  float dot_product_sp=0.0;

  // #pragma omp parallel num_threads(threads) shared(C_sp_par,Asp,Bsp)
  // {
  //   #pragma omp for schedule(static) reduction(+:dot_product_sp)
  //   for(long i=0;i<N;i++){
  //     for(long j=0;j<N;j++){
  //           C_sp_par[i][j] =0.0;
  //         dot_product_sp = 0;
  //         for(long k=0;k<N;k++){
  //           dot_product_sp += Asp[i][k]*Bsp[k][j];
  //         }
  //           C_sp_par[i][j] = dot_product_sp;
  //     }
  //   }
  // }
  //   printf("\n" );
  // for(long i=0;i<N;i++){
  //   for(long j=0;j<N;j++){
  //     printf("%lf ",C_sp_par[i][j]);
  //   }
  //   printf("\n" );
  // }
  //   printf("\n" );
  omp_set_num_threads(threads);
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int	istart = (tid * N)/threads;
    int 	iend = ((tid+1)* N)/threads;
    for(int i=istart;i<iend;i++){
      for(int j=0;j<N;j++){
        C_sp_par[i][j] =0.0;
        dot_product_sp=0.0;
        #pragma omp parallel for schedule(static) reduction(+:dot_product_sp)
        for(int k=0;k<N;k++){
          dot_product_sp += Asp[i][k]*Bsp[k][j];
        }
        C_sp_par[i][j] = dot_product_sp;
      }
    }
  }

  // for(long i=0;i<N;i++){
  //   for(long j=0;j<N;j++){
  //     printf("%lf ",C_sp_par[i][j]);
  //   }
  //   printf("\n" );
  // }


  stop = clock();
  // End of CPU parallel Computation
  time_parallel = 1000.0* (stop-start)/(double)CLOCKS_PER_SEC;

  //Start of GPU computation.
    //code
  dim3 dimGrid(N/TILE_N,N/TILE_N,1);
  dim3 dimBlock(TILE_N,TILE_N,1);
  float *Csp_host, *Csp_dev;
  double *Cdp_host, *Cdp_dev;
  float *Asp_dev,*Bsp_dev; // Single precision
  double *Adp_dev,*Bdp_dev; // double precision
  float  dot_p_gpu_sp = 0.0;
  size_t size_M;
  float *Asp_tmp = (void **)malloc(size_M),*Bsp_tmp= (void **)malloc(size_M);
  for(int l = 0;l<N;l++){
    for(int m=0;m<N;m++){
      Asp_tmp[l*N+m] = Asp[l][m];
      Bsp_tmp[l*N+m] = Bsp[l][m];
    }
  }

  start = clock();
  size_M = N*N*sizeof(float);
  cudaMalloc((void **) &Csp_dev, size_M);  // Allocate array on device
  cudaMalloc((void **) &Asp_dev, size_M);  // Allocate array on device
  cudaMalloc((void **) &Bsp_dev, size_M);  // Allocate array on device
  // Initialize array in device to 0
  cudaMemset(Csp_dev, 0, size);
  cudaMemcpy(Asp_dev, Asp_tmp, size_M, cudaMemcpyHostToDevice);
  cudaMemcpy(Bsp_dev, Bsp_tmp, size_M, cudaMemcpyHostToDevice);

  matrix_mul_sp_kernel<<<dimGrid,dimBlock>>>(Asp_dev,Bsp_dev,Csp_dev,N);

  cudaMemcpy(Csp_host,Csp_dev, size_M, cudaMemcpyDeviceToHost);

  cudaFree(Csp_dev);cudaFree(Asp_dev);cudaFree(Bsp_dev);

  for(long i=0;i<N;i++){
    for(long j=0;j<N;j++){
      printf("%lf ",Csp_host[i][j]);
    }
    printf("\n" );
  }
  stop = clock();
  //End of GPU computation.
  time_cuda = 1000.0*(stop-start)/(double)CLOCKS_PER_SEC;
  printf("Time taken to execute CPU Serial program : %g ms \n", time_serial );
  printf("Time taken to execute CPU Parallel program : %g ms\n", time_parallel);
  printf("Time taken to execute GPU Parallel program : %g ms\n", time_cuda);

  printf("\n\n");
  printf("================================Double Precision========================\n");
  Adp = generate_matrix_dp(Adp,N);
  Bdp = generate_matrix_dp(Bdp,N);
  start = clock();
  double **C_dp =matrix_mul_dp(Adp,Bdp,N);
  stop = clock();

  //End of CPU serial computation.
  time_serial = 1000.0 * (stop-start)/(double)CLOCKS_PER_SEC;


  //Start of CPU parallel computation in OMP
  double ** C_dp_par = (double **)malloc(sizeof(double *)*N);
  for(long i=0; i < N; i++) C_dp_par[i] = malloc(sizeof(double) * N);

  double dot_product_dp = 0.0;

  start= clock();
  int i=0,j=0,k=0;
  #pragma omp parallel num_threads(threads) shared(C_dp_par,Adp,Bdp)
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
    // printf("%i\n",omp_get_num_threads() );
  }
  stop = clock();

  // End of CPU parallel computation.


  //Start of GPU computation in CUDA

  //End of GPU computation in CUDA

  printf("Time taken to execute CPU Serial program : %g ms \n", time_serial );
  printf("Time taken to execute CPU Parallel program  : %g ms \n", time_parallel);
  printf("Time taken to execute GPU Parallel program  : %g ms \n", time_cuda);





}
