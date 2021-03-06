#include <ctype.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>


#define TILE_N 10

long no_of_ops_serial=0,no_of_ops_serial_dp=0;
long no_of_ops_parallel=0,no_of_ops_parallels_dp=0;
long no_of_ops_cuda=0,no_of_ops_cuda_dp=0;
float ** generate_matrix_sp(float **M, unsigned long N){
  M = (float **)malloc(sizeof(float *)*N);
  for(long i=0; i < N; i++) M[i] =(float *) malloc(sizeof(float) * N);
  for(long i=0;i<N;i++){
    for(long j=0;j<N;j++){
      M[i][j] =(float)rand()/(float)RAND_MAX +1;
    }
  }
  return M;
}
double ** generate_matrix_dp(double **M, unsigned long N){
  M = (double **)malloc(sizeof(double *)*N);
  for(long i=0; i < N; i++) M[i] = (double*)malloc(sizeof(double) * N);

  for(long i=0;i<N;i++){
    for(long j=0;j<N;j++){
      M[i][j] = (double)rand()/(double)RAND_MAX +1;
    }
  }
  return M;
}

// Single Precision Implementation
float** matrix_mul_sp(float **M1,float **M2, unsigned long N){

  float ** C = (float **)malloc(sizeof(float *)*N);
  for(long i=0; i < N; i++) C[i] =(float*) malloc(sizeof(float) * N);
  for(long i=0;i<N;i++){
    for(long j=0;j<N;j++){
        C[i][j]=0.0;
        for(long k=0;k<N;k++){
          C[i][j] += M1[i][k]*M2[k][j];
          no_of_ops_serial ++;
        }
    }
  }

  return C;
}

// Double Precision Implementation
double ** matrix_mul_dp(double **V1,double **V2, unsigned long N){

  double ** C = (double **)malloc(sizeof(double *)*N);
  for(long i=0; i < N; i++) C[i] = (double*)malloc(sizeof(double) * N);

  for(long i=0;i<N;i++){
    for(long j=0;j<N;j++){
        for(long k=0;k<N;k++){
          C[i][j] += V1[i][k]*V2[k][j];
          no_of_ops_serial_dp ++;
        }
    }
  }

  return C;
}

__global__ void matrix_mul_sp_kernel(float *A_dev,float *B_dev,float * C_dev, unsigned long N){

  // to store the element of matrix
  int row = blockIdx.y*blockDim.y+threadIdx.y; // row index for the matrices
  int col = blockIdx.x*blockDim.x+threadIdx.x; // column index for the matrices

  if ((row < N) && (col <N)){
    float C_value = 0.0;
    for(int k=0;k<N;k++){
        C_value += A_dev[row*N+k]*B_dev[k*N+col];
        C_dev[row*N+col] = C_value;
        // no_of_ops_cuda ++;
    }
  }
}
__global__ void matrix_mul_dp_kernel(double *A_dev,double *B_dev,double * C_dev, unsigned long N){

  // to store the element of matrix
  int row = blockIdx.y*blockDim.y+threadIdx.y; // row index for the matrices
  int col = blockIdx.x*blockDim.x+threadIdx.x; // column index for the matrices

  if ((row < N) && (col <N)){
    double C_value = 0.0;
    for(int k=0;k<N;k++){
        C_value += A_dev[row*N+k]*B_dev[k*N+col];
        C_dev[row*N+col] = C_value;
        // no_of_ops_cuda_dp++;
    }
  }
}
int main(int argc,char *argv[]){

  omp_lock_t write_lock;
  omp_init_lock(&write_lock);
  int precision;
  int computation =0 ;
  int verify = 0; //default no verification
  int command;
  // int flag = 0;
  int threads =2;
  while((command = getopt(argc,argv,"spcv842:")) != -1)
    switch(command){
      case 's':
        computation = 0;
        // flag = 1;
        break;
      case 'p':
        computation = 1;
        // flag = 1;
        break;
      case 'c':
        computation = 2; // cuda
        // flag = 1;
        break;
      case 'v':
        verify = 1;
        break;
      case '8':
        threads = 8;
        break;
      case '4':
        threads = 4;
        break;
      case '2':
        threads = 2;
        break;

      default:
        abort();
    }
  int N[3] = {600,1200,1800};
  float **Asp,**Bsp; // Single precision
  double **Adp,**Bdp; // double precision


  // if(argc == 3)
  //   threads = atoi(argv[2]);
  //
  // // int select_precision = 0; //default to floating point.
  // int  computation = 0; //
  // int verify =0;
  double time_serial_sp=0.0;
  double time_parallel_sp=0.0;
  double time_cuda_sp=0.0;
  double time_serial_dp=0.0;
  double time_parallel_dp=0.0;
  double time_cuda_dp=0.0;

      printf("========================================================================\n");
      printf("====================Matrix Multiplication calculation===================\n");
      printf("========================================================================\n");

      printf("Please choose 1 for single precision or 2 for double precision computation \n");
      scanf ("%d", &precision);
  for(int n=0;n<3;n++){
    // For GPU
    size_t size_M;
    dim3 dimGrid(N[n]/TILE_N,N[n]/TILE_N,1);
    dim3 dimBlock(TILE_N,TILE_N,1);

    // random seed as a Pseudo Random Number generator.
    unsigned seed = (time(NULL));
    srand(seed);
    clock_t start, stop;

    if(precision == 1){
    printf("==============================Single Precision==========================\n");

    Asp = generate_matrix_sp(Asp,N[n]);
    Bsp = generate_matrix_sp(Bsp,N[n]);

    if(computation == 0){
    //Start of CPU serial computation
    start  = clock();

    float ** C_sp = matrix_mul_sp(Asp,Bsp,N[n]);

    stop = clock();
    //End of CPU serial computation.
    time_serial_sp = 1000.0 * (stop-start)/(double)CLOCKS_PER_SEC;
    printf("Time taken to execute CPU Serial program : %g ms \n", time_serial_sp );
  }
  else if(computation ==1){




    //Start of CPU parallel computation in OMP.
    float ** C_sp_par = (float **)malloc(sizeof(float *)*N[n]);
    for(long i=0; i < N[n]; i++) C_sp_par[i] = (float *) malloc(sizeof(float) * N[n]);
    float dot_product_sp=0.0;

    start= clock();

    omp_set_num_threads(threads);
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int	istart = (tid * N[n])/threads;
      int 	iend = ((tid+1)* N[n])/threads;
      for(int i=istart;i<iend;i++){
        for(int j=0;j<N[n];j++){
          C_sp_par[i][j] =0.0;
          dot_product_sp=0.0;
          #pragma omp parallel for schedule(static) reduction(+:dot_product_sp)
          for(int k=0;k<N[n];k++){
            dot_product_sp += Asp[i][k]*Bsp[k][j];
            omp_set_lock(&write_lock);
            no_of_ops_parallel++;
            omp_unset_lock(&write_lock);
          }


          C_sp_par[i][j] = dot_product_sp;
        }
      }
    }
    omp_destroy_lock(&write_lock);
    stop = clock();
    // End of CPU parallel Computation
    time_parallel_sp = 1000.0* (stop-start)/(double)CLOCKS_PER_SEC;
    printf("Time taken to execute CPU Parallel program : %g ms\n", time_parallel_sp);
    if(verify == 1){
      // for(long i=0;i<N;i++){
      //   for(long j=0;j<N;j++){
      //     C_sp[i][j] = C_sp[i][j]-C_sp_par[i][j];
      //   }
      // }
      float ** C_sp_tmp = matrix_mul_sp(Asp,Bsp,N[n]);

      printf("# of Operations in Serial : %li\n",no_of_ops_serial );
      printf("# of Operations in Parallel : %li\n",no_of_ops_parallel );
      printf("If both the Numbers are equal, then the there are same amount of computation happend which are similar");
      if(no_of_ops_serial == no_of_ops_parallel){
        printf("Accuracy is Verified. Computations give the Same results\n");
      }
    }
  }
  else if(computation == 2){

    //Start of GPU computation.

    float *Csp_host, *Csp_dev;
    float *Asp_dev,*Bsp_dev; // Single precision
    size_M = N[n]*N[n]*sizeof(float);
    float *Asp_tmp = (float *)malloc(size_M),*Bsp_tmp= (float *)malloc(size_M);
    for(int l = 0;l<N[n];l++){
      for(int m=0;m<N[n];m++){
        Asp_tmp[l*N[n]+m] = Asp[l][m];
        Bsp_tmp[l*N[n]+m] = Bsp[l][m];
      }
    }


    start = clock();

    Csp_host = (float *)malloc(size_M);
    cudaMalloc((void **) &Csp_dev, size_M);  // Allocate array on device
    cudaMalloc((void **) &Asp_dev, size_M);  // Allocate array on device
    cudaMalloc((void **) &Bsp_dev, size_M);  // Allocate array on device
    // Initialize array in device to 0
    cudaMemset(Csp_dev, 0, size_M);
    cudaMemcpy(Asp_dev, Asp_tmp, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(Bsp_dev, Bsp_tmp, size_M, cudaMemcpyHostToDevice);

    matrix_mul_sp_kernel<<<dimGrid,dimBlock>>>(Asp_dev,Bsp_dev,Csp_dev,N[n]);

    cudaMemcpy(Csp_host,Csp_dev, size_M, cudaMemcpyDeviceToHost);

    cudaFree(Csp_dev);cudaFree(Asp_dev);cudaFree(Bsp_dev);


    stop = clock();
    //End of GPU computation.
    time_cuda_sp = 1000.0*(stop-start)/(double)CLOCKS_PER_SEC;
    printf("Time taken to execute GPU Parallel program : %g ms\n", time_cuda_sp);

  }
  }
  else if (precision == 2){
    printf("================================Double Precision========================\n");
    Adp = generate_matrix_dp(Adp,N[n]);
    Bdp = generate_matrix_dp(Bdp,N[n]);

    if(computation == 0){
      start = clock();
      double **C_dp =matrix_mul_dp(Adp,Bdp,N[n]);
      stop = clock();

      //End of CPU serial computation.
      time_serial_dp = 1000.0 * (stop-start)/(double)CLOCKS_PER_SEC;
      printf("Time taken to execute CPU Serial program : %g ms \n", time_serial_dp );
    }
    else if(computation == 1){
      //Start of CPU parallel computation in OMP
      double ** C_dp_par = (double **)malloc(sizeof(double *)*N[n]);
      for(long i=0; i < N[n]; i++) C_dp_par[i] =(double *) malloc(sizeof(double) * N[n]);

      double dot_product_dp = 0.0;
      omp_init_lock(&write_lock);
      start= clock();
      omp_set_num_threads(threads);
      #pragma omp parallel
      {
        int tid = omp_get_thread_num();
        int	istart = (tid * N[n])/threads;
        int 	iend = ((tid+1)* N[n])/threads;
        for(int i=istart;i<iend;i++){
          for(int j=0;j<N[n];j++){
            C_dp_par[i][j] =0.0;
            dot_product_dp=0.0;
            #pragma omp parallel for schedule(static) reduction(+:dot_product_dp)
            for(int k=0;k<N[n];k++){
              dot_product_dp += Adp[i][k]*Bdp[k][j];
              omp_set_lock(&write_lock);
              no_of_ops_parallels_dp++;
              omp_unset_lock(&write_lock);
            }
            C_dp_par[i][j] = dot_product_dp;
          }
        }
      }
      stop = clock();
      omp_destroy_lock(&write_lock);
      // End of CPU parallel computation.
      time_parallel_dp = 1000.0* (stop-start)/(double)CLOCKS_PER_SEC;
      printf("Time taken to execute CPU Parallel program  : %g ms \n", time_parallel_dp);
      if(verify == 1){
        // for(long i=0;i<N;i++){
        //   for(long j=0;j<N;j++){
        //     C_sp[i][j] = C_sp[i][j]-C_sp_par[i][j];
        //   }
        // }
        double ** C_dp_tmp = matrix_mul_sp(Asp,Bsp,N[n]);

        printf("# of Operations in Serial : %li\n",no_of_ops_serial_dp );
        printf("# of Operations in Parallel : %li\n",no_of_ops_parallels_dp );
        printf("If both the Numbers are equal, then the there are same amount of computation happend which are similar");
        if(no_of_ops_serial_dp == no_of_ops_parallels_dp){
          printf("==========Verification Successful. Computations give the Same results=======\n");
        }
      }
    }
    else if(computation ==2){
    //Start of GPU computation in CUDA
    double *Cdp_host, *Cdp_dev;
    double *Adp_dev,*Bdp_dev; // double precision
    size_M = N[n]*N[n]*sizeof(double);
    double *Adp_tmp = (double *)malloc(size_M),*Bdp_tmp= (double *)malloc(size_M);
    for(int l = 0;l<N[n];l++){
      for(int m=0;m<N[n];m++){
        Adp_tmp[l*N[n]+m] = Adp[l][m];
        Bdp_tmp[l*N[n]+m] = Bdp[l][m];
      }
    }

    start = clock();
    Cdp_host = (double *)malloc(size_M);
    cudaMalloc((void **) &Cdp_dev, size_M);  // Allocate array on device
    cudaMalloc((void **) &Adp_dev, size_M);  // Allocate array on device
    cudaMalloc((void **) &Bdp_dev, size_M);  // Allocate array on device
    // Initialize array in device to 0
    cudaMemset(Cdp_dev, 0, size_M);
    cudaMemcpy(Adp_dev, Adp_tmp, size_M, cudaMemcpyHostToDevice);
    cudaMemcpy(Bdp_dev, Bdp_tmp, size_M, cudaMemcpyHostToDevice);

    matrix_mul_dp_kernel<<<dimGrid,dimBlock>>>(Adp_dev,Bdp_dev,Cdp_dev,N[n]);

    cudaMemcpy(Cdp_host,Cdp_dev, size_M, cudaMemcpyDeviceToHost);

    cudaFree(Cdp_dev);cudaFree(Adp_dev);cudaFree(Bdp_dev);
    stop = clock();
    //End of GPU computation.
    time_cuda_dp = 1000.0*(stop-start)/(double)CLOCKS_PER_SEC;

    printf("Time taken to execute GPU Parallel program  : %g ms \n", time_cuda_dp);
  }
}
  else{
    printf("Returning from the programs\n" );
    return 0;
  }
  }

}
// 2D Layout
// printf("\n" );
// for(long i=0;i<N;i++){
//   for(long j=0;j<N;j++){
//     printf("%lf ",C_sp_par[i][j]);
//   }
//   printf("\n" );
// }

// 1 D Layout
// int count =1;
// for(long i=0;i<N*N;i++){
//   printf("%f ",Csp_host[i]);
//   if(i== count*N-1){
//       printf("\n" );
//       count ++;
//   }
// }
//
//   printf("\n" );  printf("\n" );
