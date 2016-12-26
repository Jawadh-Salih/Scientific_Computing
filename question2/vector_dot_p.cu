
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>
#include <string.h>

#define NUM_BLOCK 10
#define NUM_THREAD 1000
#define test_cases 3


#define THREADS_PER_BLOCK (100) //Threads per block we are using

float *   generate_vector_sp(float *V, unsigned long N){
  V = (float *)malloc(sizeof(float)*N);
  for(long i=0;i<N;i++){
    V[i] = (float)rand()/(float)RAND_MAX +1;
    //printf("%f\n",V[i] );
  }
  return V;
}
double *   generate_vector_dp(double *V, unsigned long N){
  V = (double *)malloc(sizeof(double)*N);
  for(long i=0;i<N;i++){
    V[i] = (double)rand()/(double)RAND_MAX +1;
    // printf("%.8f\n",V[i] );
  }
  return V;
}


float vector_dot_sp(float *V1,float *V2, unsigned long N){
  float dot_product = 0.0f;
  for(long i=0;i<N/10000;i++){
    float tmp = 0.0;
    for(int j=0;j<10000;j++){
      float val = V1[i*10000+j]*V2[i*10000+j];
      tmp += val;
    }
    dot_product += tmp;
  }

  return dot_product;
}
double vector_dot_dp(double *V1,double *V2, unsigned long N){
  double dot_product = 0.0f;
  for(long i=0;i<N;i++){
    dot_product = dot_product+ V1[i]*V2[i];
  }

  return dot_product;
}

__global__ void vector_dot_sp_gpu(float *V1_d,float *V2_d, float* dot_p_d, unsigned long N, int nthreads, int nblocks) {
  
  int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
   for (long i=idx; i< N; i+=nthreads*nblocks) {
       dot_p_d[idx] += V1_d[i]*V2_d[i];
   }
  
  //  printf("Dot p for now is : %f\n", V1_d[3]);
}
__global__ void vector_dot_dp_gpu(double *V1_d,double *V2_d, double* dot_p_d,unsigned long N, int nthreads, int nblocks) {
  int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks
  for (long i=idx; i< N; i+=nthreads*nblocks) {
      dot_p_d[idx] += V1_d[i]*V2_d[i];

  }
  // printf("Dot p for now is : %g\n", V1_d[3]);

}


int main(int argc, char * argv[]){


  // Output variables
  unsigned long N[3]= {10000000, 50000000, 100000000};

  size_t size,size_V;
  float *V1sp,*V2sp; // Single precision
  double *V1dp,*V2dp; // double precision
  int thread =2;
  printf("argc %i\n", argc );
  if(argc == 2)
    thread = atoi(argv[1]);


  printf("========================================================================\n");
  printf("=====================Vector Dot Product calculation=====================\n");
  printf("========================================================================\n");

  printf("Thread Count is : %i\n", thread );
  unsigned seed = (time(NULL));
  srand(seed);
  clock_t start_cpu_serial,stop_cpu_serial;
  clock_t start_cpu_par,stop_cpu_par;
  clock_t start_gpu_par,stop_gpu_par;

  for(int n=0;n<test_cases;n++){
      // Generate Random Vectors for the Computation.
      V1dp = generate_vector_dp(V1dp,N[n]);
      V2dp = generate_vector_dp(V2dp,N[n]);
      V1sp = (float *)malloc(sizeof(float)*N[n]);// generate_vector_sp(V1sp,N[n]);
      V2sp = (float *)malloc(sizeof(float)*N[n]);//generate_vector_sp(V2sp,N[n]);
      for (int l=0;l<N[n];l++){
        V1sp[l] = (float)V1dp[l];
        V2sp[l] = (float)V2dp[l];
      }

      printf("\nComputation is being processed........ \n");

      printf("\nFor N = %ld\n",N[n] );

      printf("======================================Single Precision==============================\n\n");
      // Serial Computation
      start_cpu_serial = clock();
      float dot_p_sp = vector_dot_sp(V1sp,V2sp,N[n]);
      stop_cpu_serial = clock();

      // Parallel computation in OMP
      float dot_p_parallel_sp = 0.0;
      start_cpu_par = clock();
      omp_set_dynamic(0);
      omp_set_num_threads(thread);
      #pragma omp parallel
      {
        #pragma omp for schedule(static) reduction(+:dot_p_parallel_sp)
        for (long i = 0; i < N[n]; i++) {
            dot_p_parallel_sp = dot_p_parallel_sp + (V1sp[i] * V2sp[i]);
        }
        // dot_p_parallel_sp = vector_dot_sp(V1sp,V2sp,N);

      }
      stop_cpu_par = clock();

      // GPU (CUDA) computation

      // // Variable necessary For GPU Computation
      dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions
      dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions


      float *dot_p_host_sp, *dot_p_dev_sp;
      double *dot_p_host_dp, *dot_p_dev_dp;
      float *V1sp_dev,*V2sp_dev; // Single precision
      double *V1dp_dev,*V2dp_dev; // double precision
      float  dot_p_gpu_sp = 0.0;
      // End of necessary variable declarations for GPU computation.


      size = NUM_BLOCK*NUM_THREAD*sizeof(float);  //Array memory size
      size_V = N[n]*sizeof(float);  //Array memory size
      dot_p_host_sp = (float *)malloc(size);  //  Allocate array on host

      start_gpu_par = clock(); //Starting counting time.

      cudaMalloc((void **) &dot_p_dev_sp, size);  // Allocate array on device
      cudaMalloc((void **) &V1sp_dev, size_V);  // Allocate array on device
      cudaMalloc((void **) &V2sp_dev, size_V);  // Allocate array on device
      // Initialize array in device to 0
      cudaMemset(dot_p_dev_sp, 0, size);
      cudaMemcpy(V1sp_dev, V1sp, size_V, cudaMemcpyHostToDevice);
      cudaMemcpy(V2sp_dev, V2sp, size_V, cudaMemcpyHostToDevice);

      // Do calculation on device
      vector_dot_sp_gpu <<< dimGrid,dimBlock>>> (V1sp_dev,V2sp_dev,dot_p_dev_sp ,N[n]), NUM_THREAD, NUM_BLOCK); // call CUDA kernel
      // Retrieve result from device and store it in host array
      cudaMemcpy(dot_p_host_sp, dot_p_dev_sp, size, cudaMemcpyDeviceToHost);


       for(int tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
        dot_p_gpu_sp += dot_p_host_sp[tid];

      cudaFree(dot_p_dev_sp);cudaFree(V1sp_dev);cudaFree(V2sp_dev);
      free(V1sp);free(V2sp);free(dot_p_host_sp);
      stop_gpu_par = clock();

      printf("CPU serial dot product is %f\n",dot_p_sp );
      printf("Time taken to execute Serial program : %f ms \n", 1000*(stop_cpu_serial-start_cpu_serial)/(float)CLOCKS_PER_SEC);
      printf("\n" );

      printf("CPU parallel dot product is %f\n",dot_p_parallel_sp );
      printf("Time taken to execute the program in CPU parallely : %f ms\n",  1000*(stop_cpu_par-start_cpu_par)/(float)CLOCKS_PER_SEC);
      printf("\n" );

      printf("GPU dot product is %f\n",dot_p_gpu_sp );
      printf("Time taken to execute the program in GPU : %f ms\n", 1000*(stop_gpu_par-start_gpu_par)/(float)CLOCKS_PER_SEC);
      printf("\n" );

      printf("======================================Double Precision=================================\n\n");

      // Serial computation.
      start_cpu_serial = clock();
      double dot_p_dp = vector_dot_dp(V1dp,V2dp,N[n]);
      stop_cpu_serial = clock();

      // parallel computation in OMP
      double dot_p_parallel_dp = 0.0;

      start_cpu_par = clock();
      omp_set_dynamic(0);
      omp_set_num_threads(thread);
      #pragma omp parallel
      {
        #pragma omp for schedule(static) reduction(+:dot_p_parallel_dp)
        for (long i = 0; i < N[n]; i++) {
          dot_p_parallel_dp   = dot_p_parallel_dp + (V1dp[i] * V2dp[i]);
        }
        // printf("Num threads %d\n",omp_get_num_threads() );
        // dot_p_parallel_dp = vector_dot_dp(V1dp,V2dp,N);

      }
      stop_cpu_par = clock();

      // GPU (CUDA) computation
      double  dot_p_gpu_dp = 0.0;
      size = NUM_BLOCK*NUM_THREAD*sizeof(double);  //Array memory size
      size_V = N[n]*sizeof(double);  //Array memory size
      dot_p_host_dp = (double *)malloc(size);  //  Allocate array on host

      start_gpu_par = clock(); //Starting counting time.

      cudaMalloc((void **) &dot_p_dev_dp, size);  // Allocate array on device
      cudaMalloc((void **) &V1dp_dev, size_V);  // Allocate array on device
      cudaMalloc((void **) &V2dp_dev, size_V);  // Allocate array on device
      // Initialize array in device to 0
      cudaMemset(dot_p_dev_dp, 0, size);
      cudaMemcpy(V1dp_dev, V1dp, size_V, cudaMemcpyHostToDevice);
      cudaMemcpy(V2dp_dev, V2dp, size_V, cudaMemcpyHostToDevice);

      // Do calculation on device
      vector_dot_dp_gpu <<<dimGrid, dimBlock>>> (V1dp_dev,V2dp_dev,dot_p_dev_dp ,N[n], NUM_THREAD, NUM_BLOCK); // call CUDA kernel
      // Retrieve result from device and store it in host array
      cudaMemcpy(dot_p_host_dp, dot_p_dev_dp, size, cudaMemcpyDeviceToHost);

      cudaFree(dot_p_dev_dp);cudaFree(V1dp_dev);cudaFree(V2dp_dev);
      for(int tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
        dot_p_gpu_dp += dot_p_host_dp[tid];

      stop_gpu_par = clock();

      printf("CPU serial dot product is %lf\n",dot_p_dp );
      printf("Time taken to execute the program in CPU serially : %f ms \n", 1000*(stop_cpu_serial-start_cpu_serial)/(float)CLOCKS_PER_SEC );
      printf("\n" );

      printf("CPU parallel dot product is %lf\n",dot_p_parallel_dp );
      printf("Time taken to execute Parallel program  : %f ms\n", 1000*(stop_cpu_par-start_cpu_par)/(float)CLOCKS_PER_SEC );
      printf("\n" );

      printf("GPU dot product is %lf\n",dot_p_gpu_dp );
      printf("Time taken to execute the program in GPU : %f ms\n", 1000* (stop_gpu_par-start_gpu_par)/(float)CLOCKS_PER_SEC);
      printf("\n" );

    }

}
