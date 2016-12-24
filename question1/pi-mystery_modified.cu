// source: http://cacs.usc.edu/education/cs596/src/cuda/pi.cu

// Using CUDA device to calculate pi

#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define NBIN 268435456	  // Number of bins
#define NUM_BLOCK  30  // Number of thread blocks
#define NUM_THREAD  8  // Number of threads per block
#define PI 3.1415926535
//
// #Ifdef DOUBLE_PRECISION
//
// #else
//
// #endif
int tid;
float pi = 0;
double pi_dp = 0;

// Kernel that executes on the CUDA device
__global__
void cal_pi(float *sum, int nbin, float step, int nthreads, int nblocks) {
	int i;

	float x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks

	for (i=idx; i<nbin; i+=nthreads*nblocks) {
		x = (i+0.5)*step;
		sum[idx] += 4.0/(1.0+x*x);
	}

}
__global__
void cal_pi_dp(double *sum, int nbin, double step, int nthreads, int nblocks) {
	int i;

	double x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;  // Sequential thread index across the blocks

	for (i=idx; i<nbin; i+=nthreads*nblocks) {
		x = (i+0.5)*step;
		sum[idx] += 4.0/(1.0+x*x);
	}

}

// Main routine that executes on the host
int main(void) {
	dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions
	dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions
	float *sumHost, *sumDev;  // Pointer to host & device arrays
	double *sumHost_dp,*sumDev_dp;
	clock_t start, stop;
	long N[4] = {1,16777216,67108864,268435456};

	size_t size = NUM_BLOCK*NUM_THREAD*sizeof(float);  //Array memory size
	for(int n=0;n<4;n++){
			if(n>0){
			printf("=====================SINGLE PRECISION===============================\n");
			printf("\n" );
			printf("\n" );
			printf("# of Bins = %li, # of blocks = %i, # of threads/block = %d.\n", N[n],NUM_BLOCK, NUM_THREAD);
			printf("\n" );
		}


	float step = 1.0/N[n];  // Step size
	start = clock();
	sumHost = (float *)malloc(size);  //  Allocate array on host
	cudaMalloc((void **) &sumDev, size);  // Allocate array on device
	// Initialize array in device to 0
	cudaMemset(sumDev, 0, size);
	// Do calculation on device

	cal_pi <<<dimGrid, dimBlock>>> (sumDev, NBIN, step, NUM_THREAD, NUM_BLOCK); // call CUDA kernel

	// Retrieve result from device and store it in host array
	cudaMemcpy(sumHost, sumDev, size, cudaMemcpyDeviceToHost);
	for(tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
		pi += sumHost[tid];
	pi *= step;

	stop = clock();

	if(n>0){
		// Print results
		printf("CUDA Estimated PI  = %f [error %f] \n",pi,pi-PI);
		printf("GPU pi calculated in %f ms.\n", 1000*(stop-start)/(float)CLOCKS_PER_SEC);
		// Cleanup
	}
	free(sumHost);
	cudaFree(sumDev);
	if(n>0){
		printf("=====================DOUBLE PRECISION===============================\n");
		printf("\n" );
		printf("\n" );
		printf("# of Bins = %li, # of blocks = %i, # of threads/block = %d.\n", N[n],NUM_BLOCK, NUM_THREAD);
		printf("\n" );
	}
	double step_dp = 1.0/N[n];  // Step size
	start = clock();
	size = NUM_BLOCK*NUM_THREAD*sizeof(double);
	sumHost_dp = (double *)malloc(size);  //  Allocate array on host
	cudaMalloc((void **) &sumDev_dp, size);  // Allocate array on device
	// Initialize array in device to 0
	cudaMemset(sumDev, 0, size);
	// Do calculation on device

	cal_pi_dp <<<dimGrid, dimBlock>>> (sumDev_dp, N[n], step_dp, NUM_THREAD, NUM_BLOCK); // call CUDA kernel

	// Retrieve result from device and store it in host array
	cudaMemcpy(sumHost_dp, sumDev_dp, size, cudaMemcpyDeviceToHost);
	for(tid=0; tid<NUM_THREAD*NUM_BLOCK; tid++)
		pi += sumHost_dp[tid];
	pi *= step_dp;

	stop = clock();

	if(n>0){
		// Print results
		printf("CUDA Estmated PI =  %f [error %f] \n",pi,pi-PI);
		printf("GPU pi calculated in %f ms.\n", 1000*(stop-start)/(double)CLOCKS_PER_SEC);
		// Cleanup
	}
	free(sumHost_dp);
	cudaFree(sumDev_dp);
}
	return 0;
}
