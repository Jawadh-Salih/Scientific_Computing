// source: http://cacs.usc.edu/education/cs596/src/cuda/pi.cu

// Using CUDA device to calculate pi

#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define NBIN 1118481  // Number of bins
#define NUM_BLOCK  30  // Number of thread blocks
#define NUM_THREAD  8  // Number of threads per block
#define PI 3.1415926535

#Ifdef DOUBLE_PRECISION
	
#else

#endif
int tid;
float pi = 0;

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

// Main routine that executes on the host
int main(void) {
	dim3 dimGrid(NUM_BLOCK,1,1);  // Grid dimensions
	dim3 dimBlock(NUM_THREAD,1,1);  // Block dimensions
	float *sumHost, *sumDev;  // Pointer to host & device arrays
	clock_t start, stop;

	float step = 1.0/NBIN;  // Step size
	size_t size = NUM_BLOCK*NUM_THREAD*sizeof(float);  //Array memory size

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

	printf("# of  bins  = %d, # of blocks = %d, # of threads/block = %d.\n", NBIN,NUM_BLOCK,NUM_THREAD);
	// Print results
	printf("GPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);
	printf("PI with the error of  = %f [error %f] \n",pi,pi-PI);
	// Cleanup
	free(sumHost);
	cudaFree(sumDev);

	return 0;
}
