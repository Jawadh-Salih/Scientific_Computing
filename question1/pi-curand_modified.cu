// Source: http://web.mit.edu/pocky/www/cudaworkshop/MonteCarlo/Pi.cu

// Written by Barry Wilkinson, UNC-Charlotte. Pi.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include <omp.h>

#define TRIALS_PER_THREAD 4096
#define BLOCKS 256
#define THREADS 256
#define PI 3.1415926535  // known value of pi

__global__ void gpu_monte_carlo(float *estimate, curandState *states) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	float x, y;

	curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND


	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (float) TRIALS_PER_THREAD; // return estimate of pi
}

float host_monte_carlo(long trials) {
	float x, y;
	long points_in_circle;
	for(long i = 0; i < trials; i++) {
		x = rand() / (float) RAND_MAX;
		y = rand() / (float) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}

int main (int argc, char *argv[]) {
	int threads = 8;
	clock_t start, stop;
	float host[BLOCKS * THREADS];
	float *dev;
	curandState *devStates;

	printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD,
BLOCKS, THREADS);

	start = clock();

	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float)); // allocate device mem. for counts

	cudaMalloc( (void **)&devStates, THREADS * BLOCKS * sizeof(curandState) );

	gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev, devStates);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost); // return results

	float pi_gpu =0.0;
	for(int i = 0; i < BLOCKS * THREADS; i++) {
		pi_gpu += host[i];
	}

	pi_gpu /= (BLOCKS * THREADS);

	stop = clock();

	printf("GPU pi calculated in %f ms.\n", 1000* (stop-start)/(float)CLOCKS_PER_SEC);

	start = clock();
	float pi_cpu = host_monte_carlo(BLOCKS * THREADS * TRIALS_PER_THREAD);
	stop = clock();
	printf("CPU pi calculated in %f ms.\n",1000* (stop-start)/(float)CLOCKS_PER_SEC);

	// PI calculated from CPU parallel computation
	start = clock();

	omp_set_dynamic(0);
	omp_set_num_threads(threads);
	float points_in_circle[threads];//= 0.0;
	long long trials = TRIALS_PER_THREAD*THREADS*BLOCKS;
	float x = 0.0,y = 0.0;

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int thre = threads;//omp_get_num_threads();
		long	istart = (tid * trials)/thre;
		long 	iend = ((tid+1)* trials)/thre;
		points_in_circle[tid] = 0.0;
		for(long long i = istart; i <iend; i++) {
			x = rand() / (float) RAND_MAX;
			y = rand() / (float) RAND_MAX;
			points_in_circle[tid] += (x*x + y*y <= 1.0f);
		}
		printf("%d\n",tid );
	}
	float circle_points =0.0;
	for(int k=0;k<threads;k++){
	 	circle_points += points_in_circle[k];
	}
	float pi_cpu_par = 4.0*circle_points/trials;
	stop = clock();

	printf("CPU parallel pi calculated in %f ms.\n",1000* (stop-start)/(float)CLOCKS_PER_SEC);

	printf("\n");
	printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
	printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
	printf("CPU parallel estimate of PI = %f [error of %f]\n", pi_cpu_par, pi_cpu_par - PI);

	return 0;
}
