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
__global__ void gpu_monte_carlo_dp(double *estimate, curandState *states) {
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	double x, y;

	curand_init(1234, tid, 0, &states[tid]);  // 	Initialize CURAND


	for(int i = 0; i < TRIALS_PER_THREAD; i++) {
		x = curand_uniform (&states[tid]);
		y = curand_uniform (&states[tid]);
		points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
	}
	estimate[tid] = 4.0f * points_in_circle / (double) TRIALS_PER_THREAD; // return estimate of pi
}

float host_monte_carlo(long trials) {
	double x, y;
	long points_in_circle;
	for(long i = 0; i < trials; i++) {
		x = rand() / (double) RAND_MAX;
		y = rand() / (double) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}
double host_monte_carlo_dp(long trials) {
	double x, y;
	long points_in_circle;
	for(long i = 0; i < trials; i++) {
		x = rand() / (double) RAND_MAX;
		y = rand() / (double) RAND_MAX;
		points_in_circle += (x*x + y*y <= 1.0f);
	}
	return 4.0f * points_in_circle / trials;
}

float CPU_parallel(int threads,int N){
	omp_set_dynamic(0);
	omp_set_num_threads(threads);
	float points_in_circle[threads];//= 0.0;
	long long trials = N*THREADS*BLOCKS;
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
	}
	float circle_points =0.0;
	for(int k=0;k<threads;k++){
		circle_points += points_in_circle[k];
	}
	float pi_cpu_par = 4.0*circle_points/trials;

	return pi_cpu_par;
}
double CPU_parallel_dp(int threads,int N){
	omp_set_dynamic(0);
	omp_set_num_threads(threads);
	double points_in_circle[threads];//= 0.0;
	long long trials = N*THREADS*BLOCKS;
	double x = 0.0,y = 0.0;

	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		int thre = threads;//omp_get_num_threads();
		long	istart = (tid * trials)/thre;
		long 	iend = ((tid+1)* trials)/thre;
		points_in_circle[tid] = 0.0;
		for(long long i = istart; i <iend; i++) {
			x = rand() / (double) RAND_MAX;
			y = rand() / (double) RAND_MAX;
			points_in_circle[tid] += (x*x + y*y <= 1.0f);
		}
	}
	double circle_points =0.0;
	for(int k=0;k<threads;k++){
		circle_points += points_in_circle[k];
	}
	double pi_cpu_par = 4.0*circle_points/trials;

	return pi_cpu_par;
}

float GPU_parallel(float * dev,curandState * devStates,float host[]){

	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(float)); // allocate device mem. for counts

	cudaMalloc( (void **)&devStates, THREADS * BLOCKS * sizeof(curandState) );

	gpu_monte_carlo<<<BLOCKS, THREADS>>>(dev, devStates);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost); // return results

	float pi_gpu =0.0;
	for(int i = 0; i < BLOCKS * THREADS; i++) {
		pi_gpu += host[i];
	}

	pi_gpu /= (BLOCKS * THREADS);
	return pi_gpu;
}

double GPU_parallel(double * dev,curandState * devStates,double host[]){

	cudaMalloc((void **) &dev, BLOCKS * THREADS * sizeof(double)); // allocate device mem. for counts

	cudaMalloc( (void **)&devStates, THREADS * BLOCKS * sizeof(curandState) );

	gpu_monte_carlo_dp<<<BLOCKS, THREADS>>>(dev, devStates);

	cudaMemcpy(host, dev, BLOCKS * THREADS * sizeof(double), cudaMemcpyDeviceToHost); // return results

	double pi_gpu =0.0;
	for(int i = 0; i < BLOCKS * THREADS; i++) {
		pi_gpu += host[i];
	}

	pi_gpu /= (BLOCKS * THREADS);

	return pi_gpu;
}
int main (int argc, char *argv[]) {
	int threads = 2;

	if(argc == 2){
		threads = atoi(argv[1]);
	}
	clock_t start, stop;
	float host[BLOCKS * THREADS];
	float *dev;
	double host_dp[BLOCKS * THREADS];
	double *dev_dp;
	curandState *devStates;
	int N[4] = {1,256,1024,4096};
	int precision = 1;
	      printf("Please choose 1 for single precision or 2 for double precision computation \n");
	      scanf ("%d", &precision);

	for(int n=0;n<4;n++){

		if(precision == 1){
		if(n>0){
		printf("=====================SINGLE PRECISION===============================\n");
		printf("\n" );
		printf("\n" );
		printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d. # of Threads %d\n ", N[n],BLOCKS, THREADS,threads);
		printf("\n" );
	}
		start = clock();

		float pi_gpu = GPU_parallel(dev,devStates,host);



		stop = clock();

		if(n>0){
			printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu, pi_gpu - PI);
			printf("GPU pi calculated in %f ms.\n", 1000* (stop-start)/(float)CLOCKS_PER_SEC);
			printf("\n" );
		}
		// PI calculated from CPU serial computation
		start = clock();
		float pi_cpu = host_monte_carlo(BLOCKS * THREADS * N[n]);
		stop = clock();
		if(n>0){
			printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
			printf("CPU pi calculated in %f ms.\n",1000* (stop-start)/(float)CLOCKS_PER_SEC);
			printf("\n" );
		}
		// PI calculated from CPU parallel computation
		float pi_cpu_par  =0.0;
		start = clock();
		pi_cpu_par = CPU_parallel(threads,N[n]);
		stop = clock();
		if(n>0){
			printf("CPU parallel estimate of PI = %f [error of %f]\n", pi_cpu_par, pi_cpu_par - PI);
			printf("CPU parallel pi calculated in %f ms.With Thread count of %i\n",1000* (stop-start)/(float)CLOCKS_PER_SEC, threads);
			printf("\n" );
		}
}
else if(precision == 2){
if(n>0){
		printf("=====================DOUBLE PRECISION===============================\n");
		printf("\n" );
		printf("\n" );
		printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d. # of Threads %d\n", N[n],BLOCKS, THREADS,threads);
		printf("\n" );
	}
		start = clock();
		double pi_gpu_dp = GPU_parallel(dev_dp,devStates, host_dp);
		stop = clock();

		if(n>0){
			printf("CUDA estimate of PI = %f [error of %f]\n", pi_gpu_dp, pi_gpu_dp - PI);
			printf("GPU pi calculated in %f ms.\n", 1000* (stop-start)/(double)CLOCKS_PER_SEC);
			printf("\n" );
	}
		start = clock();
		double pi_cpu_dp = host_monte_carlo_dp(BLOCKS * THREADS * N[n]);
		stop = clock();
		if(n>0){
			printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu_dp, pi_cpu_dp - PI);
			printf("CPU pi calculated in %f ms.\n",1000* (stop-start)/(double)CLOCKS_PER_SEC);
			printf("\n" );
	}
		// PI calculated from CPU parallel computation
		double pi_cpu_par_dp =0.0;
		start = clock();

		 pi_cpu_par_dp = CPU_parallel_dp(threads,N[n]);

		stop = clock();
		if(n>0){
			printf("CPU parallel estimate of PI = %f [error of %f]\n", pi_cpu_par_dp, pi_cpu_par_dp - PI);
			printf("CPU parallel pi calculated in %f ms.With Thread count of %i\n",1000* (stop-start)/(double)CLOCKS_PER_SEC, threads);
			printf("\n" );
	}
}
else{
	return 0;
}
	}
	return 0;
}
