// Source: http://web.mit.edu/pocky/www/cudaworkshop/MonteCarlo/Pi.cu

// Written by Barry Wilkinson, UNC-Charlotte. Pi.cu  December 22, 2010.
//Derived somewhat from code developed by Patrick Rogers, UNC-C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
// #include <curand_kernel.h>
#include <omp.h> // OpenMP included.


#define TRIALS 268435456

#define PI 3.1415926535  // known value of pi



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
	clock_t start, stop;
	float host[256*256];


	start = clock();
	float pi_cpu = host_monte_carlo(TRIALS);
	stop = clock();
  printf("CPU pi calculated in %f s.\n", (stop-start)/(float)CLOCKS_PER_SEC);

  printf("CPU estimate of PI = %f [error of %f]\n", pi_cpu, pi_cpu - PI);
  // printf("CPU Multithreading estimate of PI = %f [error of %f]\n", pi_cpu_par, pi_cpu_par - PI);

	return 0;
}
