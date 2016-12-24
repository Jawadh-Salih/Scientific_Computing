// Source: http://docs.nvidia.com/cuda/curand/index.html

#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <curand_kernel.h>

#include <iostream>
#include <iomanip>

#define PI 3.1415926535  // known value of pi

// we could vary M & N to find the perf sweet spot

struct estimate_pi :
    public thrust::unary_function<unsigned int, float>
{

  __device__
  float operator()(unsigned int thread_id)
  {
    float sum = 0;
    unsigned int N = 8192; // samples per thread 2^13

    unsigned int seed = thread_id;

    curandState s;

    // seed a random number generator
    curand_init(seed, 0, 0, &s);

    // take N samples in a quarter circle
    for(unsigned int i = 0; i < N; ++i)
    {
      // draw a sample from the unit square
      float x = curand_uniform(&s);
      float y = curand_uniform(&s);

      // measure distance from the origin
      float dist = sqrtf(x*x + y*y);

      // add 1.0f if (u0,u1) is inside the quarter circle
      if(dist <= 1.0f)
        sum += 1.0f;
    }
    // std::cout << thread_id << std::endl;
    // multiply by 4 to get the area of the whole circle
    sum *= 4.0f;

    // divide by N
    return sum / N;
  }
};

//double precision
struct estimate_pi_dp :
    public thrust::unary_function<unsigned int, double>
{

  __device__
  double operator()(unsigned int thread_id)
  {
    double sum = 0;
    unsigned int N = 8192; // samples per thread 2^13

    unsigned int seed = thread_id;

    curandState s;

    // seed a random number generator
    curand_init(seed, 0, 0, &s);

    // take N samples in a quarter circle
    for(unsigned int i = 0; i < N; ++i)
    {
      // draw a sample from the unit square
      double x = curand_uniform(&s);
      double y = curand_uniform(&s);

      // measure distance from the origin
      double dist = sqrtf(x*x + y*y);

      // add 1.0f if (u0,u1) is inside the quarter circle
      if(dist <= 1.0f)
        sum += 1.0f;
    }
    // std::cout << thread_id << std::endl;
    // multiply by 4 to get the area of the whole circle
    sum *= 4.0f;

    // divide by N
    return sum / N;
  }
};

int main(void)
{
  // use 30K independent seeds

  int M[4] = {1,2048,8192,32768};
  clock_t start,stop;


  for(int n=0;n<4;n++){
    if(n>0){
    std::cout << "For M = " << M[n] << " and N = " << 8192 << std::endl;
    printf("==============================SINGLE PRECISION================================\n");
  }
    start = clock();
    float estimate = thrust::transform_reduce(
          thrust::counting_iterator<int>(0),
          thrust::counting_iterator<int>(M[n]),
          estimate_pi(),
          0.0f,
          thrust::plus<float>());
    estimate /= M[n];
    stop = clock();
    double exe_time_cu_thrust = 1000* (stop-start)/(double)CLOCKS_PER_SEC;
    if (n>0){
    std::cout << std::setprecision(6);
    std::cout << "CUDA Estimate of PI =" << estimate <<"[error of " << (estimate-PI) << "]" << std::endl;
    std::cout << "CUDA Estimated value of PI is  calculated in " << exe_time_cu_thrust <<" ms "<< std::endl;

    printf("==============================DOUBLE PRECISION================================\n");
    }
    start = clock();
    double estimate_dp = thrust::transform_reduce(
          thrust::counting_iterator<int>(0),
          thrust::counting_iterator<int>(M[n]),
          estimate_pi_dp(),
          0.0f,
          thrust::plus<double>());
    estimate_dp /= M[n];
    stop = clock();
    double exe_time_cu_thrust_dp = 1000* (stop-start)/(double)CLOCKS_PER_SEC;
    if(n>0){
    std::cout << std::setprecision(6);
    std::cout << std::endl;
    std::cout << "CUDA Estimate of PI =" << estimate_dp <<"[error of " << (estimate_dp-PI) << "]" << std::endl;
    std::cout << "CUDA Estimated value of PI is  calculated in " << exe_time_cu_thrust_dp <<" ms "<< std::endl;
  }
  }
  return 0;
}
