
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <cuda.h>
// #include <math.h>
float *V1sp,*V2sp; // Single precision
double *V1dp,*V2dp; // double precision
unsigned long N=100; // Dimension of a vector.

float *   generate_vector_sp(float *V, unsigned long N){
  V = (float *)malloc(sizeof(float));
  for(int i=0;i<N;i++){
    V[i] = (float)rand()/(float)RAND_MAX +1;
    //printf("%f\n",V[i] );
  }
  return V;
}
double *   generate_vector_dp(double *V, unsigned long N){
  V = (double *)malloc(sizeof(double));
  for(int i=0;i<N;i++){
    V[i] = (double)rand()/(double)RAND_MAX +1;
    // printf("%.8f\n",V[i] );
  }
  return V;
}


float vector_dot_sp(float *V1,float *V2, unsigned long N){
  float dot_product = 0.0f;
  for(long i=0;i<N;i++){
    dot_product += V1[i]*V2[i];
  }

  return dot_product;
}
double vector_dot_dp(double *V1,double *V2, unsigned long N){
  double dot_product = 0.0f;
  for(long i=0;i<N;i++){
    dot_product += V1[i]*V2[i];
  }

  return dot_product;
}

int main(){

  printf("========================================================================\n");
  printf("=====================Vector Dot Product calculation=====================\n");
  printf("========================================================================\n");

  printf("Please enter a Size of a Vector :");
  scanf("%lu ",&N);
  printf("\nFor Single precision enter f, Double precision enter d : ");
  char precision = 'a';
  scanf("%c",&precision);
  unsigned seed = (time(NULL));
  srand(seed);
  switch (precision) {
    case 'f'/* value */:
      V1sp = generate_vector_sp(V1sp,N);
      V2sp = generate_vector_sp(V2sp,N);
      float dotP_sp = vector_dot_sp(V1sp,V2sp,N);
      printf("%f\n",dotP_sp );
       break;
    case 'd':
      V1dp = generate_vector_dp(V1dp,N);
      V2dp = generate_vector_dp(V2dp,N);
      double dotP_dp = vector_dot_dp(V1dp,V2dp,N);
      float dotP_sp1 = dotP_dp;
      // printf("%.20f\n",dotP_dp );
      printf("%lf\n",dotP_dp );
      // printf("%f\n",dotP_sp1 );
        break;
    default:
      printf("Something goes Wrong !!\n");
      break;
    }
    float answer_p = 0.0;
    int num_threads = 8;
    #pragma omp parallel num_threads(num_threads)
    {
      #pragma omp for schedule(static) reduction(+:answer_p)
      for (int i = 0; i < N; i++) {
          answer_p = answer_p + (V1sp[i] * V2sp[i]);
      }
    }

    printf(" parellel dot %f \n", answer_p);
    free(V1sp);free(V2sp);free(V1dp);free(V2dp);


  // V1dp = generate_vector_dp(V1dp,N);
  // V2dp = generate_vector_dp(V2dp,N);
  //


}
