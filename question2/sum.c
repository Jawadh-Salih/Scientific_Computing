#include <stdio.h>

int main(int argc, char *argv){

  int i=0;
  double sum = 0.0;
  int val = 1000000000;
  for(i=0;i<val;i++){
    sum += 4/(1+ ((i+0.5)/val)*((i+0.5)/val));
  }

  printf("Sum is %.20f",sum/val);
}
