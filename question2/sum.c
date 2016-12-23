#include <stdio.h>
#include <string.h>

#ifndef -p
#define  P 200
#endif
int main(int argc, char *argv[] ){

  int i=0;
  double sum = 0.0;
  int val = 1000000000;
  char * s = argv[1];
  if(strcmp(argv[1],"-p") == 0){//} && argv[2] == "8" && argv[3] == "-v"){
    // now verify the CPU version answers
    printf("%s\n",s );
  }
  // for(i=0;i<val;i++){
  //   sum += 4/(1+ ((i+0.5)/val)*((i+0.5)/val));
  // }

  // printf("Sum is %.20f",sum/val);
}
