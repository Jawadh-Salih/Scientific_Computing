# Scientific_Computing
Scientific Computing Assignments are implemented

Please compile all the programs into the exe file. The format of the compilation code is as follows

nvcc -Xcompiler -fopenmp <path to .cu file> -o <path to .out file> -O3 #.out file will be created if there is not any such files.

eg:
  nvcc -Xcompiler -fopenmp question3/matrix_mul.cu -o exefiles/matrix_mul.out -O3

  - This will copile the matrix_mul.cu file and create an exectuable file for linux.

After the compilation run the files.

//Question 1 files.


./pi-curand_modified.out <# of threads>
./pi-curand-thrust_modified.out
./pi-myrand_modified.out <# of threads>
./pi-mystery_modified.out

//Question 2 files. Vector Dot Product.

//Question 3 files. Matrix Multiplication.
./matrix_mul.out                          : run the serial program
./matrix_mul.out -s                       : run the serial program
./matrix_mul.out -p -<# of threads>       : run the parallel program
./matrix_mul.out -p -<# of threads> -v    : run the parallel program
./matrix_mul.out -c                       : run the cuda program
