/*                      INTRODUCTION TO PARALLEL COMPUTING - Prof. VELLA 

    Student: Alan Masutti - 226901 - alan.masutti@studenti.unitn.it

    Assignment 2 - Exercise 2:
      This exercise consists of creating a program for parallel matrix transposition using OpenMP. The
      program must efficiently handle matrices of real numbers. The tasks to be performed are:
      - T1: Serial matrix transposition: [DONE]
            Implement a serial matrix transpose algorithm for matrices of real numbers. Define a function
            matT that takes an input matrix A and returns its transpose. Define a second function matBlockT
            that takes an input matrix A and returns its transpose, calculating the transpose in blocks as
            shown in the picture.
      - T2: Parallel matrix transpose:
            Parallelize the matrix transpose operation using OpenMP for matrices of real numbers. Create a
            matTpar function that takes the same input as the serial version and parallelizes the algorithm.
            Experiment with OpenMP directives to improve performance. Explore different algorithms to
            find the most efficient one.
      - T3: Parallel transpose of block matrices:
            Implement a parallel matrix transpose by dividing the matrix into blocks and transposing these
            blocks. Create a function matBlockTpar that takes the same input as the serial version, divides
            the matrix into blocks, and computes its transpose using the block division. Each block must be
            transposed, and the final transposed matrix is constructed from these blocks. Experiment with
            different block sizes and different OpenMP directives to optimize performance.
      - T4: Performance analysis:
            Evaluate the performance and scalability of both parallel matrix transpose methods (with blocks
            and without blocks). Calculate the speedup and efficiency gains for different matrix sizes and
            thread counts. Consider for this task the time of the routine and compute the bandwidth as performance metrics. 
            Identify possible bottlenecks or problems and propose optimizations. 
            Compare the performance of normal transpose and block-based transpose. Discuss
            the differences that may appear in parallelization algorithms when you are working with dense
            or sparse matrices.

    Notes:

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>

#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <omp.h>

#include "ex2.h"

#ifndef COMPILATION_NOTES
    #define COMPILATION_NOTES ""
#endif

#define RUN_NOTES "omp-for_block"
#ifndef RUN_DESCRIPTION 
  #define RUN_DESCRIPTION "Using #omp parallel for collapse(2) for matTpar"
#endif

int main(int argc, char const *argv[]){
  char hostbuffer[256] = "";
  int hostname;
  uint32_t n, bs;

  //Retriving some info about the machine
  // retrieve hostname
  hostname = gethostname(hostbuffer, sizeof(hostbuffer));
  if (hostname == -1) {
    printf("Error when getting hostname\n");
  }

  //Save running information:
  FILE *infoFile = fopen("./results/infoFileBlockT.csv", "r");
  if(infoFile == NULL){
    #ifdef PRINT
      printf("Creating infoFileBlockT.csv file\n");
    #endif
    infoFile = fopen("./results/infoFileBlockT.csv", "w");
    //hostname,run_notes,run_description,compilation_time,compilation_date
    fprintf(infoFile, "%s,%s,%s,%s,%s\n", hostbuffer, RUN_NOTES, RUN_DESCRIPTION,__TIME__, __DATE__);
    fclose(infoFile);
  }    

  //opening file to write results
  FILE *matBlockTFile = fopen("./results/matBlockTFile.csv", "r");
  if(matBlockTFile == NULL){
    #ifdef PRINT  
      printf("Creating matBlockTFile.csv file\n");
    #endif
    matBlockTFile = fopen("./results/matBlockTFile.csv", "w");
    if(matBlockTFile == NULL){
      printf("Error when opening matBlockTFile.csv file\n");
      exit(1);
    }
    fprintf(matBlockTFile, "matrix_size,block_size,matBlockT_wallTime[us],matBlockTpar_wallTime[us],hostname,compilation_notes,run_notes\n");
  }else{
    fclose(matBlockTFile);
    matBlockTFile = fopen("./results/matBlockTFile.csv", "a");
  }

  //getting sizes from command line
#ifndef N
  do{
    printf("Insert matrix size: ");
    scanf("%d", &n);
  }while(n <= 0);
#else
  n = N;
#endif

#ifndef BS
  do{
    printf("Insert block size: ");
    scanf("%d", &bs);
  }while(n % bs != 0);
#else
  bs = BS;
#endif

  //Execution of the serial transpose
#ifdef PRINT
  printf("Doing serial transpose\n");
#endif
  double *A = (double *)malloc(n*n*sizeof(double));
  double *B = (double *)malloc(n*n*sizeof(double));
  if(A == NULL || B == NULL){
    printf("Error when allocating memory\n");
    return (-1);
  }

  //Execution of the block transpose
  populateMatrix(A, n, 1);
#ifdef PRINT
  printf("Doing serial block transpose\n");
#endif
  uint64_t timeBlock = matBlockT(A, B, n, bs);
#ifdef PRINT
  printf("Serial block transpose done. Wall Time: \t%ld us\n", timeBlock);
#endif

#ifdef DEBUG
  printf("A:\n");
  for(int i = 0; i < n*n; i++){
    printf("%f ", A[i]);
    if((i+1) % n == 0){
      printf("\n");
    }
  }
  printf("A Transpose by blocks:\n");
  for(int i = 0; i < n*n; i++){
    printf("%f ", B[i]);
    if((i+1) % n == 0){
      printf("\n");
    }
  }
#endif

  //Execution of the parallel block transpose
#ifdef PRINT
  printf("Doing parallel block transpose\n");
#endif
  populateMatrix(A, n, 1);
  uint32_t timeBlockPar = matBlockTpar(A, B, n, bs);
#ifdef PRINT
  printf("Parallel block transpose done. Wall Time: \t%ld us\n", timeBlockPar);
#endif

#ifdef DEBUG
  printf("A:\n");
  for(int i = 0; i < n*n; i++){
    printf("%f ", A[i]);
    if((i+1) % n == 0){
      printf("\n");
    }
  }
  printf("A Transpose by blocks in parallel:\n");
  for(int i = 0; i < n*n; i++){
    printf("%f ", B[i]);
    if((i+1) % n == 0){
      printf("\n");
    }
  }
#endif

  //Exporting results
  fprintf(matBlockTFile, "%d,%d,%ld,%ld,%s,%s,%s\n", n, bs, timeBlock, timeBlockPar, hostbuffer, COMPILATION_NOTES, RUN_NOTES);

  //Closing results files
  fclose(matBlockTFile);

  //Release memory
  free(A);
  free(B);
}
