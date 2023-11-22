#include "ex2.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <omp.h>

double randomD(int min, int max, int prec){ 
  prec = 10 * prec; 
  return (rand() % (max * prec - min * prec + 1) + min * prec) / (double)prec; 
}

void populateMatrix(double *A, int n, int prec){
  for(int i = 0; i < n*n; i++){
    A[i] = randomD(0, 100, prec);
  }
}

uint64_t matT (const double *A, double *B, int n){
  int r, c;
  struct timespec start, end;
  
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (r = 0; r < n; r++){
    for (c = 0; c < n; c++){
      B[c*n+r] = A[r*n+c];
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
}

uint64_t matBlockT(const double *A, double *B, int n, int bs){
  struct timespec start, end;
  int r, c;       //Row and column inside the block
  int br, bc;     //Row and column of the block
  int rA, cA;     //Row and column of the element in A
  int rB, cB;     //Row and column of the element in B
  int N_B = n/bs; //Number of blocks

  clock_gettime(CLOCK_MONOTONIC, &start);
  //Block transpose: transpose each block internally
  for (br = 0; br < N_B; ++br){
      for(r = 0; r < bs; ++r){
    for(bc = 0; bc < N_B; ++bc){
        for(c = 0; c < bs; ++c){  
          rA = br*bs + r;
          cA = bc*bs + c;
          rB = bc*bs + c;
          cB = br*bs + r;
          B[rB*n+cB] = A[rA*n+cA];
        }
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
}

///////////// PARALLEL VERSIONS ///////////////
uint64_t matTpar(const double* A, double* __restrict B, int n){
  // int r, c;

#ifndef _OPENMP
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int r = 0; r < n; r++){
    for (int c = 0; c < n; c++){
      B[c*n+r] = A[r*n+c];
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  return ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000);
#else
  #ifdef DEBUG
    printf("Doing TRUE parallel transpose\n");
  #endif
  double start = omp_get_wtime();
  #pragma omp parallel for collapse(2)
  for (int r = 0; r < n; r++){
    for (int c = 0; c < n; c++){
      B[c*n+r] = A[r*n+c];
    }
  }
  return (omp_get_wtime() - start) * 1000000;
#endif
}

uint64_t matBlockTpar(const double *A, double* __restrict B, int n, int bs){
#ifndef _OPENMP
  struct timespec start, end;
  int r, c;       //Row and column inside the block
  int br, bc;     //Row and column of the block
  int rA, cA;     //Row and column of the element in A
  int rB, cB;     //Row and column of the element in B
  int N_B = n/bs; //Number of blocks

  clock_gettime(CLOCK_MONOTONIC, &start);
  //Block transpose: transpose each block internally
  for (br = 0; br < N_B; ++br){
    for(r = 0; r < bs; ++r){
      for(bc = 0; bc < N_B; ++bc){
        for(c = 0; c < bs; ++c){  
          rA = br*bs + r;
          cA = bc*bs + c;
          rB = bc*bs + c;
          cB = br*bs + r;
          B[rB*n+cB] = A[rA*n+cA];
        }
      }
    }
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000;
#else
  int r, c;       //Row and column inside the block
  int br, bc;     //Row and column of the block
  int rA, cA;     //Row and column of the element in A
  int rB, cB;     //Row and column of the element in B
  int N_B = n/bs; //Number of blocks
  #ifdef DEBUG
    printf("Doing TRUE parallel transpose\n");
  #endif
  double start = omp_get_wtime();
  #pragma omp parallel for collapse(4)
  for (br = 0; br < N_B; ++br){
    for(r = 0; r < bs; ++r){
      for(bc = 0; bc < N_B; ++bc){
        for(c = 0; c < bs; ++c){  
          rA = br*bs + r;
          cA = bc*bs + c;
          rB = bc*bs + c;
          cB = br*bs + r;
          B[rB*n+cB] = A[rA*n+cA];
        }
      }
    }
  }
  return (omp_get_wtime() - start) * 1000000;
#endif
}