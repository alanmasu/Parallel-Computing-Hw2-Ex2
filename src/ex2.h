#ifndef __EX2_H__
#define __EX2_H__
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

double randomD(int min, int max, int prec);
void populateMatrix(double *A, int n, int prec);

uint64_t matT (const double *A, double *B, int n);
uint64_t matBlockT(const double *A, double *B, int n, int bs);

///////////// PARALLEL VERSIONS ///////////////
uint64_t matTpar(const double* A, double* __restrict B, int n);
uint64_t matBlockTpar(const double *A, double* __restrict B, int n, int bs);

#endif
