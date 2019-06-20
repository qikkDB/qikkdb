#pragma once
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
// Errors and debug
void _cudaErrorCheck(const char *file, int line);

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define cudaErrorCheck()  { _cudaErrorCheck(__FILE__, __LINE__); }
