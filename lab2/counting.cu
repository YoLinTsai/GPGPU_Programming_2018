#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct newline_Check{
    __host__ __device__
        bool operator()(const char x) { return !( x == '\n'); }
};

void CountPosition1(const char *text, int *pos, int text_size)
{
    thrust::device_vector<int> tmp(text_size);
    thrust::device_ptr<const char> text_b(text), text_e(text+text_size);
    thrust::device_ptr<int> pos_b(pos);
    thrust::transform(text_b,text_e, tmp.begin(), newline_Check());
    thrust::inclusive_scan_by_key(tmp.begin(), tmp.end(), tmp.begin(), pos_b);
    return;
}



void CountPosition2(const char *text, int *pos, int text_size)
{

}
