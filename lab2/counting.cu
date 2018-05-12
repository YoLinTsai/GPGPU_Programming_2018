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

__global__ void build_indexTable_of_BIT(int *indexTable, int text_size, int height)
{
    int size_sum = 0;
    int temp = text_size;
    for(int i=0;i<height;i++)
    {
        indexTable[i] = size_sum;
        size_sum += temp;
        temp /= 2;
    }
}

__global__ void build_BIT(int *BIT, int *indexTable, const char *text, int height) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(height == 0)
    {
        //check range
        if(idx < indexTable[1])
        {
            if(text[idx] == '\n')
                BIT[idx] = 0;
            else
                BIT[idx] = 1;
        }
    }
    else
    {
        int now_idx = idx + indexTable[height];
        int check_idx = idx * 2 + indexTable[height-1];
        if(now_idx < indexTable[height+1] && check_idx + 1 < indexTable[height])
        {
            if(BIT[check_idx] == 1 && BIT[check_idx+1] == 1)
                BIT[now_idx] = 1;
        }
    }
}

__global__ void count_pos(int *BIT, int *pos, int *indexTable)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx>indexTable[1]) 
        return;

    if(BIT[idx] == 0)
    {
        pos[idx] = 0;
    } 
    else
    {
        int length = 0, index = idx, height = 0, add_length = 1;
        while(true)
        {
            if(index<=0)
                break;
            if(index%2 == 0)
            {
                index -= 1;
                length += add_length;
            }
            else
            {
                int parent_index = (index / 2) + indexTable[height + 1];
                if(BIT[parent_index] == 1)
                {
                    add_length *= 2;
                    height += 1;
                    index = index / 2;
                }
                else
                {
                    break;
                }
            }
        }
        while(index >= 0 && add_length > 0 && height >= 0)
        {
            int now_index = index + indexTable[height];
            if(BIT[now_index] == 1)
            {
                length += add_length;
                index = index * 2 - 1;
                height -= 1;
            }
            else
            {
                index = index * 2 + 1;
                height -= 1;
            }
            add_length /= 2;
        }
        pos[idx] = length;
    }
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    int threads = 32;
    int blocks = CeilDiv(text_size, threads);
    int *indexTable, *BIT;
    int maxheight = 10;

    cudaMalloc(&indexTable, sizeof(int) * (maxheight + 1));
    build_indexTable_of_BIT<<< 1,1 >>>(indexTable, text_size, maxheight + 1);
    
    int *c_indexTable;
    c_indexTable = (int*) malloc(sizeof(int) * (maxheight + 1));
    cudaMemcpy(c_indexTable, indexTable, sizeof(int) * (maxheight + 1), cudaMemcpyDeviceToHost);

    int treeSize = c_indexTable[10];
    cudaMalloc(&BIT, sizeof(int) * treeSize);
    cudaMemset(BIT, 0, sizeof(int) * treeSize);

    for(int i=0;i<maxheight;i++)
    {
        blocks = ceil((float) (c_indexTable[i+1] - c_indexTable[i]) / threads);
        build_BIT<<< blocks, threads >>> (BIT, indexTable, text, i);
    }

    blocks = ceil((float) text_size / threads);
    count_pos <<< blocks, threads >>> (BIT, pos, indexTable);
}