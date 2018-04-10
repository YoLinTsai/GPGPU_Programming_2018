#include "lab1.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "SyncedMemory.h"
using namespace std;

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 480;

struct cuComplex {
    float   r;
    float   i;
    __host__ __device__ cuComplex(float a, float b) : r(a), i(b)  {}
    __device__ float magnitude2(void) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

struct Lab1VideoGenerator::Impl {
	int t = 0;
    float zoom = 1.0;
};

const float incr =0.0000001;
const float inci =-0.0005;
const float startr=-0.7;
const float starti=0.35;

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

__device__ int julia(int x, int y, cuComplex c0, int iter,float zoom) {
    float jx = (((float)x-W/2) / (W*zoom/2));
    float jy = (((float)y-H/2) / (H*zoom/2));
    cuComplex a(jx, jy);
    int i = 0;
    for(i = 0; i < (iter+2)/2 ; i++)
    {
        a = a * a + c0;
        if(a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}


__global__ void kernel(unsigned char *ptr, cuComplex c0, int iter, float zoom) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    int juliaValue = julia(x, y, c0, iter, zoom);
    ptr[offset] = 255 * juliaValue;
}



void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	// cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	unsigned char *dev_bitmap;
	cudaMalloc((void**)&dev_bitmap, W*H);
	dim3 grid(W,H);
    if((impl->t)<240)
    {
        cuComplex c(startr,starti);
        kernel<<<grid,1>>>(dev_bitmap, c, (impl->t), (impl->zoom));
        cudaMemcpy(yuv, dev_bitmap, W*H, cudaMemcpyDeviceToHost);
        cudaMemset(yuv+W*H, 128, W*H/2);
    }
    else
    {
        cuComplex c(startr+((impl->t)-240)*incr,starti+((impl->t)-240)*inci);
        kernel<<<grid,1>>>(dev_bitmap, c, 240, 5.8);
        cudaMemcpy(yuv, dev_bitmap, W*H, cudaMemcpyDeviceToHost);
        cudaMemset(yuv+W*H, 128, W*H/2);
    } 

    //free
	++(impl->t);
    (impl->zoom) = (impl->zoom) + 0.02;
}
