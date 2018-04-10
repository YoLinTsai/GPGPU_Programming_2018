#include "lab1.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "SyncedMemory.h"
using namespace std;

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 480;

struct Complex {
    float   r;
    float   i;
    __host__ __device__ Complex(float a, float b)
    {
        r=a;
        i=b;
    }
    __host__ __device__ float magnitude2(void) 
    {
        return r * r + i * i;
    }
    __device__ Complex operator*(const Complex& a) 
    {
        return Complex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ Complex operator+(const Complex& a) 
    {
        return Complex(r + a.r, i + a.i);
    }
};

struct Lab1VideoGenerator::Impl {
	int t = 2;
    float zoom = 1.0;
};

const float incr =-0.0000001;
const float inci =0.00005;
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

__device__ int julia(int x, int y, Complex c0, int iter,float zoom) {
    float jx = (((float)x-W/2) / (W*zoom/2));
    float jy = (((float)y-H/2) / (H*zoom/2));
    Complex a(jx, jy);
    int i = 1;
    for(i = 1; i < (iter+22)/2 ; i++)
    {
        a = a * a + c0;
        if(a.magnitude2() > 1000)
        {
            return i;
        }
    }
    return 0;
}


__global__ void kernel(unsigned char *ptr, Complex c0, int iter, float zoom) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;
    int Ysize = gridDim.x * gridDim.y;

    int rTime = julia(x, y, c0, iter, zoom);
    float per = ((float)rTime/((float)iter+10));
    if(rTime==0)
    {
        ptr[offset] = 0;
        if(x%2==0 && y%2 ==0)
        {
            ptr[Ysize+x/2+y*gridDim.x/4] = 128;
            ptr[Ysize+Ysize/4+x/2+y*gridDim.x/4] = 128;
        }
    }
    else
    {
        /*
        if(rTime>=1&&rTime<=150)
        {
            ptr[offset] = 255;
            if(x%2==0 && y%2==0)
            {
                ptr[Ysize+x/2+y*gridDim.x/4] = 128;
                ptr[Ysize+Ysize/4+x/2+y*gridDim.x/4] = 128;
            }
        }
        else
        {
            ptr[offset] = 0;
            if(x%2==0 && y%2==0)
            {
                ptr[Ysize+x/2+y*gridDim.x/4] = 128;
                ptr[Ysize+Ysize/4+x/2+y*gridDim.x/4] = 128;
            }
        }
        */
        if(per>0.5)
        {
            float nper = (1-per)/0.5;
            float nperc = 1-nper;
            ptr[offset] = nper*(0.299*250+0.578*200);
            if(x%2==0 && y%2==0)
            {
                ptr[Ysize+x/2+y*gridDim.x/4] = nper*(-0.169*250-0.331*200)+128;
                ptr[Ysize+Ysize/4+x/2+y*gridDim.x/4] = nper*(0.5*250-0.419*200)+128;
            }
        }
        else
        {
            float nper = (0.5-per)/0.5;
            float nperc = 1-nper;
            ptr[offset] = nperc*(0.299*250+0.578*200)+nper*255;
            if(x%2==0 && y%2==0)
            {
                ptr[Ysize+x/2+y*gridDim.x/4] = nperc*(-0.169*250-0.331*200)+128;
                ptr[Ysize+Ysize/4+x/2+y*gridDim.x/4] = nperc*(0.5*250-0.419*200)+128;
            }
        }
    }
}

void Lab1VideoGenerator::Generate(uint8_t *yuv) {
	// cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	unsigned char *ComplexMap;
	cudaMalloc((void**)&ComplexMap, W*H*3/2);
	dim3 grid(W,H);
    if((impl->t)<240)
    {
        Complex c(startr,starti);
        kernel<<<grid,1>>>(ComplexMap, c, (impl->t), (impl->zoom));
        cudaMemcpy(yuv, ComplexMap, W*H*3/2, cudaMemcpyDeviceToHost);
        //cudaMemset(yuv+W*H, 128, W*H/2);
    }
    else
    {
        Complex c(startr+((impl->t)-240)*incr,starti+((impl->t)-240)*inci);
        kernel<<<grid,1>>>(ComplexMap, c, 240, 25);
        cudaMemcpy(yuv, ComplexMap, W*H*3/2, cudaMemcpyDeviceToHost);
        //cudaMemset(yuv+W*H, 128, W*H/2);
    } 

    //free
	++(impl->t);
    (impl->zoom) = (impl->zoom) + 0.1;
}
