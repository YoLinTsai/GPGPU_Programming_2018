#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
		}
	}
}

__global__ void PoissonFixedValue(
	const float *background,
	const float *target,
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt * yt + xt;
	const int yb = oy + yt;
	const int xb = ox + xt;
	
	if(yt < ht && xt < wt) {
		if(mask[curt] > 127)
		{
			const int shift[4][2] = {{-1, 0},{0, -1},{1, 0},{0, 1}};
			int target_NSWE[4], background_NSWE[4];
			bool check_NSWE[4] = {0};
			for(int i=0;i<4;i++)
			{
				target_NSWE[i] = wt * (yt + shift[i][0]) + xt + shift[i][1];
				background_NSWE[i] = wb * (yb + shift[i][0]) + xb + shift[i][1];
			}

			check_NSWE[0] = (yt != 0)? 1:0;
			check_NSWE[1] = (xt != 0)? 1:0;
			check_NSWE[2] = (yt < ht-1)? 1:0;
			check_NSWE[3] = (xt < wt-1)? 1:0;

			for(int rgb=0;rgb<3;rgb++)
			{
				float fixed_sum = 0.0;
				for(int i=0;i<4;i++)
				{
					if(check_NSWE[i])
					{
						fixed_sum += target[3*curt+rgb] - target[3*target_NSWE[i]+rgb];
					}
					if(!check_NSWE[i] || mask[target_NSWE[i]] < 127)
					{
						fixed_sum += background[3*background_NSWE[i]+rgb];
					}
				}
				fixed[3*curt+rgb] = fixed_sum;
			}
		}
		else
		{
			const int curb = wb*yb+xb;
			for(int rgb=0;rgb<3;rgb++)
			{
				fixed[3*curt+rgb] = background[3*curb+rgb];
			}
		}
	}
}

__global__ void PoissonIteration(
	const float *mask,
	float *fixed,
	float *buf1,
	float *buf2,
	const int wt, const int ht
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt * yt + xt;
	if(yt < ht && xt < wt && mask[curt] > 128)
	{
		const int shift[4][2] = {{-1, 0},{0, -1},{1, 0},{0, 1}};
		int target_NSWE[4];
		bool check_NSWE[4] = {0};

		for(int i=0;i<4;i++)
		{
			target_NSWE[i] = wt * (yt + shift[i][0]) + xt + shift[i][1];
		}
		check_NSWE[0] = (yt != 0)? 1:0;
		check_NSWE[1] = (xt != 0)? 1:0;
		check_NSWE[2] = (yt < ht-1)? 1:0;
		check_NSWE[3] = (xt < wt-1)? 1:0;

		for(int rgb=0;rgb<3;rgb++)
		{
			float temp_sum = 0.0;
			for(int i=0;i<4;i++)
			{
				if(check_NSWE[i])
				{
					if(mask[target_NSWE[i]]>127)
					{
						temp_sum += buf1[3*target_NSWE[i]+rgb];
					}
				}
			}
			buf2[3*curt+rgb] = (fixed[3*curt+rgb] + temp_sum) * 0.25;
		}
	}
}




void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	float *fixed, *buf1, *buf2;
	cudaMalloc(&fixed, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf1, 3*wt*ht*sizeof(float));
	cudaMalloc(&buf2, 3*wt*ht*sizeof(float));

	PoissonFixedValue<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
	);

	cudaMemcpy(buf1, target, 3*wt*ht*sizeof(float), cudaMemcpyDeviceToDevice);
	for(int i=0;i<10000;i++)
	{
		PoissonIteration<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(mask, fixed, buf1, buf2, wt, ht);
		PoissonIteration<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(mask, fixed, buf2, buf1, wt, ht);
	}

	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
	);
}
