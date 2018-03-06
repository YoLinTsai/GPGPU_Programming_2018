#include <cstdio>
#include <cstdlib>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SyncedMemory.h"

#define CHECK {\
	auto e = cudaDeviceSynchronize();\
	if (e != cudaSuccess) {\
		printf("At " __FILE__ ":%d, %s\n", __LINE__, cudaGetErrorString(e));\
		abort();\
	}\
}

const int W = 40;
const int H = 13;

__global__ void Draw(char *frame) {
	// TODO: draw more complex things here
	// Do not just submit the original file provided by the TA!
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y < H && x < W) {
		char c;
		if (x == W - 1) {
			c = y == H - 1 ? '\0' : '\n';
		}
		else if (y == 0 || y == H - 1 || x == 0 || x == W - 2) {
			c = ':';
		}
		else if (y == 1 && ((x <= 14 && x >= 5) || (x >= 17 && x <= 19) || (x >= 24 && x <= 28))) {
			c = '8';
		}
		else if (y == 2 && (x==4 ||x==5 || x == 11 || x == 12 || x == 17 || x == 16 || x == 20 || x == 19 || x == 25 || x == 24 || x == 29 || x == 28 )) { 
			c = '8';
		}
		else if (y == 3 && (x == 5 || x == 6 || x == 7 || x == 8 || x == 12 || x == 11 || x == 16 || x == 15 || x == 21 || x == 20 || (x >= 24 && x <= 28))) {
			c = '8';
		}
		else if (y == 4 && (x == 8 || x == 9 || x == 11 || x == 12 || (x <= 22 && x >= 14) || x == 24 || x == 25 || x == 29 || x == 30)) {
			c = '8';
		}
		else if (y == 5 && ((x <= 9 && x >= 1) || x == 11 || x == 12 || x == 14 || x == 15 || x == 21 || x == 22 || x == 24 || x == 25 || (x >= 30 && x <= 35))) {
			c = '8';
		}
		else if (y == 7 && (x == 1 || x == 2 || x == 5 || x == 6 || x == 9 || x == 10 || x == 14 || x == 15 || x == 16 || x == 21 || x == 22 || x == 23 || x == 24 || x == 25 || (x >= 30 && x <= 35))) {
			c = '8';
		}
		else if (y == 8 && (x == 1 || x == 2 || x == 5 || x == 6 || x == 9 || x == 10 || x == 13 || x == 14 || x == 16 || x == 17 || x == 21 || x == 22 || x == 25 || x == 26 || x == 29 || x == 30)) {
			c = '8';
		}
		else if (y == 9 && (x == 1 || x == 2 || x==4||x == 5 || x == 6||x==7 || x == 9 || x == 10 || x == 12 || x == 13 || x == 17 || x == 18 || x == 21 || x == 22 ||x==23||x==24|| x == 25 || (x>=30&&x<=33))) {
			c = '8';
		}
		else if (y == 10 && (x==2||x==3||x==4||x==7||x==8||x==9||(x>=11&x<=19)||x==21||x==22||x==26||x==27||x==33||x==34)) {
			c = '8';
		}
		else if (y == 11 && (x==3||x==4||x==7||x==8||x==11||x==12||x==18||x==19||x==21||x==22||(x>=27&&x<=33))) {
			c = '8';
		}
		else {
			c = ' ';
		}
		frame[y*W + x] = c;
	}
}

int main(int argc, char **argv)
{
	MemoryBuffer<char> frame(W*H);
	auto frame_smem = frame.CreateSync(W*H);
	CHECK;

	Draw <<< dim3((W - 1) / 16 + 1, (H - 1) / 12 + 1), dim3(16, 12) >>> (frame_smem.get_gpu_wo());
	CHECK;

	puts(frame_smem.get_cpu_ro());
	CHECK;
	return 0;
}