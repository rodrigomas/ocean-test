#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math_constants.h>

#define PI 3.1415926536f

__global__ void KernelPerlinCUDA_H(unsigned char* surface, float *noise, int res, int width, int height, float Overcast, size_t pitch, float t)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float* pixel;

	if (x >= width || y >= height) return;

	pixel = (float*)(surface + y*pitch) + 4*x;

	// populate it
	float2 move;
	move.x = 0.0 * t;
	move.y = 1.0 * t;
	float perlin = 0.0;

	int px, py;

	px = (int)(x + move.x) % res; py = (int)(y + move.y) % res;	
	perlin =  noise[ py * res + px] / 2.0;
	px = (int)(x*2.0 + move.x) % res; py = (int)(y*2.0 + move.y) % res;	
	perlin +=  noise[ py * res + px] / 4.0;
	px = (int)(x*4.0 + move.x) % res; py = (int)(y*4.0 + move.y) % res;	
	perlin +=  noise[ py * res + px] / 8.0;
	px = (int)(x*8.0 + move.x) % res; py = (int)(y*8.0 + move.y) % res;	
	perlin +=  noise[ py * res + px] / 16.0;
	px = (int)(x*16.0 + move.x) % res; py = (int)(y*16.0 + move.y) % res;	
	perlin +=  noise[ py * res + px] / 32.0;
	px = (int)(x*32.0 + move.x) % res; py = (int)(y*32.0 + move.y) % res;	
	perlin +=  noise[ py * res + px] / 32.0;

	pixel[0] = 1.0 - pow(perlin, Overcast) * 2.0; // red
	pixel[1] = pixel[0]; // green
	pixel[2] = pixel[0]; // blue
	pixel[3] = 1; // alpha
}

extern "C" 
void PerlinCUDA_H(void* surface, void* noise, int res, int width, int height, float Overcast, size_t pitch, float t)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3( 16, 16 ); // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3( (width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y );

	KernelPerlinCUDA_H<<<Dg,Db>>>( (unsigned char*)surface, (float*)noise, res, width, height, Overcast,pitch, t );

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("PerlinCUDA_H() failed to launch error = %d\n", error);
	}
}

__device__
float3 cross(float3 a, float3 b)
{
	return make_float3(a.y * b.z - (b.y*a.z), a.z * b.x - (b.z*a.x), a.x * b.y - (b.x*a.y));
}

__device__
float3 normalize(float3 a)
{
	float len = sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
	len = (len > 0.0) ? len : 1.0;
	return make_float3(a.x/len, a.y/len, a.z/len);
}

__global__ void KernelPerlinCUDA_N(unsigned char* surface, unsigned char *normals, int width, int height, size_t pitch, float t )
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float* pixel;
	float *B, *C, *A ;

	if (x >= width || y >= height) return;

	pixel = (float*)(normals + y*pitch) + 4*x;

	A = (float*)(surface + y*pitch) + 4*x;
	B = (float*)(surface + ((y+1) % height)*pitch) + 4*x;
	C = (float*)(surface + y*pitch) + 4*((x+1)%width);
	
	//__syncthreads();

	float dx = A[0] - C[0];
	float dz = A[0] - B[0];

	float3 T = make_float3( 1,0,dz );
	T = normalize(T);

	float3 Bi = make_float3( 1,0,dx );
	Bi = normalize(Bi);

	float3 N = cross(Bi,T);

	pixel[0] = N.x; // red
	pixel[1] = N.y; // green
	pixel[2] = N.z; // blue
	pixel[3] = 1; // alpha			

}

extern "C" 
void PerlinCUDA_N(  float* surface, float* normals, unsigned int width, unsigned int height, size_t pitch, float t)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3( 16, 16 ); // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3( (width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y );

	KernelPerlinCUDA_N<<<Dg,Db>>>( (unsigned char*)surface, (unsigned char*)normals, width, height, pitch, t );

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("PerlinCUDA_H() failed to launch error = %d\n", error);
	}
}
