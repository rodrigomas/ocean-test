#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.1415926536f

#define SHARED_MEMORY
#define SHARED_MEMORY_CONST
#define CONST_MEMORY

__device__ __constant__ float ger_data[147];

__device__
float dotf(float3 a,float bx, float by, float bz)
{
	return a.x*bx + a.y*by + a.z*bz;
}

__global__ void KernelGerstnerCUDA_H(unsigned char* surface, int nwaves, float *data, int width, int height, size_t pitch, float t)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float* pixel;

	if (x >= width || y >= height) return;

#ifdef SHARED_MEMORY
	int cnt = max(1,7 * nwaves / (blockDim.x* blockDim.y)+1);
	__shared__ float local_data[147];	
	for(int i = 0; i< cnt; ++i)
	{
       int p = min(7 * nwaves-2, (threadIdx.x + threadIdx.y*blockDim.x)*cnt);
#ifdef SHARED_MEMORY_CONST
		local_data[p+i] = ger_data[p+i];
#else
		local_data[p+i] = data[p+i];
#endif
	}
	__syncthreads();
#endif

	pixel = (float*)(surface + y*pitch) + 4*x;

	float3 P0 = make_float3(x,0,y);
	float3 Position = make_float3(x,0,y);

	for(int i = 0; i< nwaves; ++i)
	{				
#ifdef CONST_MEMORY	
		//float Q = data[i*7];
		float FREQ = ger_data[i*7+1];
		float PHASE = ger_data[i*7+2];
		float AMP = ger_data[i*7+3];		
		float DX = ger_data[i*7+4];		
		float DY = ger_data[i*7+5];		
		float DZ = ger_data[i*7+6];	
#elif defined(SHARED_MEMORY)
		//float Q = data[i*7];
		float FREQ = local_data[i*7+1];
		float PHASE = local_data[i*7+2];
		float AMP = local_data[i*7+3];		
		float DX = local_data[i*7+4];		
		float DY = local_data[i*7+5];		
		float DZ = local_data[i*7+6];	
#else
			//float Q = data[i*7];
		float FREQ = data[i*7+1];
		float PHASE = data[i*7+2];
		float AMP = data[i*7+3];		
		float DX = data[i*7+4];		
		float DY = data[i*7+5];		
		float DZ = data[i*7+6];	
#endif
		float Q = 0.5;
		//float FREQ = 0.9;
		//float PHASE = 1.5;
		//float AMP = 2.0;		
		//float DX = 0.5;		
		//float DY = 0.0;		
		//float DZ = 0.7;			

		float angle = FREQ * dotf(P0,DX,DY,DZ) + PHASE * t; 

		float Si = sin(angle);
		float C = cos(angle);

		Position.x += Q * AMP * DX * C;
		Position.z += Q * AMP * DZ * C;
		Position.y += AMP * Si;
	}

	pixel[0] = Position.x;
	pixel[1] = Position.y;
	pixel[2] = Position.z;
	pixel[3] = 1;
}

extern "C" 
void GerstnerCUDA_H(void* surface, int nwaves, float *data, int width, int height, size_t pitch, float t)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3( 16, 16 ); // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3( (width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y );

	KernelGerstnerCUDA_H<<<Dg,Db>>>( (unsigned char*)surface, nwaves, data,  width, height, pitch, t );

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("GerstnerCUDA_H() failed to launch error = %d\n", error);
	}
}


__global__ void KernelGerstnerCUDA_N(unsigned char* surface, int nwaves, float *data, int width, int height, size_t pitch, float t)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float* pixel;

	if (x >= width || y >= height) return;

#ifdef SHARED_MEMORY
	__shared__ float local_data[147];
	for(int i = 0; i< nwaves; ++i)
	{
#ifdef SHARED_MEMORY_CONST
		local_data[i] = ger_data[i];
#else
		local_data[i] = data[i];
#endif
	}
	__syncthreads();
#endif

	pixel = (float*)(surface + y*pitch) + 4*x;

	float3 P0 = make_float3(x,0,y);
	float3 Position = make_float3(x,0,y);
	float3 N;

	for(int i = 0; i< nwaves; ++i)
	{				
#ifdef CONST_MEMORY	
		//float Q = data[i*7];
		float FREQ = ger_data[i*7+1];
		float PHASE = ger_data[i*7+2];
		float AMP = ger_data[i*7+3];		
		float DX = ger_data[i*7+4];		
		float DY = ger_data[i*7+5];		
		float DZ = ger_data[i*7+6];	
#elif defined(SHARED_MEMORY)
		//float Q = data[i*7];
		float FREQ = local_data[i*7+1];
		float PHASE = local_data[i*7+2];
		float AMP = local_data[i*7+3];		
		float DX = local_data[i*7+4];		
		float DY = local_data[i*7+5];		
		float DZ = local_data[i*7+6];	
#else
			//float Q = data[i*7];
		float FREQ = data[i*7+1];
		float PHASE = data[i*7+2];
		float AMP = data[i*7+3];		
		float DX = data[i*7+4];		
		float DY = data[i*7+5];		
		float DZ = data[i*7+6];	
#endif
		float Q = 0.5;
		//float FREQ = 0.9;
		//float PHASE = 1.5;
		//float AMP = 2.0;		
		//float DX = 0.5;		
		//float DY = 0.0;		
		//float DZ = 0.7;			

		float angle = FREQ * dotf(P0,DX,DY,DZ) + PHASE * t; 

		float Si = sin(angle);
		float C = cos(angle);

		Position.x += Q * AMP * DX * C;
		Position.z += Q * AMP * DZ * C;
		Position.y += AMP * Si;
	}

	for(int i = 0; i< nwaves; ++i)
	{				
#ifdef CONST_MEMORY	
		//float Q = data[i*7];
		float FREQ = ger_data[i*7+1];
		float PHASE = ger_data[i*7+2];
		float AMP = ger_data[i*7+3];		
		float DX = ger_data[i*7+4];		
		float DY = ger_data[i*7+5];		
		float DZ = ger_data[i*7+6];	
#elif defined(SHARED_MEMORY)
		//float Q = data[i*7];
		float FREQ = local_data[i*7+1];
		float PHASE = local_data[i*7+2];
		float AMP = local_data[i*7+3];		
		float DX = local_data[i*7+4];		
		float DY = local_data[i*7+5];		
		float DZ = local_data[i*7+6];	
#else
			//float Q = data[i*7];
		float FREQ = data[i*7+1];
		float PHASE = data[i*7+2];
		float AMP = data[i*7+3];		
		float DX = data[i*7+4];		
		float DY = data[i*7+5];		
		float DZ = data[i*7+6];	
#endif


		float Q = 0.5;
		//float FREQ = 0.9;
		//float PHASE = 1.5;
		//float AMP = 2.0;		
		//float DX = 0.5;		
		//float DY = 0.0;		
		//float DZ = 0.7;		

		float WA = FREQ * AMP;				

		float angle = FREQ * dotf(Position,DX,DY,DZ) + PHASE * t; 				

		float Si = sin(angle);
		float C = cos(angle);

		N.x -= DX * WA * C;		
		N.z -= DZ * WA * C;
		N.y -= Q * WA * Si;
	}

	pixel[0] = N.x;
	pixel[1] = N.y;
	pixel[2] = N.z;
	pixel[3] = 1;
}

extern "C" 
void GerstnerCUDA_N(void* surface, int nwaves, float *data, int width, int height, size_t pitch, float t)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3( 16, 16 ); // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3( (width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y );

	KernelGerstnerCUDA_N<<<Dg,Db>>>( (unsigned char*)surface, nwaves, data, width, height, pitch, t );

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("GerstnerCUDA_N() failed to launch error = %d\n", error);
	}
}
