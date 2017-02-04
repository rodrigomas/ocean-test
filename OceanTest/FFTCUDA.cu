///////////////////////////////////////////////////////////////////////////////
#include <cufft.h>
#include <math_constants.h>

//Round a / b to nearest higher integer value
int cuda_iDivUp(int a, int b)
{
	return (a + (b - 1)) / b;
}


// complex math functions
__device__
float2 conjugate(float2 arg)
{
	return make_float2(arg.x, -arg.y);
}

__device__
float2 complex_exp(float arg)
{
	return make_float2(cosf(arg), sinf(arg));
}

__device__
float2 complex_add(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

__device__
float2 complex_mult(float2 ab, float2 cd)
{
	return make_float2(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x);
}

// generate wave heightfield at time t based on initial heightfield and dispersion relationship
__global__ void generateSpectrumKernel(float2* h0, float2 *ht, unsigned int width, unsigned int height, float t, float patchSize)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = y*width+x;

	// calculate coordinates
	float2 k;
	k.x = CUDART_PI_F * x / (float) patchSize;
	k.y = 2.0f * CUDART_PI_F * y / (float) patchSize;

	// calculate dispersion w(k)
	float k_len = sqrtf(k.x*k.x + k.y*k.y);
	float w = sqrtf(9.81f * k_len);

	float2 h0_k = h0[i];
	float2 h0_mk = h0[(((height-1)-y)*width)+x];

	float2 h_tilda = complex_add( complex_mult(h0_k, complex_exp(w * t)),
		complex_mult(conjugate(h0_mk), complex_exp(-w * t)) );

	// output frequency-space complex values
	if ((x < width) && (y < height)) {
		ht[i] = h_tilda;
	}
}


// generate slope by partial differences in spatial domain
__global__ void calculateSlopeKernel(float* h, float2 *slopeOut, unsigned int width, unsigned int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = y*width+x;

	float2 slope;
	if ((x > 0) && (y > 0) && (x < width-1) && (y < height-1)) {
		slope.x = h[i+1] - h[i-1];
		slope.y = h[i+width] - h[i-width];
	} else {
		slope = make_float2(0.0f, 0.0f);
	}
	slopeOut[i] = slope;
}

extern "C" 
void cudaGenerateSpectrumKernel(float2* d_h0, float2 *d_ht, 
								unsigned int width, unsigned int height, 
								float animTime, float patchSize)
{
	dim3 block(8, 8, 1);
	dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
	generateSpectrumKernel<<<grid, block>>>(d_h0, d_ht, width, height, animTime, patchSize);
}

extern "C"
void cudaCalculateSlopeKernel(  float* hptr, float2 *slopeOut, 
							  unsigned int width, unsigned int height)
{
	dim3 block(8, 8, 1);
	dim3 grid2(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);
	calculateSlopeKernel<<<grid2, block>>>(hptr, slopeOut, width, height);
}

__global__ void KernelFFTCUDA_H(unsigned char* surface, float *dout, int width, int height, size_t pitch, float t)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float* pixel;
	float value;

	if (x >= width || y >= height) return;

	pixel = (float*)(surface + y*pitch) + 4*x;
	value = dout[y*width + x];

	pixel[0] = value; // red
	pixel[1] = value; // green
	pixel[2] = value; // blue
	pixel[3] = 1; // alpha
}

extern "C" 
void FFTCUDA_H(void* surface, float* dout, int width, int height, size_t pitch, float t)
{
	cudaError_t error = cudaSuccess;

	dim3 Db = dim3( 16, 16 ); // block dimensions are fixed to be 256 threads
	dim3 Dg = dim3( (width+Db.x-1)/Db.x, (height+Db.y-1)/Db.y );

	KernelFFTCUDA_H<<<Dg,Db>>>( (unsigned char*)surface, dout, width, height, pitch, t );

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("PerlinCUDA_H() failed to launch error = %d\n", error);
	}
}