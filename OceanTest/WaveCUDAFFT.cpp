#include "WaveCUDAFFT.h"

// simulation parameters
const float g = 9.81;		   
const float A = 2*.00000000775f;	// wave scale factor
float windSpeed = 10.0f;
float windDir = CUDART_PI_F/3.0f;

// FFT data
cufftHandle fftPlan;
float2 *d_h0 = 0, *d_ht = 0;
float *d_out = 0;
float2 *h_h0 = 0;
float2 *d_slope = 0;
const float patchSize = 100;        // patch size
unsigned int fftInputW, fftInputH;
unsigned int fftInputSize;

extern "C" void cudaGenerateSpectrumKernel(float2* d_h0, float2 *d_ht, unsigned int width, unsigned int height, float animTime, float patchSize);
extern "C" void cudaCalculateSlopeKernel(  float* h, float2 *slopeOut, unsigned int width, unsigned int height);
extern "C" void PerlinCUDA_N(  float* surface, float* normals, unsigned int width, unsigned int height, size_t pitch, float t);
extern "C" void FFTCUDA_H(void* surface, float *dout, int width, int height, size_t pitch, float t);

// Phillips spectrum
// Vdir - wind angle in radians
// V - wind speed
float phillips(float Kx, float Ky, float Vdir, float V, float A)
{
	float k_squared = Kx * Kx + Ky * Ky;
	float k_x = Kx / sqrtf(k_squared);
	float k_y = Ky / sqrtf(k_squared);
	float L = V * V / g;
	float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

	if (k_squared == 0) return 0;

	return A * expf(-1.0 / (k_squared * L * L)) / (k_squared * k_squared) * w_dot_k * w_dot_k;
}

// Generate base heightfield in frequency space
void generate_h0()
{
	for (unsigned int y = 0; y<fftInputH; y++) {
		for (unsigned int x = 0; x<fftInputW; x++) {
			float kx = CUDART_PI_F * x / (float) patchSize;
			float ky = 2.0f * CUDART_PI_F * y / (float) patchSize;

			// note - these random numbers should be from a Gaussian distribution really
			float Er = 2.0f * rand() / (float) RAND_MAX - 1.0f;
			float Ei = 2.0f * rand() / (float) RAND_MAX - 1.0f;

			float P = sqrt(phillips(kx, ky, windDir, windSpeed, A));  

			float h0_re = 1.0f / sqrtf(2.0f) * Er * P;
			float h0_im = 1.0f / sqrtf(2.0f) * Ei * P;

			int i = y*fftInputW+x;
			h_h0[i].x = h0_re;
			h_h0[i].y = h0_im;
		}
	}
}

//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
	return (a + (b - 1)) / b;
}

void InitializeFFTCUDA()
{
	if ( h_h0 != NULL)
		return;

	// create FFT plan
	CUFFT_SAFE_CALL(cufftPlan2d(&fftPlan, _TexWidth, _TexHeight, CUFFT_C2R) );

	// allocate memory
	//fftInputW = (_TexWidth / 2)+1;
	//fftInputH = _TexHeight;
	fftInputW = _TexWidth;
	fftInputH = _TexHeight;
	fftInputSize = (fftInputW*fftInputH)*sizeof(float2);

	cutilSafeCall(cudaMalloc((void **)&d_h0, fftInputSize) );
	cutilSafeCall(cudaMalloc((void **)&d_ht, fftInputSize) );
	cutilSafeCall(cudaMalloc((void **)&d_out, (fftInputW*fftInputH)*sizeof(float)) );
	
	h_h0 = (float2 *) malloc(fftInputSize);
	
	generate_h0();

	cutilSafeCall(cudaMemcpy(d_h0, h_h0, fftInputSize, cudaMemcpyHostToDevice) );

	//cutilSafeCall(cudaMalloc((void **)&d_slope, _TexWidth*_TexHeight*sizeof(float2)) );

/*	cutCreateTimer(&timer);
	cutStartTimer(timer);
	prevTime = cutGetTimerValue(timer);*/
}

void FFTCUDACleanup()
{
	if ( h_h0 == NULL )
	{
		return;
	}

	cutilSafeCall( cudaFree(d_h0) );
	cutilSafeCall( cudaFree(d_out) );
	cutilSafeCall( cudaFree(d_ht) );
	//cutilSafeCall( cudaFree(d_slope) );
	free(h_h0);
	h_h0 = NULL;
	cufftDestroy(fftPlan);
}

bool SimulateCUDAFFTH( float Time, IDirect3DTexture9 *_Surface )
{
	cudaGenerateSpectrumKernel(d_h0, d_ht, fftInputW, fftInputH, Time, patchSize);

	IDirect3DResource9* ppResources[1] = 
	{
		_Surface,
	};
	cudaD3D9MapResources(1, ppResources);
	cutilCheckMsg("cudaD3D9MapResources(3) failed");

	void* pData;
	size_t pitch;
	cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pData, _Surface, 0, 0) );
	cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch, NULL, _Surface, 0, 0) );

	cufftSafeCall( cufftExecC2R(fftPlan, (cufftComplex *) d_ht, (float*)d_out) );

	FFTCUDA_H( pData, d_out, _TexWidth, _TexHeight, pitch, Time);

	cudaD3D9UnmapResources(1, ppResources);
	cutilCheckMsg("cudaD3D9UnmapResources(3) failed");

	return true;
}

bool SimulateCUDAFFTN( float Time, IDirect3DTexture9 *_SurfaceH, IDirect3DTexture9 *_SurfaceN  )
{

	IDirect3DResource9* ppResources[2] = 
	{
		_SurfaceH,
		_SurfaceN
	};

	cudaD3D9MapResources(2, ppResources);
	cutilCheckMsg("cudaD3D9MapResources(3) failed");

	void* pDataH;
	void* pDataN;
	size_t pitch;
	cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pDataH, _SurfaceH, 0, 0) );
	cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPointer(&pDataN, _SurfaceN, 0, 0) );
	cutilSafeCallNoSync ( cudaD3D9ResourceGetMappedPitch(&pitch, NULL, _SurfaceN, 0, 0) );

	PerlinCUDA_N((float*)pDataH, (float*)pDataN, _TexWidth, _TexHeight, pitch, Time );

	//cudaCalculateSlopeKernel((float*)pDataH, (float2*)pDataN, _TexWidth, _TexHeight);

	cudaD3D9UnmapResources(2, ppResources);
	cutilCheckMsg("cudaD3D9UnmapResources(3) failed");

	return true;

}