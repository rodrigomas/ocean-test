#pragma once

#include <windows.h>
#include <d3d9.h>
#include <d3dx9.h>
#include <vector>

#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d9_interop.h>
#include <cutil_inline.h>
#include <cufft.h>
#include <math_constants.h>
#include <omp.h>

extern int _TexWidth;
extern int _TexHeight;

void FFTCUDACleanup();
void InitializeFFTCUDA();
bool SimulateCUDAFFTH( float Time, IDirect3DTexture9 *_Surface );
bool SimulateCUDAFFTN( float Time, IDirect3DTexture9 *_SurfaceH, IDirect3DTexture9 *_SurfaceN  );

