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

#include <omp.h>

#include "Enums.h"
#include "WaveCPUFFT.h"
#include "WaveCUDAFFT.h"

//DONE
int CUDAGertnerInitialize();
bool CUDAGertnerHSimulate(float Time, IDirect3DTexture9	*_Surface);
bool CUDAGertnerNSimulate(float Time, IDirect3DTexture9	*_Surface);
int CUDAGertnerUninitialize();

//DONE
int CUDAPerlinInitialize(int Resolution);
bool CUDAPerlinHSimulate(float Time, IDirect3DTexture9	*_Surface);
bool CUDAPerlinNSimulate(float Time,  IDirect3DTexture9 *_SurfaceH, IDirect3DTexture9 *_SurfaceN);
int CUDAPerlinUninitialize();

//DONE
int CUDAFFTInitialize();
bool CUDAFFTHSimulate(float Time, IDirect3DTexture9	*_Surface);
bool CUDAFFTNSimulate(float Time, IDirect3DTexture9 *_SurfaceH, IDirect3DTexture9 *_SurfaceN );
int CUDAFFTUninitialize();

//////////////////////////////////////////////////////////////////////////
//DONE
int CPUGertnerInitialize();
bool CPUGertnerHSimulate(float Time, IDirect3DTexture9	*_Surface);
bool CPUGertnerNSimulate(float Time, IDirect3DTexture9	*_Surface);
int CPUGertnerUninitialize();

//DONE
int CPUPerlinInitialize(int Resolution);
bool CPUPerlinHSimulate(float Time, IDirect3DTexture9	*_Surface);
bool CPUPerlinNSimulate(float Time, IDirect3DTexture9 *_SurfaceH, IDirect3DTexture9 *_SurfaceN );
int CPUPerlinUninitialize();

//DONE
int CPUFFTInitialize();
bool CPUFFTHSimulate(float Time, IDirect3DTexture9	*_Surface);
bool CPUFFTNSimulate(float Time, IDirect3DTexture9	*_Surface);
int CPUFFTUninitialize();

//////////////////////////////////////////////////////////////////////////

int GPUGertnerInitialize();
void GPUGertnerHSimulate(float Time, IDirect3DSurface9	*_Surface, IDirect3DSurface9*_Depth);
void GPUGertnerNSimulate(float Time, IDirect3DSurface9	*_Surface, IDirect3DSurface9*_Depth);
int GPUGertnerUninitialize();

int GPUPerlinInitialize(int Resolution, IDirect3DTexture9 *_Tex);
void GPUPerlinHSimulate(float Time, IDirect3DSurface9	*_Surface, IDirect3DSurface9*_Depth);
void GPUPerlinNSimulate(float Time, IDirect3DSurface9	*_Surface, IDirect3DSurface9*_Depth);
int GPUPerlinUninitialize();