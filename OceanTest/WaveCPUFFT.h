#pragma once

#include <fftw3.h>
#include <windows.h>
#include <d3d9.h>
#include <d3dx9.h>
#include <omp.h>
#include <cmath>
#include "Enums.h"
#include "Random.h"

#ifndef V
#define V(x)           { hr = x; }
#endif

#ifndef V_RETURN
#define V_RETURN(x)    { hr = x; if( FAILED(hr) ) { return hr; } }
#endif

const D3DXVECTOR2 DIR[] =
{
	D3DXVECTOR2( 1.0f			 , 0.0f				),			//N
	D3DXVECTOR2( 1.0f/sqrtf(2.0f), 1.0f/sqrtf(2.0f) ),		//NE
	D3DXVECTOR2( 0.0f			 , 1.0f				),			//E
	D3DXVECTOR2(-1.0f/sqrtf(2.0f), 1.0f/sqrtf(2.0f) ),		//SE
	D3DXVECTOR2(-1.0f			 , 0.0f				),			//S
	D3DXVECTOR2(-1.0f/sqrtf(2.0f),-1.0f/sqrtf(2.0f) ),		//SW
	D3DXVECTOR2( 0.0f			 ,-1.0f				),			//W
	D3DXVECTOR2( 1.0f/sqrtf(2.0f),-1.0f/sqrtf(2.0f) )			//NW
};

const float INVSQRT2 = 1.0f/sqrtf(2.0f);
const int REAL = 0;
const int IMAGINARY = 1;
const UINT DIRS = 8;
const float GRAVITYCONSTANT = 9.81f;
const float WATERLEVEL = 0.0f;
const float REFRACTIONCLIPHEIGHT = WATERLEVEL + 5.0f;
const float OBJECTDROPHEIGHT = WATERLEVEL + 25.0f;
const UINT WATERGRIDRESOLUTION = 64;
const float WATERSIZE = 200.0f;

const float TERRAINHEIGHTSCALE = 25.0f;
const float TERRAINHEIGHTLEVEL =-7.0f;

class WaveModelStatisticalFFT
{
	void InitializeWave();
	float WaveSpectrum_Phillips(D3DXVECTOR2 K);

public:

	WaveModelStatisticalFFT(unsigned int uiGridResolution, D3DXVECTOR2 vSize);
	~WaveModelStatisticalFFT(void);

	bool Initialize(void);
	bool CalculateWaves( double fTime, bool Normals );

	bool UpdateHeight(IDirect3DTexture9* pWaveBuffer);
	bool UpdateNormals(IDirect3DTexture9* pNormalBuffer, bool Height);

	void SetWaveChoppy( float fWaveChoppy );
	void SetWaveHeightConstant( float fWaveHeightConstant );
	void SetWindSpeed( float fWindSpeed );
	void SetWindDirection( unsigned int uiWindDirection );

	float GetWaveChoppy();
	float GetWaveHeightConstant();
	float GetWindSpeed();
	unsigned int GetWindDirection();

	void SetWaterDepthWaves( bool bWaterDepthWaves );
	void SetTerrainHeightMap( float* prgTerrainHeightMap );

	// Water grid size.
	unsigned int m_uiGridResolution;
	D3DXVECTOR2 m_vGridSize;

	// Precalculated random values for the calculation of H0.
	fftw_complex* m_prgRandomValues;

	// The height transform.
	fftw_complex* m_H0;
	fftw_complex* m_H;
	fftw_complex* m_h;

	// The normal transform.
	fftw_complex* m_Nx;
	fftw_complex* m_nx;
	fftw_complex* m_Ny;
	fftw_complex* m_ny;

	// The displacement transform.
	fftw_complex* m_Dx;
	fftw_complex* m_dx;
	fftw_complex* m_Dy;
	fftw_complex* m_dy;

	// The FFTW plan which calculates the inverse FFT, H = > h
	fftw_plan m_FFTPlan;

	// Wave constants.
	float m_fWaveChoppy;
	unsigned int m_uiWindDirection;
	float m_fWindSpeed;
	float m_fWaveHeightConstant;

	// Calculate waves based on water depth.
	bool m_bWaterDepthWaves;

	// Terrain height map which is used when the waves is calculated based on water depth.
	float* m_prgTerrainHeightMap;
};