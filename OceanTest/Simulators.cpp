#include "Simulators.h"
#include "Random.h"
#include "PerformanceManager.h"
#include <string>

using std::string;

extern "C" void GerstnerCUDA_N(void* surface,int nwaves, float* data, int width, int height, size_t pitch, float t);
extern "C" void GerstnerCUDA_H(void* surface,int nwaves, float* data, int width, int height, size_t pitch, float t);
extern "C" void PerlinCUDA_H(void* surface, void* noise, int res, int width, int height, float Overcast, size_t pitch, float t);
extern "C" void PerlinCUDA_N(  float* surface, float* normals, unsigned int width, unsigned int height, size_t pitch, float t);

struct GertnerWave
{
	float Q;
	float FREQ;
	float PHASE;
	float AMP;
	D3DXVECTOR3 DIR;
};

float *CUDAGerstner = NULL;

extern int _TexWidth;
extern int _TexHeight;
extern IDirect3DTexture9**	_Textures;	
extern IDirect3DSurface9**	_Surfaces;
extern IDirect3DSurface9**	_Depths;
extern int _nTextures;
extern int _iNormal;
extern int _iCores;
extern int _iWaves;
extern IDirect3DDevice9* _pd3dDevice;

float *CUDANoiseMap = NULL;

GertnerWave *GertnerWaves = NULL;

float **NoiseMap = NULL;
int _NoiseRes;
float Overcast = 1.1;

WaveModelStatisticalFFT *CPUFFT = NULL;



struct FloorVertex
{
	FloorVertex(float a, float b,float c, float d, float e)
	{
		x = a; y = b; z = c; u = d; v = e;
	}

	float x, y, z, u, v;
};

IDirect3DVertexBuffer9 *_pgVertexBuffer = NULL;
IDirect3DVertexDeclaration9* _pgVertexDeclaration = NULL;
ID3DXEffect *_GerstnerEffect = NULL;
D3DXHANDLE m_hgTime = NULL;
D3DXHANDLE m_hgSize = NULL;
D3DXHANDLE m_hgWaves = NULL;

IDirect3DVertexBuffer9 *_ppVertexBuffer = NULL;
IDirect3DVertexDeclaration9* _ppVertexDeclaration = NULL;
ID3DXEffect *_PelinEffect = NULL;
IDirect3DTexture9 *_NoiseTex = NULL;
IDirect3DTexture9 *_PerlinTex = NULL;
D3DXHANDLE m_hpTime = NULL;
D3DXHANDLE m_hpTex = NULL;
D3DXHANDLE m_hpSize = NULL;

int CPUGertnerInitialize()
{
	if (GertnerWaves != NULL)
		return 1;

	GertnerWaves = new GertnerWave[_iWaves];

	int Cores = omp_get_max_threads(); //omp_get_num_threads(); //omp_get_thread_num();

	omp_set_num_threads( (_iCores <= Cores) ? _iCores : Cores );

	srand(0);
	#pragma omp parallel for
	for(register int i = 0 ; i < _iWaves ; i++)
	{
		GertnerWaves[i].AMP = (rand() % 1000) / 1000.0;
		GertnerWaves[i].FREQ = (rand() % 1000) / 1000.0;
		GertnerWaves[i].PHASE = (rand() % 1000) / 1000.0;
		GertnerWaves[i].Q = (rand() % 1000) / 1000.0;
		GertnerWaves[i].DIR = D3DXVECTOR3((rand() % 1000) / 1000.0,(rand() % 1000) / 1000.0,(rand() % 1000) / 1000.0);
	}	

	PerformanceManager::getSingleton()->AddPerformaceCounter(string("GerstnerCPUH"));	
	PerformanceManager::getSingleton()->AddPerformaceCounter(string("GerstnerCPUN"));

	return 0;
}

int CPUGertnerUninitialize()
{
	SAFE_ARRAY_DELETE(GertnerWaves);

	return 0;
}

bool CPUGertnerHSimulate( float Time, IDirect3DTexture9 *_Surface)
{		
	PerformanceManager::getSingleton()->Reset(string("GerstnerCPUH"));
	D3DLOCKED_RECT sRect;
	if ( FAILED( _Surface->LockRect( 0, &sRect, NULL, D3DLOCK_DISCARD ) ) )
	{
		//		m_pDebug->Log( "HEIGHTMAP::Create() - Failed to lock texture memory" );
		return false;
	}

	float *imageData = ( float* )sRect.pBits; // start of the memory block

	#pragma omp parallel for
	for( int y = 0; y < _TexHeight; y++ )
	{
		for( int x = 0; x < _TexWidth; x++ )
		{			
			D3DXVECTOR3 P0 = D3DXVECTOR3(x,0,y);
			D3DXVECTOR3 Position = D3DXVECTOR3(P0);
			//D3DXVECTOR3 N;

			for(int i = 0; i< _iWaves; ++i)
			{				
				float angle = GertnerWaves[i].FREQ * D3DXVec3Dot(&GertnerWaves[i].DIR,&P0) + GertnerWaves[i].PHASE * Time; 

				float Si = sin(angle);
				float C = cos(angle);

				Position.x += GertnerWaves[i].Q * GertnerWaves[i].AMP * GertnerWaves[i].DIR.x * C;
				Position.z += GertnerWaves[i].Q * GertnerWaves[i].AMP * GertnerWaves[i].DIR.z * C;
				Position.y += GertnerWaves[i].AMP * Si;

				/*float WA = GertnerWaves[i].FREQ * GertnerWaves[i].AMP;				

				angle = GertnerWaves[i].FREQ * D3DXVec3Dot(GertnerWaves[i].DIR,Position) + GertnerWaves[i].PHASE * Time; 				

				Si = sin(angle);
				C = cos(angle);

				N.x -= GertnerWaves[i].DIR.x * WA * C;		
				N.z -= GertnerWaves[i].DIR.z * WA * C;
				N.y -= GertnerWaves[i].Q * WA * Si;*/
			}

			imageData[y * sRect.Pitch / 4 + x * 4 + 0 ] = Position.x;
			imageData[y * sRect.Pitch / 4 + x * 4 + 1 ] = Position.y;
			imageData[y * sRect.Pitch / 4 + x * 4 + 2 ] = Position.z;
			imageData[y * sRect.Pitch / 4 + x * 4 + 3 ] = 1.0f;
		}
	}

	//Unlock the texture
	if ( FAILED( _Surface->UnlockRect( 0 ) ) )
	{
		//m_pDebug->Log( "HEIGHTMAP::Create() - Failed to unlock texture memory" );
		return false;
	}
	PerformanceManager::getSingleton()->Update(string("GerstnerCPUH"));
}

bool CPUGertnerNSimulate( float Time, IDirect3DTexture9 *_Surface )
{
	PerformanceManager::getSingleton()->Reset(string("GerstnerCPUN"));
	D3DLOCKED_RECT sRect;
	if ( FAILED( _Surface->LockRect( 0, &sRect, NULL, D3DLOCK_DISCARD ) ) )
	{
		//		m_pDebug->Log( "HEIGHTMAP::Create() - Failed to lock texture memory" );
		return false;
	}

	float *imageData = ( float* )sRect.pBits; // start of the memory block

	#pragma omp parallel for
	for( int y = 0; y < _TexHeight; y++ )
	{
		for( int x = 0; x < _TexWidth; x++ )
		{			
			D3DXVECTOR3 P0 = D3DXVECTOR3(x,0,y);
			D3DXVECTOR3 Position = D3DXVECTOR3(P0);
			D3DXVECTOR3 N = D3DXVECTOR3(0,0,0);

			for(int i = 0; i< _iWaves; ++i)
			{				
				float angle = GertnerWaves[i].FREQ * D3DXVec3Dot(&GertnerWaves[i].DIR,&P0) + GertnerWaves[i].PHASE * Time; 

				float Si = sin(angle);
				float C = cos(angle);

				Position.x += GertnerWaves[i].Q * GertnerWaves[i].AMP * GertnerWaves[i].DIR.x * C;
				Position.z += GertnerWaves[i].Q * GertnerWaves[i].AMP * GertnerWaves[i].DIR.z * C;
				Position.y += GertnerWaves[i].AMP * Si;
			}

			for(int i = 0; i< _iWaves; ++i)
			{				
				float WA = GertnerWaves[i].FREQ * GertnerWaves[i].AMP;				

				float angle = GertnerWaves[i].FREQ * D3DXVec3Dot(&GertnerWaves[i].DIR,&Position) + GertnerWaves[i].PHASE * Time; 				

				float Si = sin(angle);
				float C = cos(angle);

				N.x -= GertnerWaves[i].DIR.x * WA * C;		
				N.z -= GertnerWaves[i].DIR.z * WA * C;
				N.y -= GertnerWaves[i].Q * WA * Si;
			}

			imageData[y * sRect.Pitch / 4 + x * 4 + 0 ] = N.x;
			imageData[y * sRect.Pitch / 4 + x * 4 + 1 ] = N.y;
			imageData[y * sRect.Pitch / 4 + x * 4 + 2 ] = N.z;
			imageData[y * sRect.Pitch / 4 + x * 4 + 3 ] = 1.0f;
		}
	}

	//Unlock the texture
	if ( FAILED( _Surface->UnlockRect( 0 ) ) )
	{
		//m_pDebug->Log( "HEIGHTMAP::Create() - Failed to unlock texture memory" );
		return false;
	}
	PerformanceManager::getSingleton()->Update(string("GerstnerCPUN"));
}

int CUDAGertnerInitialize()
{
	if (CUDAGerstner != NULL)
		return 1;

	float * data;
	data = new float[7*_iWaves];

	srand(0);
	for(register int i = 0 ; i < _iWaves ; i++)
	{
		data[i*7] = (rand() % 1000) / 1000.0;
		data[i*7+1] = (rand() % 1000) / 1000.0;
		data[i*7+2] = (rand() % 1000) / 1000.0;
		data[i*7+3] = (rand() % 1000) / 1000.0;
		data[i*7+4] = (rand() % 1000) / 1000.0;
		data[i*7+5] = (rand() % 1000) / 1000.0;
		data[i*7+6] = (rand() % 1000) / 1000.0;
	}	

	int size = 7*_iWaves*sizeof(float);	

#ifdef CONSTANT_MEMORY	
	cutilSafeCall(cudaMemcpyToSymbol((const char*)"ger_data", data, size, 0, cudaMemcpyHostToDevice));
#endif

	cutilSafeCall(cudaMalloc((void **)&CUDAGerstner, size) );
	cutilSafeCall(cudaMemcpy(CUDAGerstner, data, size, cudaMemcpyHostToDevice) );

	delete [] data;

	PerformanceManager::getSingleton()->AddPerformaceCounter(string("GerstnerCUDAH"));	
	PerformanceManager::getSingleton()->AddPerformaceCounter(string("GerstnerCUDAN"));

	return 0;
}

int CUDAGertnerUninitialize()
{
	if ( CUDAGerstner == NULL)
		return 1;

	cutilSafeCall( cudaFree(CUDAGerstner) );

	CUDAGerstner = NULL;

	return 0;
}

bool CUDAGertnerHSimulate( float Time, IDirect3DTexture9 *_Surface )
{
	PerformanceManager::getSingleton()->Reset(string("GerstnerCUDAH"));

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
	GerstnerCUDA_H(pData, _iWaves, CUDAGerstner, _TexWidth, _TexHeight, pitch, Time);

	cudaD3D9UnmapResources(1, ppResources);
	cutilCheckMsg("cudaD3D9UnmapResources(3) failed");

	PerformanceManager::getSingleton()->Update(string("GerstnerCUDAH"));

	return true;
}

bool CUDAGertnerNSimulate( float Time, IDirect3DTexture9 *_Surface )
{
	PerformanceManager::getSingleton()->Reset(string("GerstnerCUDAN"));
	
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
	GerstnerCUDA_N(pData, _iWaves, CUDAGerstner, _TexWidth, _TexHeight, pitch, Time);

	cudaD3D9UnmapResources(1, ppResources);
	cutilCheckMsg("cudaD3D9UnmapResources(3) failed");

	PerformanceManager::getSingleton()->Update(string("GerstnerCUDAN"));

	return true;
}

int CPUFFTInitialize()
{

	if( CPUFFT != NULL)
	{
		return 1;
	}

	int Cores = omp_get_max_threads(); //omp_get_num_threads();// omp_get_thread_num();

	omp_set_num_threads( (_iCores <= Cores) ? _iCores : Cores );

	CPUFFT = new WaveModelStatisticalFFT(_TexWidth, D3DXVECTOR2(_TexWidth * 2 / 3.0,_TexHeight * 2 / 3.0));
	//CPUFFT = new WaveModelStatisticalFFT(_TexWidth, D3DXVECTOR2(200.0,200.0));

	CPUFFT->Initialize();

	PerformanceManager::getSingleton()->AddPerformaceCounter(string("FFTCPUN"));	
	PerformanceManager::getSingleton()->AddPerformaceCounter(string("FFTCPUH"));

	return 0;
}

int CPUFFTUninitialize()
{
	SAFE_DELETE(CPUFFT);

	return 0;
}

bool CPUFFTHSimulate( float Time, IDirect3DTexture9 *_Surface )
{
	PerformanceManager::getSingleton()->Reset(string("FFTCPUH"));

	CPUFFT->CalculateWaves(Time, false);
	CPUFFT->UpdateHeight(_Surface);

	PerformanceManager::getSingleton()->Update(string("FFTCPUH"));

	return true;
}

bool CPUFFTNSimulate( float Time, IDirect3DTexture9 *_Surface )
{
	PerformanceManager::getSingleton()->Reset(string("FFTCPUN"));

	CPUFFT->CalculateWaves(Time, true);
	CPUFFT->UpdateNormals(_Surface,true);

	PerformanceManager::getSingleton()->Update(string("FFTCPUN"));
	
	return true;
}

int CUDAFFTInitialize()
{
	InitializeFFTCUDA();

	PerformanceManager::getSingleton()->AddPerformaceCounter(string("FFTCUDAH"));	
	PerformanceManager::getSingleton()->AddPerformaceCounter(string("FFTCUDAN"));

	return 0;
}

int CUDAFFTUninitialize()
{
	FFTCUDACleanup();

	return 0;
}

bool CUDAFFTHSimulate( float Time, IDirect3DTexture9 *_Surface )
{

	PerformanceManager::getSingleton()->Reset(string("FFTCUDAH"));

	SimulateCUDAFFTH(Time, _Surface);

	PerformanceManager::getSingleton()->Update(string("FFTCUDAH"));

	return true;

}

bool CUDAFFTNSimulate( float Time, IDirect3DTexture9 *_SurfaceH, IDirect3DTexture9 *_SurfaceN  )
{

	PerformanceManager::getSingleton()->Reset(string("FFTCUDAN"));	

	SimulateCUDAFFTN(Time, _SurfaceH, _SurfaceN);

	PerformanceManager::getSingleton()->Update(string("FFTCUDAN"));

	return true;

}

int CPUPerlinInitialize( int Resolution)
{

	if(NoiseMap != NULL)
		return 1;

	int Cores = omp_get_max_threads(); //omp_get_num_threads(); //omp_get_thread_num();

	omp_set_num_threads( (_iCores <= Cores) ? _iCores : Cores );

	_NoiseRes = Resolution;

	NoiseMap = new float*[_NoiseRes];

	srand(0);
	for( register int y = 0 ; y < _NoiseRes ; y++ )
	{
		NoiseMap[y] = new float[_NoiseRes];

		for( register int x = 0 ; x < _NoiseRes ; x++ )
		{			
			NoiseMap[y][x] = (rand() % 1000) / 1000.0;
		}
	}
	
	PerformanceManager::getSingleton()->AddPerformaceCounter(string("PerlinCPUH"));	
	PerformanceManager::getSingleton()->AddPerformaceCounter(string("PerlinCPUN"));

	return 0;
}

int CPUPerlinUninitialize()
{
	if ( NoiseMap == NULL )
		return 1;

	for( register int y = 0 ; y < _NoiseRes ; y++ )
	{
		delete [] NoiseMap[y];
	}

	SAFE_ARRAY_DELETE(NoiseMap);

	return 0;
}

bool CPUPerlinHSimulate( float Time, IDirect3DTexture9 *_Surface )
{
	PerformanceManager::getSingleton()->Reset(string("PerlinCPUH"));
	D3DLOCKED_RECT sRect;
	if ( FAILED( _Surface->LockRect( 0, &sRect, NULL, D3DLOCK_DISCARD ) ) )
	{
		//		m_pDebug->Log( "HEIGHTMAP::Create() - Failed to lock texture memory" );
		return false;
	}

	float *imageData = ( float* )sRect.pBits; // start of the memory block

	#pragma omp parallel for
	for(int y = 0; y < _TexHeight; y++ )
	{
		for(int x = 0; x < _TexWidth; x++ )
		{			

			D3DXVECTOR2 move = D3DXVECTOR2(0.0,1.0);
			D3DXVECTOR2 pos = D3DXVECTOR2(0.0,0.0);
			float perlin = 0.0;

			perlin =  NoiseMap[ (int)((x + Time * move.x)) % _NoiseRes][ (int)((y + Time * move.y)) % _NoiseRes] / 2.0;
			perlin +=  NoiseMap[ (int)((x*2.0 + Time * move.x)) % _NoiseRes][ (int)((y*2.0 + Time * move.y)) % _NoiseRes] / 4.0;
			perlin +=  NoiseMap[ (int)((x*4.0 + Time * move.x)) % _NoiseRes][ (int)((y*4.0 + Time * move.y)) % _NoiseRes] / 8.0;
			perlin +=  NoiseMap[ (int)((x*8.0 + Time * move.x)) % _NoiseRes][ (int)((y*8.0 + Time * move.y)) % _NoiseRes] / 16.0;
			perlin +=  NoiseMap[ (int)((x*16.0 + Time * move.x)) % _NoiseRes][ (int)((y*16.0 + Time * move.y)) % _NoiseRes] / 32.0;
			perlin +=  NoiseMap[ (int)((x*32.0 + Time * move.x)) % _NoiseRes][ (int)((y*32.0 + Time * move.y)) % _NoiseRes] / 32.0;
			
			imageData[y * sRect.Pitch / 4 + x * 4 + 0 ] = 1.0 - pow(perlin, Overcast) * 2.0;
			imageData[y * sRect.Pitch / 4 + x * 4 + 1 ] = imageData[y * sRect.Pitch / 4 + x * 4 + 0 ];
			imageData[y * sRect.Pitch / 4 + x * 4 + 2 ] = imageData[y * sRect.Pitch / 4 + x * 4 + 0 ];
			imageData[y * sRect.Pitch / 4 + x * 4 + 3 ] = 1.0f;			
		}
	}

	//Unlock the texture
	if ( FAILED( _Surface->UnlockRect( 0 ) ) )
	{
		//m_pDebug->Log( "HEIGHTMAP::Create() - Failed to unlock texture memory" );
		return false;
	}
	PerformanceManager::getSingleton()->Update(string("PerlinCPUH"));
	
	return true;
}

bool CPUPerlinNSimulate( float Time, IDirect3DTexture9 *_SurfaceH, IDirect3DTexture9 *_SurfaceN )
{
	PerformanceManager::getSingleton()->Reset(string("PerlinCPUN"));
	
	D3DLOCKED_RECT sRectH;
	D3DLOCKED_RECT sRect;
	if ( FAILED( _SurfaceN->LockRect( 0, &sRect, NULL, D3DLOCK_DISCARD ) ) )
	{
		//		m_pDebug->Log( "HEIGHTMAP::Create() - Failed to lock texture memory" );
		return false;
	}

	if ( FAILED( _SurfaceH->LockRect( 0, &sRectH, NULL, D3DLOCK_READONLY ) ) )
	{
		_SurfaceN->UnlockRect(0);
		//		m_pDebug->Log( "HEIGHTMAP::Create() - Failed to lock texture memory" );
		return false;
	}

	float *imageData = ( float* )sRect.pBits; // start of the memory block
	float *hdata = (float*)sRectH.pBits;

	#pragma omp parallel for
	for( int y = 0; y < _TexHeight; y++ )
	{
		for( int x = 0; x < _TexWidth; x++ )
		{			

			float dx = hdata[y * sRect.Pitch/4 + x * 4 + 0 ] - hdata[y * sRect.Pitch/4 + ((x+1)%_TexWidth) * 4 + 0 ];
			float dz = hdata[y * sRect.Pitch/4 + x * 4 + 0 ] - hdata[((y+1) % _TexHeight ) * sRect.Pitch/4 + x * 4 + 0 ];

			D3DXVECTOR3 N = D3DXVECTOR3(0,0,0);

			D3DXVECTOR3 T = D3DXVECTOR3(1,0,dz);
			D3DXVec3Normalize(&T,&T);

			D3DXVECTOR3 B = D3DXVECTOR3(1,0,dx);
			D3DXVec3Normalize(&B,&B);

			D3DXVec3Cross(&N, &T, &B);			

			imageData[y * sRect.Pitch / 4 + x * 4 + 0 ] = N.x;
			imageData[y * sRect.Pitch / 4 + x * 4 + 1 ] = N.y;
			imageData[y * sRect.Pitch / 4 + x * 4 + 2 ] = N.z;
			imageData[y * sRect.Pitch / 4 + x * 4 + 3 ] = 1.0f;			
		}
	}

	//Unlock the texture
	if ( FAILED( _SurfaceH->UnlockRect( 0 ) ) )
	{
		//m_pDebug->Log( "HEIGHTMAP::Create() - Failed to unlock texture memory" );
		_SurfaceN->UnlockRect(0);
		return false;
	}


	//Unlock the texture
	if ( FAILED( _SurfaceN->UnlockRect( 0 ) ) )
	{
		//m_pDebug->Log( "HEIGHTMAP::Create() - Failed to unlock texture memory" );
		return false;
	}

	PerformanceManager::getSingleton()->Update(string("PerlinCPUN"));

	return true;
}

int CUDAPerlinInitialize( int Resolution )
{
	if (CUDANoiseMap != NULL)
		return 1;

	float *Map;
	
	float Overcast = 0.1;
	_NoiseRes = Resolution;

	Map = new float[Resolution*Resolution];

	for( register int y = 0 ; y < Resolution*Resolution ; y++ )
	{
			Map[y] = Random::RandN();
	}

	int size = Resolution*Resolution*sizeof(float);

	cutilSafeCall(cudaMalloc((void **)&CUDANoiseMap, size) );

	cutilSafeCall(cudaMemcpy(CUDANoiseMap, Map, size, cudaMemcpyHostToDevice) );

	PerformanceManager::getSingleton()->AddPerformaceCounter(string("PerlinCUDAH"));	
	PerformanceManager::getSingleton()->AddPerformaceCounter(string("PerlinCUDAN"));

	return 0;
}

int CUDAPerlinUninitialize()
{
	if ( CUDANoiseMap == NULL)
		return 1;

	cutilSafeCall( cudaFree(CUDANoiseMap) );

	CUDANoiseMap = NULL;

	return 0;
}

bool CUDAPerlinHSimulate( float Time, IDirect3DTexture9 *_Surface )
{

	PerformanceManager::getSingleton()->Reset(string("GerstnerCUDAH"));

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

	PerlinCUDA_H(pData, CUDANoiseMap, _NoiseRes, _TexWidth, _TexHeight, Overcast, pitch, Time);

	cudaD3D9UnmapResources(1, ppResources);
	cutilCheckMsg("cudaD3D9UnmapResources(3) failed");

	PerformanceManager::getSingleton()->Update(string("GerstnerCUDAH"));

	return true;
}

bool CUDAPerlinNSimulate( float Time, IDirect3DTexture9 *_SurfaceH, IDirect3DTexture9 *_SurfaceN )
{
	PerformanceManager::getSingleton()->Reset(string("PERLINCUDAN"));	

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

	cudaD3D9UnmapResources(2, ppResources);
	cutilCheckMsg("cudaD3D9UnmapResources(3) failed");

	PerformanceManager::getSingleton()->Update(string("PERLINCUDAN"));

	return true;
}

void GPUGertnerHSimulate( float Time, IDirect3DSurface9 *_Surface, IDirect3DSurface9*_Depth )
{
	PerformanceManager::getSingleton()->Reset(string("GerstnerGPUH"));	
	IDirect3DSurface9*	_OldDepthStencil;
	IDirect3DSurface9*	_Old;
	
	D3DXMATRIX matProjection;
	D3DXMatrixIdentity(&matProjection);

	_pd3dDevice->GetRenderTarget(0, &_Old );

	_pd3dDevice->GetDepthStencilSurface( &_OldDepthStencil );

	_pd3dDevice->SetRenderTarget(0, _Surface);
	_pd3dDevice->SetDepthStencilSurface( _Depth );

	_pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_XRGB(100,100,100),	1.0f, 0);

	_pd3dDevice->BeginScene();
	_pd3dDevice->SetTransform(D3DTS_WORLD, &matProjection);
	_pd3dDevice->SetTransform(D3DTS_PROJECTION, &matProjection);

	_pd3dDevice->SetVertexDeclaration( _pgVertexDeclaration );
	_pd3dDevice->SetStreamSource( 0, _pgVertexBuffer, 0, sizeof( FloorVertex ) );
	
	_GerstnerEffect->SetFloat( m_hgTime, Time );	
	_GerstnerEffect->SetTechnique( "Gerstner" );	

	UINT cPasses;
	_GerstnerEffect->Begin( &cPasses, 0 );

	for( UINT p = 0; p < cPasses; ++p )
	{
		_GerstnerEffect->BeginPass( p );

		_GerstnerEffect->CommitChanges();
		_pd3dDevice->DrawPrimitive( D3DPT_TRIANGLELIST, 0, 2);

		_GerstnerEffect->EndPass();
	}
	_GerstnerEffect->End();

	_pd3dDevice->EndScene();
	_pd3dDevice->Present(NULL,NULL,NULL,NULL);

	_pd3dDevice->SetRenderTarget(0, _Old);
	_pd3dDevice->SetDepthStencilSurface( _OldDepthStencil );
	PerformanceManager::getSingleton()->Update(string("GerstnerGPUH"));
}

int GPUGertnerInitialize()
{
	if ( _GerstnerEffect != NULL )
		return 1;

	// Create the sky effect object.
	LPD3DXBUFFER pBuffer;

	if( FAILED( D3DXCreateEffectFromFile( _pd3dDevice, "gerstner.fx", NULL, NULL,0, NULL, &_GerstnerEffect, &pBuffer ) ) )
	{
		LPVOID pCompileErrors = pBuffer->GetBufferPointer();
		
		MessageBox(NULL, (const char*)pCompileErrors, "Compile Error",
			MB_OK|MB_ICONEXCLAMATION);
		//PostQuitMessage( 0 );
		return E_FAIL;
	}

	if( !( m_hgTime			  = _GerstnerEffect->GetParameterByName( NULL, "g_fTime" ) ) )				return E_FAIL;
	if( !( m_hgSize			  = _GerstnerEffect->GetParameterByName( NULL, "g_fSize" ) ) )				return E_FAIL;
	if( !( m_hgWaves			  = _GerstnerEffect->GetParameterByName( NULL, "nWaves" ) ) )				return E_FAIL;

	_GerstnerEffect->SetFloat( m_hgSize, _TexWidth );	
	_GerstnerEffect->SetInt( m_hgWaves, _iWaves );	
	

	D3DVERTEXELEMENT9 pVertexElementsDecl[] = 
	{
		{0,  0, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
		{0, 12, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
		D3DDECL_END()
	};

	if( FAILED( _pd3dDevice->CreateVertexDeclaration( pVertexElementsDecl, &_pgVertexDeclaration ) ) )
	{
		return E_FAIL;
	}

	if(FAILED(_pd3dDevice->CreateVertexBuffer(  6 * sizeof(FloorVertex), 
		D3DUSAGE_WRITEONLY, 0, 
		D3DPOOL_DEFAULT, &_pgVertexBuffer, NULL)))
	{
		return E_FAIL;
	}

	FloorVertex *Vertex;

	_pgVertexBuffer->Lock(0,0, (void**)&Vertex, 0);

	register int cnt = 0;	
	Vertex[cnt++] = FloorVertex(-1, 1, 0, 0, 1);
	Vertex[cnt++] = FloorVertex(1, 1, 0, 1, 1);
	Vertex[cnt++] = FloorVertex(-1, -1, 0, 0, 0);

	Vertex[cnt++] = FloorVertex(1, 1, 0, 1, 1);
	Vertex[cnt++] = FloorVertex(1, -1, 0, 1, 0);
	Vertex[cnt++] = FloorVertex(-1, -1, 0, 0, 0);

	_pgVertexBuffer->Unlock();

	PerformanceManager::getSingleton()->AddPerformaceCounter(string("GerstnerGPUH"));	
	PerformanceManager::getSingleton()->AddPerformaceCounter(string("GerstnerGPUN"));

	return 0;
}

int GPUGertnerUninitialize()
{
	if ( _GerstnerEffect == NULL )
		return 1;

	SAFE_RELEASE( _pgVertexDeclaration );
	SAFE_RELEASE( _pgVertexBuffer );
	SAFE_RELEASE( _GerstnerEffect );

	return 0;
}

int GPUPerlinInitialize(int Resolution, IDirect3DTexture9 *_Tex)
{
	if( _NoiseTex != NULL )
		return 1;

	// Create the sky effect object.
	LPD3DXBUFFER pBuffer;

	if( FAILED( D3DXCreateEffectFromFile( _pd3dDevice, "perlin.fx", NULL, NULL,0, NULL, &_PelinEffect, &pBuffer ) ) )
	{
		LPVOID pCompileErrors = pBuffer->GetBufferPointer();

		MessageBox(NULL, (const char*)pCompileErrors, "Compile Error",
			MB_OK|MB_ICONEXCLAMATION);
		//PostQuitMessage( 0 );
		return E_FAIL;
	}

	if( !( m_hpTime			  = _PelinEffect->GetParameterByName( NULL, "g_fTime" ) ) )				return E_FAIL;
	if( !( m_hpTex			  = _PelinEffect->GetParameterByName( NULL, "g_Tex" ) ) )				return E_FAIL;
	if( !( m_hpSize			  = _PelinEffect->GetParameterByName( NULL, "g_fSize" ) ) )				return E_FAIL;
	_PelinEffect->SetFloat( m_hpSize, _TexWidth );	


	D3DVERTEXELEMENT9 ppVertexElementsDecl[] = 
	{
		{0,  0, D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0},
		{0, 12, D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0},
		D3DDECL_END()
	};

	if( FAILED( _pd3dDevice->CreateVertexDeclaration( ppVertexElementsDecl, &_ppVertexDeclaration ) ) )
	{
		return E_FAIL;
	}

	if(FAILED(_pd3dDevice->CreateVertexBuffer(  6 * sizeof(FloorVertex), 
		D3DUSAGE_WRITEONLY, 0, 
		D3DPOOL_DEFAULT, &_ppVertexBuffer, NULL)))
	{
		return E_FAIL;
	}

	FloorVertex *Vertex;

	_ppVertexBuffer->Lock(0,0, (void**)&Vertex, 0);

	_PerlinTex = _Tex;

	register int cnt = 0;	

	Vertex[cnt++] = FloorVertex(-1, 1, 0, 0, 1);
	Vertex[cnt++] = FloorVertex(1, 1, 0, 1, 1);
	Vertex[cnt++] = FloorVertex(-1, -1, 0, 0, 0);

	Vertex[cnt++] = FloorVertex(1, 1, 0, 1, 1);
	Vertex[cnt++] = FloorVertex(1, -1, 0, 1, 0);
	Vertex[cnt++] = FloorVertex(-1, -1, 0, 0, 0);

	_ppVertexBuffer->Unlock();

	_NoiseRes = Resolution;

	_pd3dDevice->CreateTexture(_NoiseRes, _NoiseRes, 1, D3DUSAGE_DYNAMIC, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &_NoiseTex, NULL);

	D3DLOCKED_RECT sRect;
	if ( FAILED( _NoiseTex->LockRect( 0, &sRect, NULL, D3DLOCK_DISCARD ) ) )
	{
		//		m_pDebug->Log( "HEIGHTMAP::Create() - Failed to lock texture memory" );
		return false;
	}

	DWORD *imageData = ( DWORD* )sRect.pBits; // start of the memory block

	for( int y = 0; y < _TexHeight; y++ )
	{
		for( int x = 0; x < _TexWidth; x++ )
		{			
			float v = Random::RandN();

			imageData[y * sRect.Pitch / 4 + x ] = D3DCOLOR_XRGB((DWORD)(v* 255.0), (DWORD)(v* 255.0), (DWORD)(v* 255.0));		
		}
	}

	//Unlock the texture
	if ( FAILED( _NoiseTex->UnlockRect( 0 ) ) )
	{
		//m_pDebug->Log( "HEIGHTMAP::Create() - Failed to unlock texture memory" );
		return false;
	}

	_PelinEffect->SetTexture( m_hpTex, _NoiseTex );	

	//D3DXSaveTextureToFile()

	PerformanceManager::getSingleton()->AddPerformaceCounter(string("PerlinGPUH"));	
	PerformanceManager::getSingleton()->AddPerformaceCounter(string("PerlinGPUN"));

	return 0;
}

int GPUPerlinUninitialize()
{
	if( _NoiseTex == NULL )
		return 1;

	_PerlinTex = NULL;

	SAFE_RELEASE( _ppVertexDeclaration );
	SAFE_RELEASE( _ppVertexBuffer );
	SAFE_RELEASE( _PelinEffect );
	SAFE_RELEASE( _NoiseTex );

	return 0;
}

void GPUPerlinHSimulate( float Time, IDirect3DSurface9 *_Surface, IDirect3DSurface9*_Depth )
{
	PerformanceManager::getSingleton()->Reset(string("PerlinGPUH"));	
	IDirect3DSurface9*	_OldDepthStencil;
	IDirect3DSurface9*	_Old;

	D3DXMATRIX matProjection;
	D3DXMatrixIdentity(&matProjection);

	_pd3dDevice->GetRenderTarget(0, &_Old );

	_pd3dDevice->GetDepthStencilSurface( &_OldDepthStencil );

	_pd3dDevice->SetRenderTarget(0, _Surface);
	_pd3dDevice->SetDepthStencilSurface( _Depth );

	_pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_XRGB(100,100,100),	1.0f, 0);

	_pd3dDevice->BeginScene();
	_pd3dDevice->SetTransform(D3DTS_WORLD, &matProjection);
	_pd3dDevice->SetTransform(D3DTS_PROJECTION, &matProjection);

	_pd3dDevice->SetVertexDeclaration( _ppVertexDeclaration );
	_pd3dDevice->SetStreamSource( 0, _ppVertexBuffer, 0, sizeof( FloorVertex ) );

	_PelinEffect->SetFloat( m_hpTime, Time );	
	_PelinEffect->SetTechnique( "Perlin" );	
	_PelinEffect->SetTexture( m_hpTex, _NoiseTex );	

	UINT cPasses;
	_PelinEffect->Begin( &cPasses, 0 );

	for( UINT p = 0; p < cPasses; ++p )
	{
		_PelinEffect->BeginPass( p );

		_PelinEffect->CommitChanges();
		_pd3dDevice->DrawPrimitive( D3DPT_TRIANGLELIST, 0, 2);

		_PelinEffect->EndPass();
	}
	_PelinEffect->End();

	_pd3dDevice->EndScene();
	_pd3dDevice->Present(NULL,NULL,NULL,NULL);

	_pd3dDevice->SetRenderTarget(0, _Old);
	_pd3dDevice->SetDepthStencilSurface( _OldDepthStencil );
	
	PerformanceManager::getSingleton()->Update(string("PerlinGPUH"));
}

void GPUGertnerNSimulate( float Time, IDirect3DSurface9 *_Surface, IDirect3DSurface9*_Depth )
{
	PerformanceManager::getSingleton()->Reset(string("GerstnerGPUN"));	
	IDirect3DSurface9*	_OldDepthStencil;
	IDirect3DSurface9*	_Old;

	D3DXMATRIX matProjection;
	D3DXMatrixIdentity(&matProjection);

	_pd3dDevice->GetRenderTarget(0, &_Old );

	_pd3dDevice->GetDepthStencilSurface( &_OldDepthStencil );

	_pd3dDevice->SetRenderTarget(0, _Surface);
	_pd3dDevice->SetDepthStencilSurface( _Depth );

	_pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_XRGB(100,100,100),	1.0f, 0);

	_pd3dDevice->BeginScene();
	_pd3dDevice->SetTransform(D3DTS_WORLD, &matProjection);
	_pd3dDevice->SetTransform(D3DTS_PROJECTION, &matProjection);

	_pd3dDevice->SetVertexDeclaration( _pgVertexDeclaration );
	_pd3dDevice->SetStreamSource( 0, _pgVertexBuffer, 0, sizeof( FloorVertex ) );

	_GerstnerEffect->SetFloat( m_hgTime, Time );	
	_GerstnerEffect->SetTechnique( "GerstnerNormal" );	

	UINT cPasses;
	_GerstnerEffect->Begin( &cPasses, 0 );

	for( UINT p = 0; p < cPasses; ++p )
	{
		_GerstnerEffect->BeginPass( p );

		_GerstnerEffect->CommitChanges();
		_pd3dDevice->DrawPrimitive( D3DPT_TRIANGLELIST, 0, 2);

		_GerstnerEffect->EndPass();
	}
	_GerstnerEffect->End();

	_pd3dDevice->EndScene();
	_pd3dDevice->Present(NULL,NULL,NULL,NULL);

	_pd3dDevice->SetRenderTarget(0, _Old);
	_pd3dDevice->SetDepthStencilSurface( _OldDepthStencil );
	PerformanceManager::getSingleton()->Update(string("GerstnerGPUN"));
}

void GPUPerlinNSimulate( float Time, IDirect3DSurface9 *_Surface, IDirect3DSurface9*_Depth )
{
	PerformanceManager::getSingleton()->Reset(string("PerlinGPUN"));	
	IDirect3DSurface9*	_OldDepthStencil;
	IDirect3DSurface9*	_Old;

	D3DXMATRIX matProjection;
	D3DXMatrixIdentity(&matProjection);

	_pd3dDevice->GetRenderTarget(0, &_Old );

	_pd3dDevice->GetDepthStencilSurface( &_OldDepthStencil );

	_pd3dDevice->SetRenderTarget(0, _Surface);
	_pd3dDevice->SetDepthStencilSurface( _Depth );

	_pd3dDevice->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_XRGB(100,100,100),	1.0f, 0);

	_pd3dDevice->BeginScene();
	_pd3dDevice->SetTransform(D3DTS_WORLD, &matProjection);
	_pd3dDevice->SetTransform(D3DTS_PROJECTION, &matProjection);

	_pd3dDevice->SetVertexDeclaration( _ppVertexDeclaration );
	_pd3dDevice->SetStreamSource( 0, _ppVertexBuffer, 0, sizeof( FloorVertex ) );

	_PelinEffect->SetFloat( m_hpTime, Time );	
	_PelinEffect->SetTechnique( "PerlinNormal" );	
	_PelinEffect->SetTexture( m_hpTex, _PerlinTex );	

	UINT cPasses;
	_PelinEffect->Begin( &cPasses, 0 );

	for( UINT p = 0; p < cPasses; ++p )
	{
		_PelinEffect->BeginPass( p );

		_PelinEffect->CommitChanges();
		_pd3dDevice->DrawPrimitive( D3DPT_TRIANGLELIST, 0, 2);

		_PelinEffect->EndPass();
	}
	_PelinEffect->End();

	_pd3dDevice->EndScene();
	_pd3dDevice->Present(NULL,NULL,NULL,NULL);

	_pd3dDevice->SetRenderTarget(0, _Old);
	_pd3dDevice->SetDepthStencilSurface( _OldDepthStencil );

	PerformanceManager::getSingleton()->Update(string("PerlinGPUN"));
}