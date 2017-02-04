#include <windows.h>
#include <crtdbg.h>
#include <d3d9.h>
#include <d3dx9.h>
#include <vector>

#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d9_interop.h>
#include <cutil_inline.h>

#include "Enums.h"
#include "LogManager.h"
#include "Simulators.h"
#include "PerformanceManager.h"

#if (defined(DEBUG) || defined(_DEBUG)) && defined(_WIN32)
//#	include <vld.h>
#endif

#define CLASS_NAME "OceanTest"
#define WINDOW_CAPTION "Ocean Engine Test Program"

static int WIDTH = 800;
static int HEIGHT = 600;

int _TexWidth = 256;
int _TexHeight = 256;

#define ADD_TIME 5.0

bool g_Keys[256];

IDirect3D9 *_pD3D = NULL;
IDirect3DDevice9* _pd3dDevice = NULL;
ID3DXFont* _pd3dxFont = NULL;
unsigned int          g_iAdapter;

IDirect3DTexture9**	_Textures = NULL;	
IDirect3DSurface9**	_Surfaces = NULL;
IDirect3DSurface9**	_Depths = NULL;
LPD3DXSPRITE _Sprite = NULL;
int _nTextures = 0;
int _iNormal = 0;
int _iCores = 4;
int _iWaves = 20;
int _SimTime = 0;
char _FileName[512] = "Measure.csv";
char _SaveFileName[512] = "";
bool _Save = false;

SimType _NormalType = FFT;
SimType _HeightType = FFT;
SimMode _Mode = CPU;
//SimType _NormalType = Gerstner;
//SimType _HeightType = Gerstner;
//SimMode _Mode = CUDA;

void FinishData();

float UpdateTime()
{
	static unsigned int previousTime = timeGetTime();
	unsigned int currentTime = timeGetTime();
	unsigned int elapsedTime = currentTime - previousTime;
	previousTime = currentTime;
	return (float)(elapsedTime)*0.001f;
}

HRESULT CreateFont2D()
{
	//
	// To create a Windows friendly font using only a point size, an 
	// application must calculate the logical height of the font.
	// 
	// This is because functions like CreateFont() and CreateFontIndirect() 
	// only use logical units to specify height.
	//
	// Here's the formula to find the height in logical pixels:
	//
	//             -( point_size * LOGPIXELSY )
	//    height = ----------------------------
	//                          72
	//

	HDC hDC;
	HFONT hFont;
	int nHeight;
	int nPointSize = 7;
	char strFontName[] = "Verdana";

	hDC = GetDC( NULL );

	nHeight = -( MulDiv( nPointSize, GetDeviceCaps(hDC, LOGPIXELSY), 72 ) );


	hFont = CreateFont( nHeight, 0, 0, 0,
		FW_DONTCARE,
		false, false, false,
		DEFAULT_CHARSET,
		OUT_DEFAULT_PRECIS,
		CLIP_DEFAULT_PRECIS,
		DEFAULT_QUALITY,
		DEFAULT_PITCH | FF_DONTCARE,
		strFontName );

	if( hFont != NULL )
	{

		//if( FAILED( D3DXCreateFont( g_pd3dDevice, hFont, &g_pd3dxFont ) ) )
		if( FAILED( D3DXCreateFont( _pd3dDevice,nHeight, 5, 1, 1, false, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH, strFontName, &_pd3dxFont ) ) )
		{
			MessageBox(NULL,"Call to D3DXCreateFont failed!", "ERROR",MB_OK|MB_ICONEXCLAMATION);
			return E_FAIL;
		}

		DeleteObject( hFont );
	}
	else
	{
		MessageBox(NULL,"Call to CreateFont failed!", "ERROR",MB_OK|MB_ICONEXCLAMATION);
		return E_FAIL;
	}

	return S_OK;
}

HRESULT InitCUDA()
{
	
	printf("Starting CUDA                               ");	
	cudaD3D9SetDirect3DDevice(_pd3dDevice);
	cutilCheckMsg("cudaD3D9SetDirect3DDevice failed");
	printf("[DONE]\n");

	printf("Initialized CUDA D3DDevice = %p\n", _pd3dDevice);
	return S_OK;
}

HRESULT ReleaseCUDA()
{
	cudaThreadExit();
	cutilCheckMsg("cudaThreadExit failed");
	return S_OK;
}

int Initialize( HWND hWnd, bool Fullscreen, int w, int h)
{
	printf("Starting DirectX                            ");
	LogManager::getSingleton()->Initialize("log.log");

	// Create the D3D object.
	if( NULL == ( _pD3D = Direct3DCreate9( D3D_SDK_VERSION ) ) )
		return E_FAIL;

	// Find the first CUDA capable device
	for(g_iAdapter = 0; g_iAdapter < _pD3D->GetAdapterCount(); g_iAdapter++)
	{
		D3DCAPS9 caps;
		if (FAILED(_pD3D->GetDeviceCaps(g_iAdapter, D3DDEVTYPE_HAL, &caps)))
			// Adapter doesn't support Direct3D
			continue;
		D3DADAPTER_IDENTIFIER9 ident;
		int device;
		_pD3D->GetAdapterIdentifier(g_iAdapter, 0, &ident);
		cudaD3D9GetDevice(&device, ident.DeviceName);
		if (cudaSuccess == cudaGetLastError() )
			break;
	}

	// we check to make sure we have found a cuda-compatible D3D device to work on
	if(g_iAdapter == _pD3D->GetAdapterCount() ) 
	{
		printf("No CUDA-compatible Direct3D9 device available\n");
		printf("Test PASSED\n");
		// destroy the D3D device
		_pD3D->Release();
		exit(0);
	}

	// Create the D3D Display Device
	D3DDISPLAYMODE        d3ddm;    
	_pD3D->GetAdapterDisplayMode(g_iAdapter, &d3ddm);

	D3DPRESENT_PARAMETERS d3dpp;    
	ZeroMemory( &d3dpp, sizeof(d3dpp) );

	// Set up the structure used to create the D3DDevice
	if(Fullscreen)
	{
		d3dpp.Windowed = FALSE;
		d3dpp.BackBufferWidth = w;
		d3dpp.BackBufferHeight = h;
		d3dpp.hDeviceWindow          = hWnd;
	}else 
	{
		d3dpp.Windowed = TRUE;	
	//	d3dpp.BackBufferFormat = D3DFMT_UNKNOWN;
	}

	d3dpp.BackBufferCount        = 1;
	d3dpp.BackBufferFormat       = d3ddm.Format;

	d3dpp.SwapEffect = D3DSWAPEFFECT_DISCARD;	
	d3dpp.EnableAutoDepthStencil = TRUE;
	d3dpp.AutoDepthStencilFormat = D3DFMT_D24S8;
	d3dpp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
	//d3dpp.MultiSampleType = D3DMULTISAMPLE_4_SAMPLES;	

	if (FAILED (_pD3D->CreateDevice (g_iAdapter, D3DDEVTYPE_HAL, hWnd, 
		D3DCREATE_HARDWARE_VERTEXPROCESSING, 
		&d3dpp, &_pd3dDevice) ))
	{
		return E_FAIL;	
	}

	D3DADAPTER_IDENTIFIER9 ident;
	_pD3D->GetAdapterIdentifier(g_iAdapter, 0, &ident);

	printf("[DONE]\n");	

	printf("Initialized D3DDevice (%p): %s\n", _pd3dDevice, ident.Description);

	if( InitCUDA() != S_OK )
	{
		return E_FAIL;
	}

	_pd3dDevice->SetRenderState( D3DRS_CULLMODE, D3DCULL_NONE);
	_pd3dDevice->SetRenderState( D3DRS_LIGHTING, FALSE);
	_pd3dDevice->SetRenderState( D3DRS_ZENABLE, D3DZB_TRUE);
	_pd3dDevice->SetRenderState(D3DRS_SHADEMODE, D3DSHADE_GOURAUD);
	_pd3dDevice->SetRenderState(D3DRS_ZWRITEENABLE,true);
	_pd3dDevice->SetRenderState(D3DRS_ZFUNC,D3DCMP_LESSEQUAL);
	_pd3dDevice->SetRenderState(D3DRS_ALPHABLENDENABLE,true);	


	CreateFont2D();	

	if( SUCCEEDED( D3DXCreateSprite(_pd3dDevice,&_Sprite) ) )
	{
		// created OK
	}

	UpdateTime();	

	return S_OK;
}

void Cleanup()
{
	printf("Cleaning                                    ");
	switch(_HeightType)
	{
	case Gerstner:
		if( _Mode == CPU )
		{
			CPUGertnerUninitialize();
		} else if( _Mode == SHADER ) {
			GPUGertnerUninitialize();
		} else if( _Mode == CUDA ) {
			CUDAGertnerUninitialize();
		}
	break;

	case FFT:
		if( _Mode == CPU )
		{	
			CPUFFTUninitialize();
		} else if( _Mode == CUDA ) {
			CUDAFFTUninitialize();
		}
		break;

	case Perlin:
		if( _Mode == CPU )
		{
			CPUPerlinUninitialize();
		} else if( _Mode == SHADER ) {
			GPUPerlinUninitialize();
		} else if( _Mode == CUDA ) {
			CUDAPerlinUninitialize();
		}
		break;
	}


	switch(_NormalType)
	{
	case Gerstner:
		if( _Mode == CPU )
		{
			CPUGertnerUninitialize();
		} else if( _Mode == SHADER ) {
			GPUGertnerUninitialize();
		} else if( _Mode == CUDA ) {
			CUDAGertnerUninitialize();
		}
		break;

	case FFT:
		if( _Mode == CPU )
		{
			CPUFFTUninitialize();
		} else if( _Mode == CUDA )
		{
			CUDAFFTUninitialize();
		}
		break;

	case Perlin:
		if( _Mode == CPU )
		{
			CPUPerlinUninitialize();
		} else if( _Mode == SHADER ){
			GPUPerlinUninitialize();
		} else if( _Mode == CUDA ){
			CUDAPerlinUninitialize();
		}
		break;
	}

	SAFE_RELEASE(_pd3dxFont);
	SAFE_RELEASE(_pd3dDevice);	
	SAFE_RELEASE(_pD3D);

	for( register int i = 0; i < _nTextures ; i++)
	{
		cudaD3D9UnregisterResource(_Textures[i]);
		cutilCheckMsg("cudaD3D9UnregisterResource (g_texture_2d) failed");
		SAFE_RELEASE(_Textures[i]);
		SAFE_RELEASE(_Depths[i]);
	}

	ReleaseCUDA();
	printf("[DONE]\n");	
}

LRESULT WINAPI MsgProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg)
	{
	case WM_DESTROY:
		printf("[DONE]\n");
		FinishData();
		PostQuitMessage(0);
		Cleanup();
		return 0;

	case WM_KEYDOWN:							
		g_Keys[wParam] = TRUE;				

		if (g_Keys['Q'])
		{
			printf("[DONE]\n");
			FinishData();
			PostQuitMessage(0);
			Cleanup();
			return 0;	
		}

		break;	

	case WM_PAINT:
		break;

	case WM_KEYUP:
		g_Keys[wParam] = FALSE;	
		break;					
	}

	return DefWindowProc( hWnd, msg, wParam, lParam );
}

float UpdateFPS(float Time)
{
	static int _FrameCount = 0;
	static float _TimerCounter = 0;
	static float _FPS = 0;

	_TimerCounter += Time;
	_FrameCount++; 

	if( _TimerCounter >= 1.0f)
	{
		_FPS = _FrameCount / _TimerCounter;

		_TimerCounter = 0.0f;
		_FrameCount = 0;
	}

	return _FPS;
}

float Render(float Timer)
{
	float Time = UpdateTime();	
	float FPS = UpdateFPS(Time);

	int Count = 0;
	static char Text[256];
	RECT TextPos;
	SetRect( &TextPos, 5, 5, 0, 0 );

	if( SUCCEEDED(_pd3dDevice->BeginScene()))
	{		
		_pd3dDevice->SetRenderState( D3DRS_CULLMODE, D3DCULL_NONE);
		_pd3dDevice->Clear( 0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_XRGB(255, 255, 255), 1.0f, 0);


		sprintf_s(Text, 256, "FPS: %.4f", FPS );
		_pd3dxFont->DrawText(NULL, Text, -1, &TextPos, DT_NOCLIP, D3DXCOLOR( 0.0f, 0.0f, 0.0f, 1.0f ));

		TextPos.left = 150;
		sprintf_s(Text, 256, "Time: %.2f s", Timer );
		_pd3dxFont->DrawText(NULL, Text, -1, &TextPos, DT_NOCLIP, D3DXCOLOR( 0.0f, 0.0f, 0.0f, 1.0f ));

		
		TextPos.left = 20; TextPos.top = 20 + _TexHeight;
		_pd3dxFont->DrawText(NULL, "HeightMap", -1, &TextPos, DT_NOCLIP, D3DXCOLOR( 0.0f, 0.0f, 0.0f, 1.0f ));
		TextPos.left = 20 + _TexWidth + 10;
		_pd3dxFont->DrawText(NULL, "NormalMap", -1, &TextPos, DT_NOCLIP, D3DXCOLOR( 0.0f, 0.0f, 0.0f, 1.0f ));

		_pd3dDevice->EndScene();
	}

	_Sprite->Begin(D3DXSPRITE_ALPHABLEND);

	D3DXVECTOR3 pos;

	pos.x = 20.0f; pos.y= 20.0f; pos.z=0.0f;

	_Sprite->Draw(_Textures[0], NULL,NULL, &pos ,0xFFFFFFFF);

	pos.x= 20.0f + _TexWidth + 10.0f; pos.y=20.0f; pos.z=0.0f;

	_Sprite->Draw(_Textures[1], NULL,NULL, &pos ,0xFFFFFFFF);

	_Sprite->End();
	
	_pd3dDevice->Present( NULL, NULL, NULL, NULL );

	return Time;
}

void InputAnalysis(int argc, char **argv)
{
	char Text[512];

	if( argc == 1)
		return;

	if( strncmp("-h",argv[1],2) == 0 )
	{
		printf("\n\nusage: oceantest [-opts?[=value]]\n\n");
		printf("\t-size=[integer],[integer]\tTexture size(width, height)\n");
		printf("\t-cores=[integer]\tNumber of CPU cores\n");
		printf("\t-waves=[integer]\tNumber of Gerstner Waves\n");
		printf("\t-time=[integer]\tTime to simulate\n");
		printf("\t-mode=[string]\tSilumation Mode: CUDA, CPU, GPGPU\n");
		printf("\t-heightmap=[string]\tHeightmap synthesis technique: Gerstner, Perlin FFT\n");
		printf("\t-normalmap=[string]\tNormalmap synthesis technique: Gerstner, Perlin FFT\n");
		printf("\t-output=[string]\tOutput filename\n");
		printf("\t-save\tSave the images in DDS file format\n");
		printf("\t-h\tShow this help\n");
		printf("\n");
		scanf("%*s");
		exit(0);
		return;
	}

	for( register int i = 1; i < argc ; i++)
	{
		if( strncmp("-size=",argv[i],6) == 0 )
		{
			sscanf(argv[i], "-size=%d,%d", &_TexWidth, &_TexHeight);
			continue;
		}

		if( strncmp("-cores=",argv[i],7) == 0 )
		{
			sscanf(argv[i], "-cores=%d", &_iCores);
			continue;
		}

		if( strncmp("-waves=",argv[i],7) == 0 )
		{
			sscanf(argv[i], "-waves=%d", &_iWaves);
			continue;
		}

		if( strncmp("-time=",argv[i],6) == 0 )
		{
			sscanf(argv[i], "-time=%d", &_SimTime);
			continue;
		}

		if( strncmp("-mode=",argv[i],6) == 0 )
		{
			sscanf(argv[i], "-mode=%s", Text);

			if(strcmp(Text,"CUDA") == 0)
			{
				_Mode = CUDA;
			} else if (strcmp(Text,"CPU") == 0)
			{
				_Mode = CPU;
			} else if (strcmp(Text,"GPGPU") == 0)
			{
				_Mode = SHADER;
			}

			continue;
		}

		if( strncmp("-heightmap=",argv[i],11) == 0 )
		{
			sscanf(argv[i], "-heightmap=%s", Text);

			if(strcmp(Text,"Gerstner") == 0)
			{
				_HeightType = Gerstner;
			} else if (strcmp(Text,"Perlin") == 0)
			{
				_HeightType = Perlin;
			} else if (strcmp(Text,"FFT") == 0)
			{
				_HeightType = FFT;
			}

			continue;
		}

		if( strncmp("-normalmap=",argv[i],11) == 0 )
		{
			sscanf(argv[i], "-normalmap=%s", Text);

			if(strcmp(Text,"Gerstner") == 0)
			{
				_NormalType = Gerstner;
			} else if (strcmp(Text,"Perlin") == 0)
			{
				_NormalType = Perlin;
			} else if (strcmp(Text,"FFT") == 0)
			{
				_NormalType = FFT;
			}

			continue;
		}

		if( strncmp("-output=",argv[i],8) == 0 )
		{
			sscanf(argv[i], "-output=%s", &_FileName);
			continue;
		}

		if( strncmp("-save",argv[i],5) == 0 )
		{
			_Save = true;
			continue;
		}

	}
}

void SetupSimulators()
{
	_nTextures = 2;
	_iNormal = 1;	

	_Textures = new IDirect3DTexture9*[_nTextures];
	_Surfaces = new IDirect3DSurface9*[_nTextures];
	_Depths = new IDirect3DSurface9*[_nTextures];

	IDirect3DTexture9* pTexture;
	for( register int i = 0 ; i < _nTextures; i++)
	{
		_Textures[i] = NULL;

		if( _Mode == SHADER )
		{
			if( FAILED(_pd3dDevice->CreateTexture(_TexWidth,_TexHeight, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A32B32G32R32F, D3DPOOL_DEFAULT, &_Textures[i],NULL)))
			{
				continue;
			}
		} else 
		{
			if( FAILED(_pd3dDevice->CreateTexture(_TexWidth,_TexHeight, 1, D3DUSAGE_DYNAMIC, D3DFMT_A32B32G32R32F, D3DPOOL_DEFAULT, &_Textures[i],NULL)))
			{
				continue;
			}
		}
		
		_Textures[i]->GetSurfaceLevel( 0,&_Surfaces[i] );

		_pd3dDevice->CreateDepthStencilSurface(_TexWidth,_TexHeight,D3DFMT_D24S8, D3DMULTISAMPLE_NONE, 0, true, &_Depths[i], NULL );

		cudaD3D9RegisterResource(_Textures[i], cudaD3D9RegisterFlagsNone);
		cutilCheckMsg("cudaD3D9RegisterResource (_Textures[i]) failed");
	}	

	switch(_HeightType)
	{
	case Gerstner:
		if( _Mode == CPU )
		{
			CPUGertnerInitialize();
		} else if( _Mode == SHADER ) {

		} else if( _Mode == CUDA ) {
			CUDAGertnerInitialize();
		}
		break;

	case FFT:
		if( _Mode == CPU )
		{
			CPUFFTInitialize();
		} else if( _Mode == CUDA ) {
			CUDAFFTInitialize();
		}
		break;

	case Perlin:
		if( _Mode == CPU )
		{
			CPUPerlinInitialize(32);
		} else if( _Mode == SHADER ) {

		} else if( _Mode == CUDA ) {
			CUDAPerlinInitialize(32);
		}
		break;
	}


	switch(_NormalType)
	{
	case Gerstner:
		if( _Mode == CPU )
		{
			CPUGertnerInitialize();
		} else if( _Mode == SHADER ) {
			GPUGertnerInitialize();
	} else if( _Mode == CUDA ) {
			CUDAGertnerInitialize();
		}
		break;

	case FFT:
		if( _Mode == CPU )
		{
			CPUFFTInitialize();
		} else if( _Mode == CUDA ) {
			CUDAFFTInitialize();
		}
		break;

	case Perlin:
		if( _Mode == CPU )
		{
			CPUPerlinInitialize(32);
		} else if( _Mode == SHADER ) {
			GPUPerlinInitialize(32, _Textures[0]);
		} else if( _Mode == CUDA ) {
			CUDAPerlinInitialize(32);
		}
		break;
	}
}

void Simulate(float time)
{
	if( _Mode == CUDA)
	{
		switch(_HeightType)
		{
		case Gerstner:
			CUDAGertnerHSimulate(time, _Textures[0]);
			break;

		case FFT:
			CUDAFFTHSimulate(time, _Textures[0]);
			break;

		case Perlin:
			CUDAPerlinHSimulate(time, _Textures[0]);
			break;
		}

		switch(_NormalType)
		{
		case Gerstner:
			CUDAGertnerNSimulate(time, _Textures[1]);
			break;

		case FFT:
			CUDAFFTNSimulate(time, _Textures[0], _Textures[1]);
			break;

		case Perlin:
			CUDAPerlinNSimulate(time, _Textures[0], _Textures[1]);
			break;
		}
	} else if( _Mode == CPU)
	{
		switch(_HeightType)
		{
		case Gerstner:
			CPUGertnerHSimulate(time, _Textures[0]);
			break;

		case FFT:
			CPUFFTHSimulate(time, _Textures[0]);
			break;

		case Perlin:
			CPUPerlinHSimulate(time, _Textures[0]);
			break;
		}

		switch(_NormalType)
		{
		case Gerstner:
			CPUGertnerNSimulate(time, _Textures[1]);
			break;

		case FFT:
			CPUFFTNSimulate(time, _Textures[1]);
			break;

		case Perlin:
			CPUPerlinNSimulate(time, _Textures[0], _Textures[1]);
			break;
		}

	} else if( _Mode == SHADER)
	{
		switch(_HeightType)
		{
		case Gerstner:
			GPUGertnerHSimulate(time, _Surfaces[0], _Depths[0]);
			break;

		case Perlin:
			GPUPerlinHSimulate(time, _Surfaces[0], _Depths[0]);
			break;
		}

		switch(_NormalType)
		{
		case Gerstner:
			GPUGertnerNSimulate(time, _Surfaces[1], _Depths[1]);
			break;

		case Perlin:
			GPUPerlinNSimulate(time, _Surfaces[1], _Depths[1]);
			break;
		}

	}
}

void FinishData()
{	
	printf("Exporting data                              ");
	FILE *File = fopen(_FileName,"w");

	int nelem = 0;
	int StartCount = 2;
	bool Found = false;
	char COLLUMNS[] = "ABCDEFGHIJKLMNOPQRSTUWXYZ";
	int size;
	std::list<double> **lists = PerformanceManager::getSingleton()->getLists(size);
	std::vector<std::string> &names = PerformanceManager::getSingleton()->getListsNames();

	if (size == 0)
		return;

	std::list<double>::iterator *iter = new std::list<double>::iterator[size];
	std::list<double>::iterator *eiter = new std::list<double>::iterator[size];

	for(register int j = 0 ; j < size ; j++)
	{
		iter[j] = lists[j]->begin();
		eiter[j] = lists[j]->end();
		fprintf(File,"%s;", names[j].c_str());
	}

	fprintf(File,"\n");

	while(iter[0] != eiter[0])
	{
		for(register int j = 0 ; j < size ; j++)
		{			
			if(iter[j] != eiter[j])
			{
				if( names[j] == "Time" && *iter[j] >= ADD_TIME && Found == false )
				{
					StartCount = nelem;
					Found = true;
				}			

				fprintf(File,"%lf;", *iter[j]);
				iter[j]++;
			}
			else
				fprintf(File,"0.0;");						
		}
		fprintf(File,"\n");
		nelem++;
	}

	for(register int j = 0 ; j < size ; j++)
	{
		if( names[j] == "Time" )
		{
			fprintf(File,"AVERAGE;");
		}else
		{
			fprintf(File,"=AVERAGE(%c%d:%c%d);", COLLUMNS[j], StartCount+1, COLLUMNS[j], nelem + 1);
		}
	}

	fprintf(File,"\n");

	for(register int j = 0 ; j < size ; j++)
	{
		if( names[j] == "Time" )
		{
			fprintf(File,"STDEV;");
		}else
		{
			fprintf(File,"=STDEV(%c%d:%c%d);", COLLUMNS[j], StartCount+1, COLLUMNS[j], nelem + 1);
		}
	}

	fprintf(File,"\n");

	fclose(File);

	SAFE_ARRAY_DELETE(lists);
	SAFE_ARRAY_DELETE(iter);
	SAFE_ARRAY_DELETE(eiter);

	printf("[DONE]\n");
}

int main(int argc, char **argv)
{
#if (defined(DEBUG) || defined(_DEBUG)) && defined(_WIN32)
	_CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
	//#	include <vld.h>
#endif

	printf(WINDOW_CAPTION);
	printf("\n\nThis program is part of Rodrigo Marques A. Silva Msc. Thesis\n\n");

	printf("Analysing input options                     ");
	InputAnalysis(argc,argv);	
	printf("[DONE]\n");

	printf("Creating Window                             ");
	WNDCLASSEX wc = { sizeof(WNDCLASSEX),
		CS_CLASSDC,
		MsgProc,
		0,
		0,
		GetModuleHandle(NULL),
		NULL, NULL, NULL, NULL,
		CLASS_NAME,
		NULL};

	wc.hCursor = LoadCursor(NULL, IDC_ARROW);

	RegisterClassEx(&wc);

	int Width = ::GetSystemMetrics(SM_CXFULLSCREEN);
	int Height = ::GetSystemMetrics(SM_CYFULLSCREEN);

	HWND hWnd = CreateWindow( CLASS_NAME,
		WINDOW_CAPTION,
		WS_OVERLAPPEDWINDOW,
		(Width - WIDTH) / 2, (Height - HEIGHT) / 2, WIDTH, HEIGHT,
		GetDesktopWindow(),
		NULL,
		wc.hInstance,
		NULL
		);

	printf("[DONE]\n");
	

	if( FAILED(Initialize(hWnd, false, WIDTH, HEIGHT)) )
	{
		MessageBoxA(hWnd, "FAILED", WINDOW_CAPTION, 0);
		return 0;
	}		

	PerformanceManager::getSingleton()->AddPerformaceCounter(std::string("Time"));
	PerformanceManager::getSingleton()->AddPerformaceCounter(std::string("Interval"));	

	printf("Initializing Simulators                     ");	
	SetupSimulators();
	printf("[DONE]\n");

	char strPathName[_MAX_PATH];
	::GetModuleFileName(NULL, strPathName, _MAX_PATH);

	for(register int i = strlen(strPathName) - 1; i >= 0 ; i-- )
	{
		if( strPathName[i] == '\\')
		{
			strPathName[i+1] = '\0';
			break;
		}
	}

	::SetCurrentDirectory(strPathName);

	ShowWindow(hWnd, SW_SHOW);
	UpdateWindow(hWnd);	

	MSG msg;
	ZeroMemory(&msg, sizeof(msg));
	
	PerformanceManager::getSingleton()->Reset(std::string("Time"));

	printf("Running                                     ");	

	while( msg.message != WM_QUIT )
	{
		if( PeekMessage(&msg, NULL, 0, 0, PM_REMOVE) )
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		} else 
		{
			static double Time = 0;
			static int Index = 0;

			PerformanceManager::getSingleton()->Update(std::string("Time"));

			PerformanceManager::getSingleton()->Reset(std::string("Interval"));
			Simulate(Time);						
			Render(Time);
			Time += PerformanceManager::getSingleton()->Update(std::string("Interval"));	

			if(_Save)
			{
				sprintf_s(_SaveFileName, 512, "HeightMap%d.jpg", Index);
				D3DXSaveTextureToFile(_SaveFileName, D3DXIFF_DDS, _Textures[0], NULL);

				sprintf_s(_SaveFileName, 512, "NormalMap%d.jpg", Index);
				D3DXSaveTextureToFile(_SaveFileName, D3DXIFF_DDS, _Textures[1], NULL);
			}

			if ( _SimTime > 0 && Time > _SimTime )
			{
				printf("[DONE]\n");
				FinishData();
				PostQuitMessage(0);
				Cleanup();
			}

			Index++;
		}
	}

	UnregisterClass(CLASS_NAME, wc.hInstance);
	printf("\n\n");
	return 0;
}