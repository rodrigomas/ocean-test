#pragma	once

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)      { if (p) { (p)->Release(); (p)=NULL; } }
#endif

#ifndef SAFE_DELETE
#define SAFE_DELETE(p)      { if (p) { delete (p); (p)=NULL; } }
#endif

#ifndef SAFE_ARRAY_DELETE
#define SAFE_ARRAY_DELETE(p)      { if (p) { delete [] (p); (p)=NULL; } }
#endif

typedef enum __SimTypes__
{
	Gerstner,
	FFT,
	Perlin
} SimType;


typedef enum __SimModes__
{
	CPU,
	MCPU,
	CUDA,
	SHADER
} SimMode;