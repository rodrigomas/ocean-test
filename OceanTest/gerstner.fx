float g_fTime = 0.0;
float g_fSize = 256.0;
int nWaves = 20;

struct Wave {
  float freq;  // 2*PI / wavelength
  float q;
  float amp;   // amplitude
  float phase; // speed * 2*PI / wavelength
  float2 dir;
};

#define NWAVES 20
Wave wave[NWAVES] = {
	{ 1.0, 0.5, 1.0, 0.5, float2(-1, 0) },
	{ 2.0, 0.5, 0.5, 1.7, float2(-0.7, 0.67) },
	{ 3.0, 0.5, 0.5, 1.7, float2(-0.5, 0.57) },
	{ 4.0, 0.5, 0.5, 1.7, float2(-0.89, 0.47) },
	{ 5.0, 0.5, 1.0, 0.5, float2(-1, 0) },
	{ 6.0, 0.5, 0.5, 1.7, float2(-0.7, 0.27) },
	{ 7.0, 0.5, 0.5, 1.7, float2(-0.0, 0.37) },
	{ 8.0, 0.5, 0.5, 1.7, float2(-0.4, 0.17) },	
	{ 9.0, 0.5, 0.5, 1.7, float2(-0.3, 0.97) },
	{ 1.1, 0.5, 0.5, 1.7, float2(-0.6, 0.87) },	
	{ 2.2, 0.5, 1.0, 0.5, float2(-1, 0) },
	{ 5.4, 0.5, 0.5, 1.7, float2(-0.3, 0.07) },
	{ 9.5, 0.5, 0.5, 1.7, float2(-0.07, 0.57) },
	{ 2.6, 0.5, 0.5, 1.7, float2(-0.2, 0.87) },
	{ 1.7, 0.5, 1.0, 0.5, float2(-1.4, 0) },
	{ 2.8, 0.5, 0.5, 1.7, float2(-0.56, 0.787) },
	{ 2.9, 0.5, 0.5, 1.7, float2(-0.77, 0.76) },
	{ 5.1, 0.5, 0.5, 1.7, float2(-0.9, 0.74) },	
	{ 2.6, 0.5, 0.5, 1.7, float2(-0.8, 0.73) },
	{ 2.7, 0.5, 0.5, 1.7, float2(-0.6, 0.72) }	
};

struct VS_INPUT
{
    float3  Pos     : POSITION;
    float2  tc     : TEXCOORD0;
};

struct VS_OUTPUT
{
	float4  Pos			: POSITION;
	float2  tc			: TEXCOORD0;	
};

VS_OUTPUT VShader(VS_INPUT i)
{
	VS_OUTPUT o;
	
	o.Pos = float4(i.Pos,1.0);
	o.tc = i.tc;
	
	return o;
}


float4 PShader(VS_OUTPUT i) : COLOR
{	
	float3 Position = float3(i.tc.x *g_fSize,0,i.tc.y *g_fSize);
	float3 P0 = float3(Position);
	
	for(int i = 0; i< nWaves; ++i)
	{				
		float angle = wave[i].freq * dot(wave[i].dir, P0.xz) + wave[i].phase * g_fTime; 

		float Si = sin(angle);
		float C = cos(angle);

		Position.x += wave[i].q * wave[i].amp * wave[i].dir.x * C;
		Position.z += wave[i].q * wave[i].amp * wave[i].dir.y * C;
		Position.y += wave[i].amp * Si;
	}    

	return float4(Position.x,Position.y,Position.z,1.0);
}


float4 PNormalShader(VS_OUTPUT i) : COLOR
{	
	float3 Position = float3(i.tc.x *g_fSize,0,i.tc.y *g_fSize);
	float3 P0 = float3(Position);
	float3 N = float3(0,0,0); 
	
	for(int i = 0; i< nWaves; ++i)
	{				
		float angle = wave[i].freq * dot(wave[i].dir, P0.xz) + wave[i].phase * g_fTime; 

		float Si = sin(angle);
		float C = cos(angle);

		Position.x += wave[i].q * wave[i].amp * wave[i].dir.x * C;
		Position.z += wave[i].q * wave[i].amp * wave[i].dir.y * C;
		Position.y += wave[i].amp * Si;
	}
	
	for(int i = 0; i< nWaves; ++i)
	{				
		float WA = wave[i].freq * wave[i].amp;				

		float angle = wave[i].freq * dot(float3(wave[i].dir.x,0.0,wave[i].dir.y),Position) + wave[i].phase * g_fTime; 				

		float Si = sin(angle);
		float C = cos(angle);

		N.x -= wave[i].dir.x * WA * C;		
		N.z -= wave[i].dir.y * WA * C;
		N.y -= wave[i].q * WA * Si;
	}    

	return float4(N.x,N.y,N.z,1.0);
}

technique Gerstner
{
	pass P0
	{
		vertexshader = compile vs_2_0 VShader();
		pixelshader = compile ps_3_0 PShader();  
	}
}

technique GerstnerNormal
{
	pass P0
	{
		vertexshader = compile vs_2_0 VShader();
		pixelshader = compile ps_3_0 PNormalShader();  
	}
}

