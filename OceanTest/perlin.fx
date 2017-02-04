float g_fTime = 0.0;
float Overcast = 1.1;
float g_fSize = 256.0;
texture g_Tex;

sampler Texture = sampler_state
{  
    Texture = <g_Tex>; 
    MipFilter = NONE; 
    MinFilter = LINEAR; 
    MagFilter = LINEAR; 

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
     float2 move = float2(0.0,1.0);
     float4 perlin = tex2D(Texture, (i.tc)+g_fTime*move)/2.0;
     perlin += tex2D(Texture, (i.tc)*2.0+g_fTime*move)/4.0;
     perlin += tex2D(Texture, (i.tc)*4.0+g_fTime*move)/8.0;
     perlin += tex2D(Texture, (i.tc)*8.0+g_fTime*move)/16.0;
     perlin += tex2D(Texture, (i.tc)*16.0+g_fTime*move)/32.0;
     perlin += tex2D(Texture, (i.tc)*32.0+g_fTime*move)/32.0;    
     
     perlin.rgb = 1.0 - pow(perlin.r, Overcast) * 2.0;
     perlin.a = 1.0;
	 
	return perlin;
}

float4 PShaderN(VS_OUTPUT i) : COLOR
{	
     float4 CA = tex2D(Texture, i.tc + float2(1.0/g_fSize,0.0));
	 
	 float4 BA = tex2D(Texture, i.tc + float2(0.0,1.0/g_fSize));
	 
	 float4 A = tex2D(Texture, i.tc);
	 
	float3 T = float3(1,0,CA.x-A.x);
	normalize(T);
	
	float3 B = float3(1,0,BA.x-A.x);
	normalize(B);	
	
	float3 N;
	
	N = cross(B,T);
	 
	return float4(N.x,N.y,N.z,1.0);
}

technique Perlin
{
	pass P0
	{
		vertexshader = compile vs_2_0 VShader();
		pixelshader = compile ps_2_0 PShader();  
	}
}

technique PerlinNormal
{
	pass P0
	{
		vertexshader = compile vs_2_0 VShader();
		pixelshader = compile ps_2_0 PShaderN();  
	}
}

