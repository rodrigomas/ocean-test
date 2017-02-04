#include "Random.h"
#include <cmath>

//--------------------------------------------------------------------------------------
float Random::RandN()
{
	float S = 0.449871f, T = -0.386595f, A = 0.19600f, B = 0.25472f;
	float R1 = 0.27597f, R2 = 0.27846f;

	float U;
	float V;

	while( true )
	{
		U = RandU();
		V = RandU();

		V = 1.7156f * (V - 0.5f);

		float X = U - S;
		float Y = fabsf( V ) - T;
		float Q = X*X + Y*(A*Y - B*X);

		if( Q < R1 )
			return V/U;

		if( Q <= R2 && V*V <= -4.0f*logf(U) * U*U )
			return V/U;
	}
}

//--------------------------------------------------------------------------------------
float Random::RandU()
{
	static int Index = 0;

	Index++;

	if( Index > 278 )
		Index = 0;

	return Random_U[Index];
}