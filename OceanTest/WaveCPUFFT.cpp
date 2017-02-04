#include "WaveCPUFFT.h"

WaveModelStatisticalFFT::WaveModelStatisticalFFT(unsigned int uiGridResolution, D3DXVECTOR2 vSize) : 
m_uiGridResolution( uiGridResolution ),
m_vGridSize( vSize ),
m_H0( NULL ),
m_H ( NULL ),
m_h	( NULL ),
m_Dx( NULL ),
m_dx( NULL ),
m_Dy( NULL ),
m_dy( NULL ),
m_FFTPlan( NULL ),
m_prgRandomValues( NULL ),
m_fWaveChoppy( 0.0f ),
m_uiWindDirection( 0 ),
m_fWindSpeed( 10.0f ),
m_fWaveHeightConstant( 0.00000075f ),
m_bWaterDepthWaves( false ),
m_prgTerrainHeightMap( NULL )
{

	m_prgRandomValues = new fftw_complex[ m_uiGridResolution * m_uiGridResolution ];
}

WaveModelStatisticalFFT::~WaveModelStatisticalFFT(void)
{
	SAFE_ARRAY_DELETE( m_prgRandomValues );

	if( m_FFTPlan )	fftw_destroy_plan( m_FFTPlan );

	if( m_H0 )	fftw_free( m_H0 );
	if( m_H )	fftw_free( m_H );
	if( m_h )	fftw_free( m_h );

	if( m_Dx )	fftw_free( m_Dx );
	if( m_dx )	fftw_free( m_dx );
	if( m_Dy )	fftw_free( m_Dy );
	if( m_dy )	fftw_free( m_dy );

	if( m_Nx )	fftw_free( m_Nx );
	if( m_nx )	fftw_free( m_nx );
	if( m_Ny )	fftw_free( m_Ny );
	if( m_ny )	fftw_free( m_ny );
}

//--------------------------------------------------------------------------------------
// Calculate the wave model for time fTime.
//--------------------------------------------------------------------------------------
bool WaveModelStatisticalFFT::CalculateWaves( double fTime, bool Normals)
{
	// Should the choppy wave displacement be calculated. Not if m_fWaveChoppy is zero.
	bool bChoppy = false;
	if( m_fWaveChoppy > 0.0f )
		bChoppy = true;

	D3DXVECTOR2 K;
	int iHalfGridResolution = (int)(m_uiGridResolution/2);

	for( int m = -iHalfGridResolution; m < iHalfGridResolution; m++ )
	{
		K.y = (2.0f*M_PI * (float)m) / m_vGridSize.y;
		for( int n = -iHalfGridResolution; n < iHalfGridResolution; n++ )
		{
			K.x = (2.0f*M_PI * (float)n) / m_vGridSize.x;

			float k = D3DXVec2Length( &K );

			int x = n + iHalfGridResolution;
			int y = m + iHalfGridResolution;
			int Index = x + y * m_uiGridResolution;
			int IndexNegative = (m_uiGridResolution - x)%m_uiGridResolution 
				+((m_uiGridResolution - y)%m_uiGridResolution) * m_uiGridResolution;

			float fOmega = 0.0f;

			if( m_bWaterDepthWaves )
			{
				float fDepth = WATERLEVEL - m_prgTerrainHeightMap[ Index ];

				if( fDepth < 0.0f )
					fDepth = 0.0f;

				fOmega = sqrtf( GRAVITYCONSTANT * k * tanhf( k * fDepth ) );
			}
			else
				fOmega = sqrtf( GRAVITYCONSTANT * k );

			double fCosValue = cos( fOmega * fTime );
			double fSinValue = sin( fOmega * fTime );

			double H0K_Real		  = m_H0[ Index         ][REAL];			//H0(K).re
			double H0K_Imaginary  = m_H0[ Index         ][IMAGINARY];		//H0(K).im
			double H0NK_Real	  = m_H0[ IndexNegative ][REAL];			//H0(-K).re
			double H0NK_Imaginary = m_H0[ IndexNegative ][IMAGINARY];		//H0(-K).im

			// The height transform H is calculated like this from the formula 26 in the Tessendorf paper.
			// H(K,t) = H0( K) * e^( i*omega(k)*t ) + H'0(-K) e^( -i*omega(k)*t ) =
			//	      =  H0( K) * ( cos( i*omega(k)*t ) + i * sin( i*omega(k)*t ) )
			//	      + H'0(-K) * ( cos( i*omega(k)*t ) - i * sin( i*omega(k)*t ) ) =
			//	      = ( H0( K).re + H0( K).im ) * ( cos( i*omega(k)*t ) + i * sin( i*omega(k)*t ) )
			//	      + ( H0(-K).re - H0(-K).im ) * ( cos( i*omega(k)*t ) - i * sin( i*omega(k)*t ) ) =
			//	      = H0( K).re * ( cos( i*omega(k)*t ) + i * sin( i*omega(k)*t ) ) +
			//	      + H0( K).im * ( cos( i*omega(k)*t ) + i * sin( i*omega(k)*t ) ) +
			//	      + H0(-K).re * ( cos( i*omega(k)*t ) - i * sin( i*omega(k)*t ) ) -
			//	      - H0(-K).im * ( cos( i*omega(k)*t ) - i * sin( i*omega(k)*t ) ) =
			//
			//	Re:	 H0( K).re * cos( i*omega(k)*t ) - H0( K).im * sin( i*omega(k)*t ) +
			//	   + H0(-K).re * cos( i*omega(k)*t ) - H0(-K).im * sin( i*omega(k)*t )
			//
			//	Im:  H0( K).re * sin( i*omega(k)*t ) + H0( K).im * cos( i*omega(k)*t ) -
			//     - H0(-K).re * sin( i*omega(k)*t ) - H0(-K).im * cos( i*omega(k)*t )

			m_H[ Index ][REAL]	    = H0K_Real  * fCosValue - H0K_Imaginary   * fSinValue
				+ H0NK_Real * fCosValue - H0NK_Imaginary  * fSinValue;

			m_H[ Index ][IMAGINARY] = H0K_Real  * fSinValue + H0K_Imaginary   * fCosValue
				- H0NK_Real * fSinValue - H0NK_Imaginary  * fCosValue;


			// The normal transform N is calculated like this from the formula 20 in the Tessendorf paper.
			// N(x,t) = i * K * H(K,t) :
			//
			// Re:      -K * H(K,t).im 
			// Im:       K * H(K,t).re

			if( Normals )
			{
				m_Nx[ Index ][REAL]		 =-K.x * m_H[ Index ][IMAGINARY];
				m_Nx[ Index ][IMAGINARY] = K.x * m_H[ Index ][REAL];
				m_Ny[ Index ][REAL]		 =-K.y * m_H[ Index ][IMAGINARY];
				m_Ny[ Index ][IMAGINARY] = K.y * m_H[ Index ][REAL];
			}

			if( bChoppy )
			{
				// The displacement transform D is calculated like this from the formula 29 in the Tessendorf paper.
				// D(x,t) = -i * (K/k) * H(K,t) :
				//
				// Re:        (K/k) * H(K,t).im 
				// Im:      + (K/k) * H(K,t).re

				float KScaleX = 0.0f;
				float KScaleY = 0.0f;

				if( k != 0.0f )
				{
					KScaleX = K.x/k;
					KScaleY = K.y/k;
				}

				m_Dx[ Index ][REAL]		 = KScaleX * m_H[ Index ][IMAGINARY];
				m_Dx[ Index ][IMAGINARY] =-KScaleX * m_H[ Index ][REAL];
				m_Dy[ Index ][REAL]		 = KScaleY * m_H[ Index ][IMAGINARY];
				m_Dy[ Index ][IMAGINARY] =-KScaleY * m_H[ Index ][REAL];
			}

		}
	}
/*
	// Calculate the FFT
	fftw_execute( m_FFTPlan );
	fftw_execute_dft( m_FFTPlan, m_Nx, m_nx );
	fftw_execute_dft( m_FFTPlan, m_Ny, m_ny );

	if( bChoppy )
	{
		fftw_execute_dft( m_FFTPlan, m_Dx, m_dx );
		fftw_execute_dft( m_FFTPlan, m_Dy, m_dy );
	}

	D3DLOCKED_RECT sRect;
	if ( FAILED( pWaveBuffer->LockRect( 0, &sRect, NULL, D3DLOCK_DISCARD ) ) )
	{
//		m_pDebug->Log( "HEIGHTMAP::Create() - Failed to lock texture memory" );
		return false;
	}
	float *imageData = ( float* )sRect.pBits; // start of the memory block

	#pragma omp parallel for
	for( int y = 0; y <m_uiGridResolution; y++ )
	{
		for( int x = 0; x < m_uiGridResolution; x++ )
		{						
			if( (x+y)%2 != 0 )		// cos( ( n + m )*PI ) = -1 if (n + m) is odd otherwise 1.
				m_h [ x + y * m_uiGridResolution * 4 ][REAL] *= -1.0f;

			imageData[y * sRect.Pitch + x * 4 + 1 ] = (float)m_h [ x + y * m_uiGridResolution ][REAL];

			if( bChoppy )
			{
				if( (x+y)%2 == 0 )		// cos( ( n + m )*PI ) = -1 if (n + m) is odd otherwise 1 but remember the minus sign 
					// in the D equation so it will be the inverse.
				{
					m_dx[ x + y * m_uiGridResolution ][REAL] *= -1.0f;
					m_dy[ x + y * m_uiGridResolution ][REAL] *= -1.0f;
				}

				imageData[y * sRect.Pitch + x * 4 + 0 ]= (float)m_dx[ x + y * m_uiGridResolution ][REAL];
				imageData[y * sRect.Pitch + x * 4 + 2 ] = (float)m_dy[ x + y * m_uiGridResolution ][REAL];
			}			
		}
	}

	//Unlock the texture
	if ( FAILED( pWaveBuffer->UnlockRect( 0 ) ) )
	{
		//m_pDebug->Log( "HEIGHTMAP::Create() - Failed to unlock texture memory" );
		return false;
	}

	// Fill the normal buffer.	

	sRect;
	if ( FAILED( pNormalBuffer->LockRect( 0, &sRect, NULL, D3DLOCK_DISCARD ) ) )
	{
		//		m_pDebug->Log( "HEIGHTMAP::Create() - Failed to lock texture memory" );
		return false;
	}

	imageData = ( float* )sRect.pBits; // start of the memory block

	#pragma omp parallel for
	for( unsigned short y = 0; y < m_uiGridResolution; y++ )
	{
		for( unsigned short x = 0; x < m_uiGridResolution; x++ )
		{
			if( (x+y)%2 == 0 )
			{
				m_nx[ x + y * m_uiGridResolution ][REAL] *= -1.0f;
				m_ny[ x + y * m_uiGridResolution ][REAL] *= -1.0f;
			}

			D3DXVECTOR3 N = D3DXVECTOR3( (float)m_nx[ x + y * m_uiGridResolution ][REAL], 1.0f, (float)m_ny[ x + y * m_uiGridResolution ][REAL] );			
			D3DXVec3Normalize(&N, &N);

			imageData[y * sRect.Pitch + x * 4 + 0 ] = N.x;
			imageData[y * sRect.Pitch + x * 4 + 1 ] = N.y;
			imageData[y * sRect.Pitch + x * 4 + 2 ] = N.z;
		}
	}

	//Unlock the texture
	if ( FAILED( pNormalBuffer->UnlockRect( 0 ) ) )
	{
		//m_pDebug->Log( "HEIGHTMAP::Create() - Failed to unlock texture memory" );
		return false;
	}
*/
	return true;
}


//--------------------------------------------------------------------------------------
// Initialize the wave model.
//--------------------------------------------------------------------------------------
bool WaveModelStatisticalFFT::Initialize( void )
{
	// Allocate memory for the diffrent kind of transformations.
	m_H0 = (fftw_complex*)fftw_malloc( sizeof(fftw_complex) * m_uiGridResolution * m_uiGridResolution ); 
	m_H  = (fftw_complex*)fftw_malloc( sizeof(fftw_complex) * m_uiGridResolution * m_uiGridResolution );
	m_h  = (fftw_complex*)fftw_malloc( sizeof(fftw_complex) * m_uiGridResolution * m_uiGridResolution );

	m_Dx = (fftw_complex*)fftw_malloc( sizeof(fftw_complex) * m_uiGridResolution * m_uiGridResolution );
	m_dx = (fftw_complex*)fftw_malloc( sizeof(fftw_complex) * m_uiGridResolution * m_uiGridResolution );
	m_Dy = (fftw_complex*)fftw_malloc( sizeof(fftw_complex) * m_uiGridResolution * m_uiGridResolution );
	m_dy = (fftw_complex*)fftw_malloc( sizeof(fftw_complex) * m_uiGridResolution * m_uiGridResolution );

	m_Nx = (fftw_complex*)fftw_malloc( sizeof(fftw_complex) * m_uiGridResolution * m_uiGridResolution );
	m_nx = (fftw_complex*)fftw_malloc( sizeof(fftw_complex) * m_uiGridResolution * m_uiGridResolution );
	m_Ny = (fftw_complex*)fftw_malloc( sizeof(fftw_complex) * m_uiGridResolution * m_uiGridResolution );
	m_ny = (fftw_complex*)fftw_malloc( sizeof(fftw_complex) * m_uiGridResolution * m_uiGridResolution );

	// Create the FFTW plan. The plan transforms frequency based complex samples to
	// time based complex samples representing height.
	// Because the sample transform is hermitian is should be possible to tweak
	// the algorithm by using the complex to real plan in FFTW but there is some 
	// stupid bug in the hermitian based real output so the heights is not calculated
	// correctly. Until I find the bug I use the complex to complex transformation and 
	// just ignore the imaginary part because it will always be zero due to the hermitian.
	// The difference in performance is not that big on small grids anyway :)
	m_FFTPlan = fftw_plan_dft_2d( m_uiGridResolution,
		m_uiGridResolution,
		m_H,
		m_h,
		FFTW_BACKWARD,
		FFTW_MEASURE );

	// Fill the random number grid. The Random number grid is used when the wave is initialized.
	// These means that the wave parameters can be changed in real-time without the complete
	// randomized grid is changed.
	for( UINT m = 0; m < m_uiGridResolution; m++ )
	{
		for( UINT n = 0; n < m_uiGridResolution; n++ )
		{
			m_prgRandomValues[ n + m * m_uiGridResolution ][REAL]	   = Random::RandN();
			m_prgRandomValues[ n + m * m_uiGridResolution ][IMAGINARY] = Random::RandN();
		}
	}

	// Initialize the wave with the default parameters.
	InitializeWave();

	// Create the effect parameter handle.
	//if( !( m_hChoppy = pEffect->GetParameterByName( NULL, "g_fStaticalFFTChoppyLamda" ) ) )		return E_FAIL;

	return true;
}

//--------------------------------------------------------------------------------------
// Calculate the initial H0.
//--------------------------------------------------------------------------------------
void WaveModelStatisticalFFT::InitializeWave()
{
	D3DXVECTOR2 K;
	int iHalfGridResolution = (int)(m_uiGridResolution/2);

	for( int m = -iHalfGridResolution; m < iHalfGridResolution; m++ )
	{
		K.y = (2.0f*M_PI * (float)m) / m_vGridSize.y;

		for( int n = -iHalfGridResolution; n < iHalfGridResolution; n++ )
		{
			K.x = (2.0f*M_PI* (float)n) / m_vGridSize.x;

			UINT uiIndex = (n+iHalfGridResolution) + (m+iHalfGridResolution) * m_uiGridResolution;

			float fWaveSpectrumSqrt = sqrtf( WaveSpectrum_Phillips( K ) );

			// Formula 25 from the Tessendorf paper.
			m_H0[ uiIndex ][REAL]	   = INVSQRT2 * m_prgRandomValues[ uiIndex ][REAL]      * fWaveSpectrumSqrt;
			m_H0[ uiIndex ][IMAGINARY] = INVSQRT2 * m_prgRandomValues[ uiIndex ][IMAGINARY] * fWaveSpectrumSqrt;
		}
	}
}

//--------------------------------------------------------------------------------------
// Calculates the phillips wavespectrum
//--------------------------------------------------------------------------------------
float WaveModelStatisticalFFT::WaveSpectrum_Phillips(D3DXVECTOR2 K)
{
	D3DXVECTOR2 KNorm;

	D3DXVec2Normalize( &KNorm, &K );

	float KdotW = D3DXVec2Dot( &KNorm, &DIR[ m_uiWindDirection ] );
	float ksq	= D3DXVec2LengthSq( &K );

	if( ksq == 0.0f )
		return 0;

	float L	= ( m_fWindSpeed * m_fWindSpeed ) / GRAVITYCONSTANT;
	float l = L / 80.0f;

	float fPhillips = m_fWaveHeightConstant * ( expf( -1.0f / (ksq * L * L) ) / ( ksq * ksq ) ) * fabsf(KdotW) * fabsf(KdotW);
	float fDamping  = expf( -ksq * l * l );

	return fPhillips * fDamping;
}

//--------------------------------------------------------------------------------------
// Set wave constant to the given effect file.
//--------------------------------------------------------------------------------------
/*void WaveModelStatisticalFFT::SetWaveConstants( LPD3DXEFFECT pEffect )
{
HRESULT hr;

V( pEffect->SetFloat( m_hChoppy, m_fWaveChoppy ) );
}*/

//--------------------------------------------------------------------------------------
// Set the wave choppy constant.
//--------------------------------------------------------------------------------------
void WaveModelStatisticalFFT::SetWaveChoppy( float fWaveChoppy )
{
	m_fWaveChoppy = fWaveChoppy;
}

//--------------------------------------------------------------------------------------
// Set the wave height constant.
//--------------------------------------------------------------------------------------
void WaveModelStatisticalFFT::SetWaveHeightConstant( float fWaveHeightConstant )
{
	if( m_fWaveHeightConstant != fWaveHeightConstant )
	{
		m_fWaveHeightConstant = fWaveHeightConstant;
		InitializeWave();						// The wave height constant is changed
	}											// so we have to recalculate H0
}

//--------------------------------------------------------------------------------------
// Set the wind speed constant.
//--------------------------------------------------------------------------------------
void WaveModelStatisticalFFT::SetWindSpeed( float fWindSpeed )
{
	if( m_fWindSpeed != fWindSpeed )
	{
		m_fWindSpeed = fWindSpeed;
		InitializeWave();						// The wind speed is changed so
	}											// we have to recalculate H0.
}

//--------------------------------------------------------------------------------------
// Set the wind direction.
//--------------------------------------------------------------------------------------
void WaveModelStatisticalFFT::SetWindDirection( UINT uiWindDirection )
{
	if( m_uiWindDirection != uiWindDirection )
	{
		m_uiWindDirection = uiWindDirection;
		InitializeWave();						// The wind direction is changed so
	}											// we have to recalculate H0.
}

//--------------------------------------------------------------------------------------
// Get the wave choppy constant.
//--------------------------------------------------------------------------------------
float WaveModelStatisticalFFT::GetWaveChoppy()
{
	return m_fWaveChoppy;
}

//--------------------------------------------------------------------------------------
// Get the wave height constant.
//--------------------------------------------------------------------------------------
float WaveModelStatisticalFFT::GetWaveHeightConstant()
{
	return m_fWaveHeightConstant;
}

//--------------------------------------------------------------------------------------
// Get the wind speed constant.
//--------------------------------------------------------------------------------------
float WaveModelStatisticalFFT::GetWindSpeed()
{
	return m_fWindSpeed;
}

//--------------------------------------------------------------------------------------
// Get the wind direction.
//--------------------------------------------------------------------------------------
UINT WaveModelStatisticalFFT::GetWindDirection()
{
	return m_uiWindDirection;
}

//--------------------------------------------------------------------------------------
// Set if the waves should be based on the water depth or not.
//--------------------------------------------------------------------------------------
void WaveModelStatisticalFFT::SetWaterDepthWaves( bool bWaterDepthWaves )
{
	m_bWaterDepthWaves = bWaterDepthWaves;
}

//--------------------------------------------------------------------------------------
// Set the terrain height map.
//--------------------------------------------------------------------------------------
void WaveModelStatisticalFFT::SetTerrainHeightMap( float* prgTerrainHeightMap )
{
	m_prgTerrainHeightMap = prgTerrainHeightMap;
}

bool WaveModelStatisticalFFT::UpdateHeight(IDirect3DTexture9* pWaveBuffer)
{
	bool bChoppy = false;
	if( m_fWaveChoppy > 0.0f )
		bChoppy = true;

	// Calculate the FFT
	fftw_execute( m_FFTPlan );

	if( bChoppy )
	{
		fftw_execute_dft( m_FFTPlan, m_Dx, m_dx );
		fftw_execute_dft( m_FFTPlan, m_Dy, m_dy );
	}

	D3DLOCKED_RECT sRect;
	if ( FAILED( pWaveBuffer->LockRect( 0, &sRect, NULL, D3DLOCK_DISCARD ) ) )
	{
		//		m_pDebug->Log( "HEIGHTMAP::Create() - Failed to lock texture memory" );
		return false;
	}
	float *imageData = ( float* )sRect.pBits; // start of the memory block

	#pragma omp parallel for
	for( int y = 0; y <m_uiGridResolution; y++ )
	{
		for( int x = 0; x < m_uiGridResolution; x++ )
		{						
			if( (x+y)%2 != 0 )		// cos( ( n + m )*PI ) = -1 if (n + m) is odd otherwise 1.
				m_h [ x + y * m_uiGridResolution ][REAL] *= -1.0f;

			imageData[y * sRect.Pitch / 4 + x * 4 + 1 ] = (float)m_h [ x + y * m_uiGridResolution ][REAL];

			if( bChoppy )
			{
				if( (x+y)%2 == 0 )		// cos( ( n + m )*PI ) = -1 if (n + m) is odd otherwise 1 but remember the minus sign 
					// in the D equation so it will be the inverse.
				{
					m_dx[ x + y * m_uiGridResolution ][REAL] *= -1.0f;
					m_dy[ x + y * m_uiGridResolution ][REAL] *= -1.0f;
				}

				imageData[y * sRect.Pitch / 4 + x * 4 + 0 ]= (float)m_dx[ x + y * m_uiGridResolution ][REAL];
				imageData[y * sRect.Pitch / 4 + x * 4 + 2 ] = (float)m_dy[ x + y * m_uiGridResolution ][REAL];				
			} else 
			{
				imageData[y * sRect.Pitch / 4 + x * 4 + 0 ]= imageData[y * sRect.Pitch / 4 + x * 4 + 1 ];
				imageData[y * sRect.Pitch / 4 + x * 4 + 2 ] = imageData[y * sRect.Pitch / 4 + x * 4 + 1 ];		
			}
			imageData[y * sRect.Pitch / 4 + x * 4 + 3 ] = 1.0;
		}
	}

	//Unlock the texture
	if ( FAILED( pWaveBuffer->UnlockRect( 0 ) ) )
	{
		//m_pDebug->Log( "HEIGHTMAP::Create() - Failed to unlock texture memory" );
		return false;
	}

	return true;
}


bool WaveModelStatisticalFFT::UpdateNormals(IDirect3DTexture9* pNormalBuffer, bool Height)
{
	// Calculate the FFT
	
	if(Height)
		fftw_execute( m_FFTPlan );

	fftw_execute_dft( m_FFTPlan, m_Nx, m_nx );
	fftw_execute_dft( m_FFTPlan, m_Ny, m_ny );

	D3DLOCKED_RECT sRect;
	
	if ( FAILED( pNormalBuffer->LockRect( 0, &sRect, NULL, D3DLOCK_DISCARD ) ) )
	{
		//		m_pDebug->Log( "HEIGHTMAP::Create() - Failed to lock texture memory" );
		return false;
	}

	float *imageData = ( float* )sRect.pBits; // start of the memory block

	#pragma omp parallel for
	for( int y = 0; y < m_uiGridResolution; y++ )
	{
		for( int x = 0; x < m_uiGridResolution; x++ )
		{
			if( (x+y)%2 == 0 )
			{
				m_nx[ x + y * m_uiGridResolution ][REAL] *= -1.0f;
				m_ny[ x + y * m_uiGridResolution ][REAL] *= -1.0f;
			}

			D3DXVECTOR3 N = D3DXVECTOR3( (float)m_nx[ x + y * m_uiGridResolution ][REAL], 1.0f, (float)m_ny[ x + y * m_uiGridResolution ][REAL] );			
			D3DXVec3Normalize(&N, &N);

			imageData[y * sRect.Pitch / 4 + x * 4 + 0 ] = N.x;
			imageData[y * sRect.Pitch / 4 + x * 4 + 1 ] = N.y;
			imageData[y * sRect.Pitch / 4 + x * 4 + 2 ] = N.z;
			imageData[y * sRect.Pitch / 4 + x * 4 + 3 ] = 1.0;
		}
	}

	//Unlock the texture
	if ( FAILED( pNormalBuffer->UnlockRect( 0 ) ) )
	{
		//m_pDebug->Log( "HEIGHTMAP::Create() - Failed to unlock texture memory" );
		return false;
	}

	return true;
}