#include "PerformanceTimer.h"

void PerformanceTimer::reset()
{
	unsigned __int64 pf;
	QueryPerformanceFrequency( (LARGE_INTEGER *)&pf );
	freq_ = 1.0 / (double)pf;
	QueryPerformanceCounter( (LARGE_INTEGER *)&baseTime_ );
}

double PerformanceTimer::seconds()
{
	unsigned __int64 val;
	
	QueryPerformanceCounter( (LARGE_INTEGER *)&val );
	return (val - baseTime_) * freq_;
}