#pragma once

#include <windows.h>

class PerformanceTimer {
public:
	PerformanceTimer() {
		reset();
	}

	void reset();

	double seconds();

	double milliseconds() {
		return seconds() * 1000.0;
	}
private:
	double freq_;
	unsigned __int64 baseTime_;
};