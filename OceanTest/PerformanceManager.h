#pragma once

#include "Singleton.h"
#include "PerformanceTimer.h"

#include <hash_map>
#include <list>
#include <vector>

class PerformanceManager : public Singleton<PerformanceManager>
{
	struct __PerformanceInfo__
	{
		PerformanceTimer _Timer;
		std::list<double> _Values;
	};

	stdext::hash_map<std::string,struct __PerformanceInfo__> _PerformanceInfos;
public:

	std::vector<std::string> getListsNames();

	std::list<double>** getLists(int &size);
	
	void AddPerformaceCounter(std::string &Name);

	void RemovePerformaceCounter(std::string &Name);

	void ClearPerformaceCounterList(std::string &Name);

	PerformanceTimer &getTimer(std::string &Name);

	std::list<double> &getValueList(std::string &Name);

	void Reset(std::string &Name);

	double Update(std::string &Name);

	PerformanceManager(void);

	~PerformanceManager(void);
};
