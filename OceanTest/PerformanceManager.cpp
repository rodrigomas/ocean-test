#include "PerformanceManager.h"
#include "LogManager.h"

PerformanceManager::PerformanceManager(void)
{
	//Empty
}

PerformanceManager::~PerformanceManager(void)
{
	_PerformanceInfos.clear();
}

void PerformanceManager::AddPerformaceCounter( std::string &Name )
{
	struct __PerformanceInfo__ info;

	info._Timer.reset();

	_PerformanceInfos.insert( std::pair<std::string, struct __PerformanceInfo__>(Name, info) );
}

void PerformanceManager::RemovePerformaceCounter( std::string &Name )
{
	stdext::hash_map <std::string, struct __PerformanceInfo__>::iterator iter;

	if( (iter = _PerformanceInfos.find(Name)) != _PerformanceInfos.end() )
	{
		_PerformanceInfos.erase(iter);
	} else 
	{
		REPORT2("Item not exits (%s)", Name.c_str());
	}
}

void PerformanceManager::ClearPerformaceCounterList( std::string &Name )
{
	stdext::hash_map <std::string, struct __PerformanceInfo__>::iterator iter;

	if( (iter = _PerformanceInfos.find(Name)) != _PerformanceInfos.end() )
	{
		iter->second._Values.clear();
	} else 
	{
		REPORT2("Item not exits (%s)", Name.c_str());
	}
}

PerformanceTimer &PerformanceManager::getTimer( std::string &Name )
{
	stdext::hash_map <std::string, struct __PerformanceInfo__>::iterator iter;

	if( (iter = _PerformanceInfos.find(Name)) != _PerformanceInfos.end() )
	{
		return iter->second._Timer;
	} else 
	{
		REPORT2("Item not exits (%s)", Name.c_str());	

		AddPerformaceCounter(Name);

		return _PerformanceInfos[Name]._Timer;
	}
}

std::list<double> & PerformanceManager::getValueList( std::string &Name )
{
	stdext::hash_map <std::string, struct __PerformanceInfo__>::iterator iter;

	if( (iter = _PerformanceInfos.find(Name)) != _PerformanceInfos.end() )
	{
		return iter->second._Values;
	} else 
	{
		REPORT2("Item not exits (%s)", Name.c_str());

		AddPerformaceCounter(Name);

		return _PerformanceInfos[Name]._Values;
	}
}

void PerformanceManager::Reset( std::string &Name )
{
	PerformanceTimer &t = getTimer(Name);

	t.reset();
}

double PerformanceManager::Update( std::string &Name )
{
	PerformanceTimer &t = getTimer(Name);
	std::list<double> &l = getValueList(Name);

	double v = t.seconds();

	l.push_back(v);

	return v;
}

std::list<double>** PerformanceManager::getLists(int &size)
{
	int n = 0;
	stdext::hash_map <std::string, struct __PerformanceInfo__>::iterator iter;

	iter = _PerformanceInfos.begin();

	while(iter != _PerformanceInfos.end())
	{
		n++;
		iter++;
	}
	
	if(n == 0)
		return NULL;

	size = n;
	std::list<double>** lists = new  std::list<double>*[size];

	iter = _PerformanceInfos.begin();
	for( register int i = 0; iter != _PerformanceInfos.end(); i++, iter++)
	{
		lists[i] = &iter->second._Values;
	}

	return lists;

}

std::vector<std::string> PerformanceManager::getListsNames()
{
	std::vector<std::string> lt;
	stdext::hash_map <std::string, struct __PerformanceInfo__>::iterator iter;

	iter = _PerformanceInfos.begin();

	while(iter != _PerformanceInfos.end())
	{
		lt.push_back(iter->first);
		iter++;
	}

	return lt;
}