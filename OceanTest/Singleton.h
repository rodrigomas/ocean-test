#pragma	once
#include <windows.h>

template <typename T> class Singleton
{
private:
	const Singleton& operator=(const Singleton&	src);

protected:
	
	static T* _Instance;
	static volatile	LONG _Mutex;

	static T* GetWkr()
	{
		T* pTmp	= NULL;
		try
		{
			static T tVar;
			pTmp = &tVar;
		}
		catch(...)
		{			
			pTmp = NULL;
		}
		return pTmp;
	}


public:
	static T* getSingleton()
	{
		if (_Instance == NULL)
		{
			while (::InterlockedExchange(&_Mutex, 1)	!= 0)
			{
				Sleep(1);
			}

			if (_Instance == NULL) // double-check
				_Instance = GetWkr();

			::InterlockedExchange(&_Mutex, 0);
		}

		return _Instance;
	}
};

template <typename  T> T* Singleton<T>::_Instance	= NULL;
template <typename  T> volatile LONG Singleton<T>::_Mutex = 0;