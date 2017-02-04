#pragma once

#ifdef _WIN32
#include <windows.h>
#endif

#include <string>
#include <ctime>
#include <stdarg.h>

#define LOG_DATE "%d/%m/%Y"
#define LOG_TIME "%H:%M:%S"

#ifndef REPORT
#define REPORT(p)      {  LogManager::getSingleton()->WriteData(__FILE__, __LINE__, __FUNCTION__, (p)); }
#define REPORT2(f,a)      {  LogManager::getSingleton()->WriteData(__FILE__, __LINE__, __FUNCTION__, f, a); }
#define REPORT3(f,a,b)      {  LogManager::getSingleton()->WriteData(__FILE__, __LINE__, __FUNCTION__, f, a, b); }
#define REPORT4(f,a,b,c)      {  LogManager::getSingleton()->WriteData(__FILE__, __LINE__, __FUNCTION__, f, a, b, c); }
#define REPORT5(f,a,b,c, d)      {  LogManager::getSingleton()->WriteData(__FILE__, __LINE__, __FUNCTION__, f, a, b, c, d); }
#endif

#ifndef LOGWRITE
#define LOGWRITE(p)      {  LogManager::getSingleton()->WriteText((p)); }
#endif

static const char LOG_VERSION [] = "0.1.1.2";
static const int _MAX_TIME_ = 256;
static char Tmp[ _MAX_TIME_ ];

#include "Singleton.h"

class LogManager : public Singleton<LogManager>
{			
	FILE *_File;

	const char* WriteDateTime( const char* Format )
	{
		//  Ponteiro para estrutura de Tempo e Data
		struct tm *ptr;
		//  Tempo e Data Locais
		time_t lt;
		//  Obtendo o Tempo e Data do Sistema
		lt = time( NULL );
		//  Colocando os Dados na estrutura
		ptr = localtime( &lt );
		//  Escrevendo a string
		strftime(Tmp, _MAX_TIME_, Format ,ptr);

		return Tmp;		
	}

public:

	void Initialize(const char* FileName);

	void WriteText( const char* Format, ... );

	void WriteData( const char* FileName, int Line, const char* Function, const char* Format, ... );

	void Finish()
	{
		if(_File != NULL)
			fclose(_File);

		_File = NULL;
	}

	void forceWrite( void )
	{
		if ( !_File )
			return;

		fflush(_File);
	}

	LogManager(void);

	~LogManager(void);
};