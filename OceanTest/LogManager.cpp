#include "LogManager.h"

LogManager::LogManager(void)
{
	_File = NULL;
}

LogManager::~LogManager(void)
{
	if(_File != NULL)
		fclose(_File);
}

void LogManager::Initialize(const char* FileName )
{
#if defined(_DEBUG) || defined(DEBUG)
	if(_File != NULL)
	{
		fclose(_File);
		_File = NULL;
	}

	_File = fopen(FileName, "r");

	if(_File == NULL)
	{
		_File = fopen(FileName, "w");
	} else 
	{
		fclose(_File);
		_File = fopen(FileName, "a");
	}

	//  Falha de Abertura
	if ( !_File ) {
		return;
	}

	fprintf( _File, "OceanEngine Log File - Copyright (c) Rodrigo Marques\n" );
	fprintf( _File, "Version: %s\n", LOG_VERSION, 7 );

	//  Marcação de data ( Nome ) e hora de criação
	fprintf( _File, "Date: %s\n", WriteDateTime( LOG_DATE ), _MAX_TIME_ );
	fprintf( _File, "Time: %s\n\n", WriteDateTime( LOG_TIME ), _MAX_TIME_ );
#endif
}

void LogManager::WriteText( const char* Format, ... )
{
#if defined(_DEBUG) || defined(DEBUG)
	if(_File == NULL)
	{
		Initialize("log.txt");
	}

	if ( Format == NULL || _File == NULL )
		return;

	va_list arglist;

	va_start(arglist, Format);

	//  Grava os dados e Marcação de hora
	fprintf( _File, "[%s] : %s", WriteDateTime( LOG_TIME ));

	vfprintf( _File, Format, arglist );

	fprintf( _File, "\n");

	va_end(arglist);
#endif
}

void LogManager::WriteData( const char* FileName, int Line, const char* Function, const char* Format, ... )
{
#if defined(_DEBUG) || defined(DEBUG)
	if(_File == NULL)
	{
		Initialize("log.txt");
	}

	if ( Format == NULL || _File == NULL )
		return;

	va_list arglist;

	va_start(arglist, Format);

	//  Grava os dados e Marcação de hora
	fprintf( _File, "(%s[%d]@%s) [%s] : ", FileName, Line, Function, WriteDateTime( LOG_TIME ));

	vfprintf( _File, Format, arglist );

	fprintf( _File, "\n");

	va_end(arglist);
#endif
}