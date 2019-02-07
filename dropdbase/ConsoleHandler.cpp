#include "ConsoleHandler.h"

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

BOOL WinSigHandler(DWORD dwCtrlId)
{

}

#else
#include <unistd.h>
#include <csignal>
void UnixSigHandler(int signal)
{
	exit(0);
}
#endif

void RegisterCtrlCHandler()
{
#ifdef WIN32
	SetConsoleCtrlHandler(WinSigHandler, TRUE);
#else
	struct sigaction sigIntHandler;

	sigIntHandler.sa_handler = my_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;

	sigaction(SIGINT, &UnixSigHandler, NULL);
#endif
}