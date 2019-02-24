#include "ConsoleHandler.h"


static TCPServer<TCPClientHandler, ClientPoolWorker>* currentServer = nullptr;

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

BOOL WinSigHandler(DWORD dwCtrlId)
{
	if (currentServer != nullptr && dwCtrlId == CTRL_C_EVENT)
	{
		currentServer->Abort();
		currentServer = nullptr;
		return TRUE;
	}
	return FALSE;
}

#else
#include <unistd.h>
#include <csignal>
void UnixSigHandler(int signal)
{
	if (currentServer != nullptr)
	{
		currentServer->Abort();
		currentServer = nullptr;
	}
	else
	{
		abort();
	}
}
#endif

void RegisterCtrlCHandler(TCPServer<TCPClientHandler, ClientPoolWorker>* server)
{
	currentServer = server;
#ifdef WIN32
	SetConsoleCtrlHandler(WinSigHandler, TRUE);
#else
	struct sigaction sigIntHandler;

	sigIntHandler.sa_handler = UnixSigHandler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;

	sigaction(SIGINT, &sigIntHandler, NULL);
#endif
}