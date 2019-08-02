#include "ConsoleHandler.h"
#include "TCPServer.h"
#include "TCPClientHandler.h"
#include "ClientPoolWorker.h"

static TCPServer<TCPClientHandler, ClientPoolWorker>* currentServer = nullptr;

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

/// <summary>
/// For Windows. A console process uses this function to handle control signals received by the process. When the signal is received,
/// the system creates a new thread in the process to execute the function.
/// </summary>
/// <returns">If the function handles the control signal, it should return TRUE. If it returns FALSE,
/// the next handler function in the list of handlers for this process is used.</returns>
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
/// <summary>
/// For Unix. A console process uses this function to handle control signals received by the process. When the signal is received,
/// the system creates a new thread in the process to execute the function.
/// </summary>
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

/// <summary>
/// Register Handler to Ctrl+C sequence in system.
/// </summary>
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