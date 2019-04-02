#pragma once
#include "TCPServer.h"
#include "TCPClientHandler.h"
#include "ClientPoolWorker.h"

/// <summary>
/// Register Handler to Ctrl+C sequence in system.
/// </summary>
void RegisterCtrlCHandler(TCPServer<TCPClientHandler, ClientPoolWorker>* server);