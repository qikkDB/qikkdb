#pragma once
#include "TCPServer.h"
#include "TCPClientHandler.h"
#include "ClientPoolWorker.h"

void RegisterCtrlCHandler(TCPServer<TCPClientHandler, ClientPoolWorker>* server);