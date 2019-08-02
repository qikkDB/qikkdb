#pragma once
class TCPClientHandler;
class ClientPoolWorker;

template <class ClientHandler, class Worker>
class TCPServer;

/// <summary>
/// Register Handler to Ctrl+C sequence in system.
/// </summary>
void RegisterCtrlCHandler(TCPServer<TCPClientHandler, ClientPoolWorker>* server);