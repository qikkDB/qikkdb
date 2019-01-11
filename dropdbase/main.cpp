#include "TCPServer.h"
#include "TCPClientHandler.h"
#include "ClientPoolWorker.h"

int main(int argc, char** argv)
{
	TCPServer<TCPClientHandler, ClientPoolWorker> server("127.0.0.1", 12345);
	server.Run();
	return 0;
}