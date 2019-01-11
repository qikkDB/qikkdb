#include "ITCPWorker.h"
#include "IClientHandler.h"


ITCPWorker::ITCPWorker(std::set<std::shared_ptr<ITCPWorker>>& activeWorkers, std::unique_ptr<IClientHandler> clientHandler, boost::asio::ip::tcp::socket socket, int requestTimeout) 
	: activeWorkers_(activeWorkers),clientHandler_(std::move(clientHandler)),socket_(std::move(socket)),requestTimeout_(requestTimeout)

{
	activeWorkers_.insert(shared_from_this());
}


ITCPWorker::~ITCPWorker()
{
}

void ITCPWorker::Abort()
{
	activeWorkers_.erase(shared_from_this());
}
