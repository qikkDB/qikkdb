#include "ITCPWorker.h"
#include "IClientHandler.h"


ITCPWorker::ITCPWorker(std::unique_ptr<IClientHandler>&& clientHandler, boost::asio::ip::tcp::socket socket, int requestTimeout)
: requestTimeout_(requestTimeout), socket_(std::move(socket)), clientHandler_(std::move(clientHandler))

{
}


ITCPWorker::~ITCPWorker()
{
}
