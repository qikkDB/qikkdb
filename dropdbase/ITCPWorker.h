#pragma once
#include <memory>
#include <set>
#include <boost/asio.hpp>
#include "Database.h"

class IClientHandler;

class ITCPWorker : std::enable_shared_from_this<ITCPWorker>
{
protected:
	int requestTimeout_;
	boost::asio::ip::tcp::socket socket_;
	std::unique_ptr<IClientHandler> clientHandler_;
	std::set<std::shared_ptr<ITCPWorker>>& activeWorkers_;
	std::shared_ptr<ITCPWorker> GetSharedFromThis() { return shared_from_this(); }
public:
	ITCPWorker(std::set<std::shared_ptr<ITCPWorker>>& activeWorkers, std::unique_ptr<IClientHandler>&& clientHandler, boost::asio::ip::tcp::socket socket, int requestTimeout);
	virtual ~ITCPWorker();
	virtual void HandleClient() = 0;
	virtual void Abort();
	ITCPWorker(const ITCPWorker&) = delete;
	ITCPWorker& operator=(const ITCPWorker&) = delete;
	std::shared_ptr<Database> currentDatabase_;

};

