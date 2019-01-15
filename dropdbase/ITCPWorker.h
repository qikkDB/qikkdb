#pragma once
#include <memory>
#include <set>
#include <boost/asio.hpp>
#include "Database.h"

class IClientHandler;

class ITCPWorker
{
protected:
	int requestTimeout_;
	boost::asio::ip::tcp::socket socket_;
	std::unique_ptr<IClientHandler> clientHandler_;
public:
	ITCPWorker(std::unique_ptr<IClientHandler>&& clientHandler, boost::asio::ip::tcp::socket socket, int requestTimeout);
	virtual ~ITCPWorker();
	virtual void HandleClient() = 0;
	virtual void Abort() = 0;
	ITCPWorker(const ITCPWorker&) = delete;
	ITCPWorker& operator=(const ITCPWorker&) = delete;
	std::shared_ptr<Database> currentDatabase_;

};

