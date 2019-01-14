#pragma once
#include <boost/asio.hpp>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <set>
#include "IClientHandler.h"
#include "ITCPWorker.h"

template <class ClientHandler, class Worker>
class TCPServer final
{
	static_assert(std::is_base_of<IClientHandler, ClientHandler>::value, "ClientHandler must inherit from IClientHandler");
	static_assert(std::is_base_of<ITCPWorker, Worker>::value, "Worker must inherit from ITCPWorker");
private:
	boost::asio::io_context ioContext_;
	boost::asio::ip::tcp::acceptor acceptor_;
	std::set<std::shared_ptr<ITCPWorker>> activeWorkers_;
	void Listen()
	{
		acceptor_.async_accept([this](boost::system::error_code ec, boost::asio::ip::tcp::socket socket)
		{
			if (!ec)
			{
				std::thread handlerThread([this]() { std::make_shared<Worker>(activeWorkers_, std::unique_ptr<IClientHandler>(dynamic_cast<IClientHandler*>(new ClientHandler())), std::move(socket), 60000)->HandleClient(); });
				handlerThread.detach();
			}
			Listen();
		});
	}

public:
	TCPServer(const std::string& ipAddress, short port)
		: ioContext_(), acceptor_(ioContext_, boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address(ipAddress), port)) 
	{
	};
	
	void Run() 
	{
		Listen();
		ioContext_.run();
	}


	void Abort()
	{
		acceptor_.cancel();
		ioContext_.stop();
	}

	TCPServer(const TCPServer&) = delete;
	TCPServer& operator=(const TCPServer&) = delete;
};

