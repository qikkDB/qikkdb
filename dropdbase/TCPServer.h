#pragma once
#include <boost/asio.hpp>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <set>
#include <boost/log/trivial.hpp>
#include "IClientHandler.h"
#include "ITCPWorker.h"

/// <summary>
/// TCP listener and client processor
/// </summary>
/// <param name="worker">Type of worker, must implement ITCPWorker</param>
/// <param name="clientHander">Type of client handler, must implement IClientHandler</param>
template <class ClientHandler, class Worker>
class TCPServer final
{
	static_assert(std::is_base_of<IClientHandler, ClientHandler>::value, "ClientHandler must inherit from IClientHandler");
	static_assert(std::is_base_of<ITCPWorker, Worker>::value, "Worker must inherit from ITCPWorker");
private:
	boost::asio::io_context ioContext_;
	boost::asio::ip::tcp::acceptor acceptor_;
	
	/// <summary>
	/// Listen for new client requests asynchronously
	/// </summary>
	void Listen()
	{
		acceptor_.async_accept([this](boost::system::error_code ec, boost::asio::ip::tcp::socket socket)
		{
			if (!ec)
			{
				std::thread handlerThread([this](boost::asio::ip::tcp::socket sock)
				{
					try
					{
						BOOST_LOG_TRIVIAL(info) << "Accepting client " << sock.remote_endpoint().address().to_string() << "\n";
						Worker worker(std::make_unique<ClientHandler>(), std::move(sock), 60000);
						worker.HandleClient();
					}
					catch (std::exception& e)
					{
						printf("Exception in worker: %s", e.what());
					}
				}, std::move(socket));
				handlerThread.detach();
			}
			Listen();
		});
	}

public:
	/// <summary>
	/// Create new instance of TCPServer class.
	/// </summary>
	/// <param name="ipAddress">IPAddress on which to listen</param>
	/// <param name="port">Port on which to listen</param>
	TCPServer(const char* ipAddress, short port)
		: ioContext_(), acceptor_(ioContext_, boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address(ipAddress), port)) 
	{
	};

	/// <summary>
	/// Starts processing network requests in loop
	/// </summary>
	void Run() 
	{
		Listen();
		ioContext_.run();
	}

	/// <summary>
	/// Stop listening for client requests
	/// </summary>
	void Abort()
	{
		acceptor_.cancel();
		ioContext_.stop();
	}

	TCPServer(const TCPServer&) = delete;
	TCPServer& operator=(const TCPServer&) = delete;
};

