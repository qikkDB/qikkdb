#pragma once
#include <boost/asio.hpp>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <set>
#include <boost/log/trivial.hpp>
#include "ITCPWorker.h"
#include <atomic>
#include "Configuration.h"
#include "Database.h"
class IClientHandler;

/// <summary>
/// TCP listener and client processor
/// </summary>
/// <param name="worker">Type of worker, must implement ITCPWorker</param>
/// <param name="clientHander">Type of client handler, must implement IClientHandler</param>
template <class ClientHandler, class Worker>
class TCPServer final
{
    static_assert(std::is_base_of<IClientHandler, ClientHandler>::value,
                  "ClientHandler must inherit from IClientHandler");
    static_assert(std::is_base_of<ITCPWorker, Worker>::value, "Worker must inherit from ITCPWorker");

private:
    boost::asio::io_context ioContext_;
    boost::asio::ip::tcp::acceptor acceptor_;
    boost::asio::steady_timer autoSaveDeadline_;
    bool saveDBAutomatically_;
    /// <summary>
    /// Listen for new client requests asynchronously
    /// </summary>
    void Listen()
    {
        acceptor_.async_accept([this](boost::system::error_code ec, boost::asio::ip::tcp::socket socket) {
            if (!ec)
            {
                try
                {
                    BOOST_LOG_TRIVIAL(info)
                        << "Accepting client " << socket.remote_endpoint().address().to_string();

                    std::make_shared<Worker>(std::make_unique<ClientHandler>(), std::move(socket),
                                             Configuration::GetInstance().GetTimeout())
                        ->HandleClient();
                }
                catch (std::exception& e)
                {
                    BOOST_LOG_TRIVIAL(info) << "Exception in worker: " << e.what();
                }
            }
            Listen();
        });
    }

    void AutoSaveDB()
    {
        autoSaveDeadline_.async_wait([this](const boost::system::error_code& error) {
			if (error == boost::asio::error::operation_aborted)
			{
				return;
			}
            // Check whether the deadline has passed. We compare the deadline
            // against the current time since a new asynchronous operation may
            // have moved the deadline before this actor had a chance to run.
            if (autoSaveDeadline_.expiry() <= boost::asio::steady_timer::clock_type::now())
            {
                // The deadline has passed. Save databases.
                BOOST_LOG_TRIVIAL(info) << "Autosaving databases...";
                Database::SaveModifiedToDisk();
            }
            else
            {
                // Put the actor back to sleep.
                AutoSaveDB();
            }
        });
    }

public:
    /// <summary>
    /// Create new instance of TCPServer class.
    /// </summary>
    /// <param name="ipAddress">IPAddress on which to listen</param>
    /// <param name="port">Port on which to listen</param>
    TCPServer(const char* ipAddress, short port, bool saveDBAutomatically = true)
    : ioContext_(),
      acceptor_(ioContext_, boost::asio::ip::tcp::endpoint(boost::asio::ip::make_address(ipAddress), port)),
      saveDBAutomatically_(saveDBAutomatically),
	  autoSaveDeadline_(ioContext_)	{};

    /// <summary>
    /// Starts processing network requests in loop
    /// </summary>
    void Run()
    {
        Listen();
        if (saveDBAutomatically_ && Configuration::GetInstance().GetDBSaveInterval() > 0)
        {
            autoSaveDeadline_.expires_after(
                std::chrono::seconds(Configuration::GetInstance().GetDBSaveInterval()));
            AutoSaveDB();
        }
        ioContext_.run();
    }

    /// <summary>
    /// Stop listening for client requests
    /// </summary>
    void Abort()
    {
		autoSaveDeadline_.cancel();
        acceptor_.cancel();
        ioContext_.stop();
    }

    TCPServer(const TCPServer&) = delete;
    TCPServer& operator=(const TCPServer&) = delete;
};
