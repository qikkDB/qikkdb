#pragma once
#include <memory>
#include <boost/asio/ip/tcp.hpp>
class IClientHandler;
class Database;

/// <summary>
/// Receiving and dispatching for network requests
/// </summary>
class ITCPWorker : public std::enable_shared_from_this<ITCPWorker>
{
protected:
    /// <summary>
    /// Timeout for TCP read and write in ms
    /// </summary>
    int requestTimeout_;
    /// <summary>
    /// Client handled by this instance
    /// </summary>
    boost::asio::ip::tcp::socket socket_;
    /// <summary>
    /// Instance of object responsible for handling messages
    /// </summary>
    std::unique_ptr<IClientHandler> clientHandler_;


public:
    /// <summary>
    /// Create new instance of ITCPWorker object
    /// </summary>
    /// <param name="clientHandler">Instance of object responsible for handling messages</param>
    /// <param name="socket">Client that will be handled by this instance</param>
    /// <param name="requestTimeout">Timeout for TCP read and write in ms</param>
    ITCPWorker(std::unique_ptr<IClientHandler>&& clientHandler, boost::asio::ip::tcp::socket socket, int requestTimeout);
    virtual ~ITCPWorker();
    virtual void HandleClient() = 0;
    virtual void Abort() = 0;
    ITCPWorker(const ITCPWorker&) = delete;
    ITCPWorker& operator=(const ITCPWorker&) = delete;

    /// <summary>
    /// Current working database
    /// </summary>
    std::weak_ptr<Database> currentDatabase_;
};
