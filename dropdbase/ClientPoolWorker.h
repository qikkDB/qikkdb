#pragma once
#include <array>
#include <vector>
#include "ITCPWorker.h"
#include "NetworkMessage.h"

class ClientPoolWorker final : public ITCPWorker
{
private:
    static constexpr int32_t MAXIMUM_BULK_FRAGMENT_SIZE = 8192 * 1024;
    static constexpr size_t NULL_BUFFER_SIZE =
        (MAXIMUM_BULK_FRAGMENT_SIZE + sizeof(char) * 8 - 1) / (sizeof(char) * 8);

    std::unique_ptr<char[]> dataBuffer_;
    std::unique_ptr<char[]> nullBuffer_;

    bool quit_;
    void ClientLoop();
    NetworkMessage networkMessage_;

public:
    /// <summary>
    /// Create new instance of ClientPoolWorker object
    /// </summary>
    /// <param name="activeWorkers">Instance of object responsible for handling messages</param>
    /// <param name="handler">Instance of object responsible for handling messages</param>
    /// <param name="socket">Client that will be handled by this instance</param>
    /// <param name="requestTimeout">Timeout for TCP read and write in ms</param>
    ClientPoolWorker(std::unique_ptr<IClientHandler>&& clientHandler,
                     boost::asio::ip::tcp::socket socket,
                     int requestTimeout);

    // Inherited via ITCPWorker
    virtual void HandleClient() override;

    // Inherited via ITCPWorker
    virtual void Abort() override;

    inline bool HasStopped()
    {
        return quit_;
    }

    ClientPoolWorker(const ClientPoolWorker&) = delete;
    ClientPoolWorker& operator=(const ClientPoolWorker&) = delete;
};
