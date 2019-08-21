#include "ClientPoolWorker.h"
#include "messages/InfoMessage.pb.h"
#include "messages/QueryMessage.pb.h"
#include "messages/CSVImportMessage.pb.h"
#include "messages/SetDatabaseMessage.pb.h"
#include "messages/BulkImportMessage.pb.h"
#include "IClientHandler.h"
#include <boost/log/trivial.hpp>


/// <summary>
/// Create new instance of ClientPoolWorker object
/// </summary>
/// <param name="activeWorkers">Instance of object responsible for handling messages</param>
/// <param name="handler">Instance of object responsible for handling messages</param>
/// <param name="socket">Client that will be handled by this instance</param>
/// <param name="requestTimeout">Timeout for TCP read and write in ms</param>
ClientPoolWorker::ClientPoolWorker(std::unique_ptr<IClientHandler>&& clientHandler,
                                   boost::asio::ip::tcp::socket socket,
                                   int requestTimeout)
: ITCPWorker(std::move(clientHandler), std::move(socket), requestTimeout),
  dataBuffer_(std::make_unique<char[]>(MAXIMUM_BULK_FRAGMENT_SIZE)),
  nullBuffer_(std::make_unique<char[]>(NULL_BUFFER_SIZE)),
  networkMessage_(), 
#if BOOST_VERSION < 107000
	socketDeadline_{socket.get_executor().context()} 
#else
	socketDeadline_{ socket.get_executor()}
#endif
{
    socketDeadline_.expires_at(boost::asio::steady_timer::time_point::max());
}

/// <summary>
/// Main client request handler
/// </summary>
void ClientPoolWorker::HandleClient()
{
    OnTimeout(socketDeadline_);
    BOOST_LOG_TRIVIAL(debug) << "Waiting for hello from " << socket_.remote_endpoint().address().to_string();
    auto self(shared_from_this());
    socketDeadline_.expires_after(std::chrono::milliseconds(requestTimeout_));
    networkMessage_.ReadFromNetwork(socket_, [this, self](google::protobuf::Any recvMsg) {
        ColmnarDB::NetworkClient::Message::InfoMessage outInfo;
        if (!recvMsg.UnpackTo(&outInfo))
        {
            BOOST_LOG_TRIVIAL(error) << "Invalid message received from client "
                                     << socket_.remote_endpoint().address().to_string();
            return;
        }
        BOOST_LOG_TRIVIAL(debug) << "Hello from " << socket_.remote_endpoint().address().to_string();
        outInfo.set_message("");
        outInfo.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
        networkMessage_.WriteToNetwork(outInfo, socket_, [this, self]() {
            BOOST_LOG_TRIVIAL(debug) << "Hello to " << socket_.remote_endpoint().address().to_string();
            ClientLoop();
        });
    });
}

void ClientPoolWorker::ClientLoop()
{
    auto self(shared_from_this());
    BOOST_LOG_TRIVIAL(debug)
        << "Waiting for message from " << socket_.remote_endpoint().address().to_string();
    socketDeadline_.expires_after(std::chrono::milliseconds(requestTimeout_));
    networkMessage_.ReadFromNetwork(socket_, [this, self](google::protobuf::Any recvMsg) {
        HandleMessage(self, recvMsg);
    });
}

void ClientPoolWorker::HandleMessage(std::shared_ptr<ITCPWorker> self, google::protobuf::Any& recvMsg)
{
    ColmnarDB::NetworkClient::Message::InfoMessage outInfo;
    BOOST_LOG_TRIVIAL(debug) << "Got message from " << socket_.remote_endpoint().address().to_string();
    if (recvMsg.Is<ColmnarDB::NetworkClient::Message::InfoMessage>())
    {
        ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
        recvMsg.UnpackTo(&infoMessage);
        BOOST_LOG_TRIVIAL(debug) << "Info message from " << socket_.remote_endpoint().address().to_string();
        std::unique_ptr<google::protobuf::Message> resultMessage =
            clientHandler_->HandleInfoMessage(*this, infoMessage);

        if (resultMessage != nullptr)
        {
            networkMessage_.WriteToNetwork(*resultMessage, socket_, [this, self]() { ClientLoop(); });
        }
    }
    else if (recvMsg.Is<ColmnarDB::NetworkClient::Message::QueryMessage>())
    {
        ColmnarDB::NetworkClient::Message::QueryMessage queryMessage;
        recvMsg.UnpackTo(&queryMessage);
        BOOST_LOG_TRIVIAL(debug) << "Query message from " << socket_.remote_endpoint().address().to_string();
        std::unique_ptr<google::protobuf::Message> waitMessage =
            clientHandler_->HandleQuery(*this, queryMessage);
        if (waitMessage != nullptr)
        {
            networkMessage_.WriteToNetwork(*waitMessage, socket_, [this, self]() { ClientLoop(); });
        }
    }
    else if (recvMsg.Is<ColmnarDB::NetworkClient::Message::CSVImportMessage>())
    {
        ColmnarDB::NetworkClient::Message::CSVImportMessage csvImportMessage;
        recvMsg.UnpackTo(&csvImportMessage);
        BOOST_LOG_TRIVIAL(debug) << "CSV message from " << socket_.remote_endpoint().address().to_string();
        std::unique_ptr<google::protobuf::Message> importResultMessage =
            clientHandler_->HandleCSVImport(*this, csvImportMessage);
        if (importResultMessage != nullptr)
        {
            networkMessage_.WriteToNetwork(*importResultMessage, socket_,
                                           [this, self]() { ClientLoop(); });
        }
    }
    else if (recvMsg.Is<ColmnarDB::NetworkClient::Message::SetDatabaseMessage>())
    {
        ColmnarDB::NetworkClient::Message::SetDatabaseMessage setDatabaseMessage;
        recvMsg.UnpackTo(&setDatabaseMessage);
        BOOST_LOG_TRIVIAL(debug)
            << "Set database message from " << socket_.remote_endpoint().address().to_string();
        std::unique_ptr<google::protobuf::Message> setDatabaseResult =
            clientHandler_->HandleSetDatabase(*this, setDatabaseMessage);
        if (setDatabaseResult != nullptr)
        {
            networkMessage_.WriteToNetwork(*setDatabaseResult, socket_,
                                           [this, self]() { ClientLoop(); });
        }
    }
    else if (recvMsg.Is<ColmnarDB::NetworkClient::Message::BulkImportMessage>())
    {
        ColmnarDB::NetworkClient::Message::BulkImportMessage bulkImportMessage;
        recvMsg.UnpackTo(&bulkImportMessage);
        BOOST_LOG_TRIVIAL(debug)
            << "BulkImport message from " << socket_.remote_endpoint().address().to_string();
        std::memset(nullBuffer_.get(), 0, NULL_BUFFER_SIZE);
        DataType columnType = static_cast<DataType>(bulkImportMessage.columntype());
        int32_t elementCount = bulkImportMessage.elemcount();
        bool isNullable = bulkImportMessage.isnullable();
        if (elementCount * GetDataTypeSize(columnType) > MAXIMUM_BULK_FRAGMENT_SIZE)
        {
            outInfo.set_message("Data fragment larger than allowed");
            outInfo.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
            networkMessage_.WriteToNetwork(outInfo, socket_, [this, self]() { ClientLoop(); });
            return;
        }
        socketDeadline_.expires_after(std::chrono::milliseconds(requestTimeout_));
        networkMessage_.ReadRaw(
            socket_, dataBuffer_.get(), elementCount, columnType,
            [this, self, isNullable, bulkImportMessage](char* resultBuffer, int32_t elementCount) {
                if (isNullable)
                {
                    size_t nullBufferSize = (elementCount + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
                    socketDeadline_.expires_after(std::chrono::milliseconds(requestTimeout_));
                    networkMessage_.ReadRaw(
                        socket_, nullBuffer_.get(), nullBufferSize, DataType::COLUMN_INT8_T,
                        [this, self, bulkImportMessage](char* resultBuffer, int32_t elementCount) {
                            std::unique_ptr<google::protobuf::Message> importResultMessage =
                                clientHandler_->HandleBulkImport(*this, bulkImportMessage,
                                                                 dataBuffer_.get(), resultBuffer);
                            if (importResultMessage != nullptr)
                            {
                                networkMessage_.WriteToNetwork(*importResultMessage, socket_,
                                                               [this, self]() { ClientLoop(); });
                            }
                        });
                }
                else
                {
                    std::unique_ptr<google::protobuf::Message> importResultMessage =
                        clientHandler_->HandleBulkImport(*this, bulkImportMessage,
                                                         dataBuffer_.get(), nullBuffer_.get());
                    if (importResultMessage != nullptr)
                    {
                        networkMessage_.WriteToNetwork(*importResultMessage, socket_,
                                                       [this, self]() { ClientLoop(); });
                    }
                }
            });
    }
    else
    {
        BOOST_LOG_TRIVIAL(error)
            << "Invalid message from " << socket_.remote_endpoint().address().to_string();
        ClientLoop();
    }
}

/// <summary>
/// Stops the worker after reading next message
/// </summary>
void ClientPoolWorker::Abort()
{
    socket_.close();
    socketDeadline_.cancel();
}

void ClientPoolWorker::OnTimeout(boost::asio::steady_timer& deadline)
{
    auto self(shared_from_this());
    deadline.async_wait([this, self, &deadline](const boost::system::error_code& /*error*/) {
        // Check if the connection was closed while the operation was pending.
        if (HasStopped())
        {
            return;
        }
        // Check whether the deadline has passed. We compare the deadline
        // against the current time since a new asynchronous operation may
        // have moved the deadline before this actor had a chance to run.
        if (deadline.expiry() <= boost::asio::steady_timer::clock_type::now())
        {
            // The deadline has passed. Close the connection.
            Abort();
        }
        else
        {
            // Put the actor back to sleep.
            OnTimeout(deadline);
        }
    });
}