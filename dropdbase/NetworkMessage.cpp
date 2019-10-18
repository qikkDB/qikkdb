#include "NetworkMessage.h"
#include <boost/endian/conversion.hpp>
#include <stdexcept>
#include <boost/asio.hpp>
#include <boost/log/trivial.hpp>

/// <summary>
/// Write protobuffer message to the network.
/// </summary>
/// <param name="message">Protobuffer message.</param>
/// <param name="socket">Socket to which the message will be written.</param>
void NetworkMessage::WriteToNetwork(const google::protobuf::Message& message,
                                    boost::asio::ip::tcp::socket& socket,
                                    std::function<void()> handler,
                                    std::function<void()> abortHandler)
{
    std::string client = socket.remote_endpoint().address().to_string();
    google::protobuf::Any packedMsg;
    packedMsg.PackFrom(message);
    size_ = packedMsg.ByteSize();
    serializedMessage_ = std::unique_ptr<char[]>(new char[size_]);
    packedMsg.SerializeToArray(serializedMessage_.get(), size_);
    boost::endian::native_to_big_inplace(size_);
    boost::asio::async_write(
        socket, boost::asio::buffer(&size_, sizeof(size_)),
        [this, &socket, size = packedMsg.ByteSize(), handler, client,
         abortHandler](const boost::system::error_code& error, std::size_t bytes) {
            if (!error)
            {
                boost::asio::async_write(socket, boost::asio::buffer(serializedMessage_.get(), size),
                                         [handler, client, abortHandler](const boost::system::error_code& error,
                                                                         std::size_t bytes) {
                                             if (!error)
                                             {
                                                 handler();
                                             }
                                             else
                                             {
                                                 std::string message = error.message();
                                                 BOOST_LOG_TRIVIAL(info) << client << " " << message;
                                                 abortHandler();
                                             }
                                         });
            }
            else
            {
                std::string message = error.message();
                BOOST_LOG_TRIVIAL(info) << client << " " << message;
                abortHandler();
            }
        });
}

/// <summary>
/// Read message from network.
/// </summary>
/// <param name="socket">Socket from which the message will be read.</param>
/// <returns>Any protobuffer message.</param>
void NetworkMessage::ReadFromNetwork(boost::asio::ip::tcp::socket& socket,
                                     std::function<void(google::protobuf::Any)> handler,
                                     std::function<void()> abortHandler)
{
    std::string client = socket.remote_endpoint().address().to_string();
    boost::asio::async_read(
        socket, boost::asio::buffer(lengthBuffer_, 4),
        [this, &socket, handler, client, abortHandler](const boost::system::error_code& error, std::size_t bytes) {
            if (!error)
            {
                int32_t readSize = *(reinterpret_cast<const int32_t*>(lengthBuffer_.data()));
                boost::endian::big_to_native_inplace(readSize);
                serializedMessage_ = std::unique_ptr<char[]>(new char[readSize]);
                boost::asio::async_read(socket, boost::asio::buffer(serializedMessage_.get(), readSize),
                                        [this, readSize, handler, client,
                                         abortHandler](const boost::system::error_code& error, std::size_t bytes) {
                                            if (!error)
                                            {
                                                google::protobuf::Any ret;
                                                if (!ret.ParseFromArray(serializedMessage_.get(), readSize))
                                                {
                                                    throw std::invalid_argument(
                                                        "Failed to parse message from stream");
                                                }
                                                handler(ret);
                                            }
                                            else
                                            {
                                                std::string message = error.message();
                                                BOOST_LOG_TRIVIAL(info) << client << " " << message;
                                                abortHandler();
                                            }
                                        });
            }
            else
            {
                std::string message = error.message();
                BOOST_LOG_TRIVIAL(info) << client << " " << message;
                abortHandler();
            }
        });
}

void NetworkMessage::WriteRaw(boost::asio::ip::tcp::socket& socket,
                              char* dataBuffer,
                              int32_t elementCount,
                              DataType dataType,
                              std::function<void()> handler,
                              std::function<void()> abortHandler)
{
    int32_t elementSize = GetDataTypeSize(dataType);
    int32_t totalSize = elementCount * elementSize;
    std::string client = socket.remote_endpoint().address().to_string();
    boost::asio::async_write(socket, boost::asio::buffer(dataBuffer, totalSize),
                             [client, handler, abortHandler](const boost::system::error_code& error,
                                                             std::size_t bytes) {
                                 if (!error)
                                 {
                                     handler();
                                 }
                                 else
                                 {
                                     std::string message = error.message();
                                     BOOST_LOG_TRIVIAL(info) << client << " " << message;
                                     abortHandler();
                                 }
                             });
}

void NetworkMessage::ReadRaw(boost::asio::ip::tcp::socket& socket,
                             char* dataBuffer,
                             int32_t elementCount,
                             int32_t dataSize,
                             std::function<void(char*, int32_t)> handler,
                             std::function<void()> abortHandler)
{
    int32_t totalSize = dataSize;
    std::string client = socket.remote_endpoint().address().to_string();
    boost::asio::async_read(socket, boost::asio::buffer(dataBuffer, totalSize),
                            [handler, dataBuffer, elementCount, client,
                             abortHandler](const boost::system::error_code& error, std::size_t bytes) {
                                if (!error)
                                {
                                    handler(dataBuffer, elementCount);
                                }
                                else
                                {
                                    std::string message = error.message();
                                    BOOST_LOG_TRIVIAL(info) << client << " " << message;
                                    abortHandler();
                                }
                            });
}