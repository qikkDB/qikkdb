#include "NetworkMessage.h"
#include <boost/endian/conversion.hpp>
#include <stdexcept>
#include <boost/asio.hpp>

/// <summary>
/// Write protobuffer message to the network.
/// </summary>
/// <param name="message">Protobuffer message.</param>
/// <param name="socket">Socket to which the message will be written.</param>
void NetworkMessage::WriteToNetwork(const google::protobuf::Message& message,
                                    boost::asio::ip::tcp::socket& socket,
                                    std::function<void()> handler)
{
    google::protobuf::Any packedMsg;
    packedMsg.PackFrom(message);
    int size = packedMsg.ByteSize();
    std::unique_ptr<char[]> serializedMessage = std::make_unique<char[]>(size);
    packedMsg.SerializeToArray(serializedMessage.get(), size);
    boost::endian::native_to_big_inplace(size);
    boost::asio::async_write(
        socket, boost::asio::buffer(&size, sizeof(size)),
        [&socket, size = packedMsg.ByteSize(), serializedMessage{std::move(serializedMessage)},
         handler](const boost::system::error_code& error, std::size_t bytes) {
            if (!error)
            {
                boost::asio::async_write(socket, boost::asio::buffer(serializedMessage.get(), size),
                                         [handler](const boost::system::error_code& error, std::size_t bytes) {
                                             if (!error)
                                             {
                                                 handler();
                                             }
                                         });
            }
        });
}

/// <summary>
/// Read message from network.
/// </summary>
/// <param name="socket">Socket from which the message will be read.</param>
/// <returns>Any protobuffer message.</param>
void NetworkMessage::ReadFromNetwork(boost::asio::ip::tcp::socket& socket,
                                     std::function<void(google::protobuf::Any)> handler)
{
    std::array<char, 4> readBuff;
    boost::asio::async_read(
        socket, boost::asio::buffer(readBuff, 4),
        [&socket, readBuff, handler](const boost::system::error_code& error, std::size_t bytes) {
            if (!error)
            {
                int32_t readSize = *(reinterpret_cast<const int32_t*>(readBuff.data()));
                boost::endian::big_to_native_inplace(readSize);
                std::unique_ptr<char[]> serializedMessage = std::make_unique<char[]>(readSize);
                boost::asio::async_read(socket, boost::asio::buffer(serializedMessage.get(), readSize),
                                        [&socket, serializedMessage{std::move(serializedMessage)}, readSize,
                                         handler](const boost::system::error_code& error, std::size_t bytes) {
                                            if (!error)
                                            {
                                                google::protobuf::Any ret;
                                                if (!ret.ParseFromArray(serializedMessage.get(), readSize))
                                                {
                                                    throw std::invalid_argument(
                                                        "Failed to parse message from stream");
                                                }
                                                handler(ret);
                                            }
                                        });
            }
        });
}

void NetworkMessage::WriteRaw(boost::asio::ip::tcp::socket& socket,
                              char* dataBuffer,
                              int32_t elementCount,
                              DataType dataType,
                              std::function<void()> handler)
{
    int32_t elementSize = GetDataTypeSize(dataType);
    int32_t totalSize = elementCount * elementSize;
    boost::asio::async_write(socket, boost::asio::buffer(dataBuffer, totalSize),
                             [handler](const boost::system::error_code& error, std::size_t bytes) {
                                 if (!error)
                                 {
                                     handler();
                                 }
                             });
}

void NetworkMessage::ReadRaw(boost::asio::ip::tcp::socket& socket,
                             char* dataBuffer,
                             int32_t elementCount,
                             DataType dataType,
                             std::function<void(char*, int32_t)> handler)
{
    int32_t elementSize = GetDataTypeSize(dataType);
    int32_t totalSize = elementCount * elementSize;
    boost::asio::async_read(socket, boost::asio::buffer(dataBuffer, totalSize),
                            [handler, dataBuffer,
                             elementCount](const boost::system::error_code& error, std::size_t bytes) {
                                if (!error)
                                {
                                    handler(dataBuffer, elementCount);
                                }
                            });
}