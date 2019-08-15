#pragma once
#include <google/protobuf/message.h>
#include <google/protobuf/any.pb.h>
#include <boost/asio/ip/tcp.hpp>
#include <array>
#include "DataType.h"

class NetworkMessage final
{
private:
    std::array<char, 4> lengthBuffer_;
    std::unique_ptr<char[]> serializedMessage_;
    int size_;

public:
    /// <summary>
    /// Write protobuffer message to the network.
    /// </summary>
    /// <param name="message">Protobuffer message.</param>
    /// <param name="socket">Socket to which the message will be written.</param>
    void WriteToNetwork(const google::protobuf::Message& message,
                        boost::asio::ip::tcp::socket& socket,
                        std::function<void()> handler);

    /// <summary>
    /// Read message from network.
    /// </summary>
    /// <param name="socket">Socket from which the message will be read.</param>
    /// <returns>Any protobuffer message.</param>
    void ReadFromNetwork(boost::asio::ip::tcp::socket& socket, std::function<void(google::protobuf::Any)> handler);

    void ReadRaw(boost::asio::ip::tcp::socket& socket,
                 char* dataBuffer,
                 int32_t elementCount,
                 DataType dataType,
                 std::function<void(char*, int32_t)> handler);
    void WriteRaw(boost::asio::ip::tcp::socket& socket,
                  char* dataBuffer,
                  int32_t elementCount,
                  DataType dataType,
                  std::function<void()> handler);
};
