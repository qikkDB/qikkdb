#pragma once
#include <google/protobuf/message.h>
#include <google/protobuf/any.pb.h>
#include <boost/asio.hpp>
class NetworkMessage final
{
private:
	NetworkMessage() {};
public:
	static void WriteToNetwork(const google::protobuf::Message& message, boost::asio::ip::tcp::socket& socket);
	static google::protobuf::Any ReadFromNetwork(boost::asio::ip::tcp::socket& socket);
};

