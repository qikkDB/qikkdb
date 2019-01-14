#include "NetworkMessage.h"
#include <boost/endian/conversion.hpp>
#include <stdexcept>

void NetworkMessage::WriteToNetwork(const google::protobuf::Message & message, boost::asio::ip::tcp::socket & socket)
{
	google::protobuf::Any packedMsg;
	packedMsg.PackFrom(message);
	int size = packedMsg.ByteSize();
	boost::endian::native_to_little_inplace(size);
	std::vector<char> serializedMessage;
	serializedMessage.reserve(size);
	packedMsg.SerializeToArray(serializedMessage.data(), size);
	boost::asio::write(socket, boost::asio::buffer(&size, sizeof(size)));
	boost::asio::write(socket, boost::asio::buffer(serializedMessage));
}

google::protobuf::Any NetworkMessage::ReadFromNetwork(boost::asio::ip::tcp::socket & socket)
{
	int readSize;
	size_t read = boost::asio::read(socket, boost::asio::buffer(&readSize, sizeof(readSize)));
	boost::endian::little_to_native_inplace(readSize);
	std::vector<char> messageBuffer;
	messageBuffer.reserve(readSize);
	boost::asio::read(socket, boost::asio::buffer(messageBuffer, readSize));
	google::protobuf::Any ret;
	if (!ret.ParseFromArray(messageBuffer.data(), readSize))
	{
		throw std::invalid_argument("Failed to parse message from stream");
	}
	return ret;
}
