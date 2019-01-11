#include "NetworkMessage.h"
#include <boost/endian/conversion.hpp>
#include <stdexcept>

void NetworkMessage::WriteToNetwork(google::protobuf::Message & message, boost::asio::ip::tcp::socket & socket)
{
	google::protobuf::Any packedMsg;
	packedMsg.PackFrom(message);
	int size = packedMsg.ByteSize();
	boost::endian::native_to_little_inplace(size);
	std::vector<char> serializedMessage;
	serializedMessage.reserve(size);
	packedMsg.SerializeToArray(serializedMessage.data(), size);
	boost::asio::write(socket, boost::asio::buffer(serializedMessage));
}

google::protobuf::Any NetworkMessage::ReadFromNetwork(boost::asio::ip::tcp::socket & socket)
{
	std::array<char, 4> buffer;
	size_t read = boost::asio::read(socket, boost::asio::buffer(buffer, 4));
	int32_t readSize = *(reinterpret_cast<int32_t*>(&buffer[0]));
	boost::endian::little_to_native_inplace(readSize);
	std::vector<char> messageBuffer;
	messageBuffer.reserve(readSize);
	boost::asio::read(socket, boost::asio::buffer(buffer, readSize));
	google::protobuf::Any ret;
	if (!ret.ParseFromArray(buffer.data(), readSize))
	{
		throw std::invalid_argument("Failed to parse message from stream");
	}
	return ret;
}
