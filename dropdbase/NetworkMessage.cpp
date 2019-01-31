#include "NetworkMessage.h"
#include <boost/endian/conversion.hpp>
#include <stdexcept>

void NetworkMessage::WriteToNetwork(const google::protobuf::Message & message, boost::asio::ip::tcp::socket & socket)
{
	google::protobuf::Any packedMsg;
	packedMsg.PackFrom(message);
	int size = packedMsg.ByteSize();
	std::unique_ptr<char[]> serializedMessage = std::make_unique<char[]>(size);
	packedMsg.SerializeToArray(serializedMessage.get(), size);
	boost::endian::native_to_big_inplace(size);
	boost::asio::write(socket, boost::asio::buffer(&size, sizeof(size)));
	boost::asio::write(socket, boost::asio::buffer(serializedMessage.get(), packedMsg.ByteSize()));
}

google::protobuf::Any NetworkMessage::ReadFromNetwork(boost::asio::ip::tcp::socket & socket)
{
	std::array<char, 4> readBuff;
	size_t read = boost::asio::read(socket, boost::asio::buffer(readBuff, 4));
	int32_t readSize = *(reinterpret_cast<int32_t*>(readBuff.data()));
	boost::endian::big_to_native_inplace(readSize);
	std::unique_ptr<char[]> serializedMessage = std::make_unique<char[]>(readSize);
	boost::asio::read(socket, boost::asio::buffer(serializedMessage.get(), readSize));
	google::protobuf::Any ret;
	if (!ret.ParseFromArray(serializedMessage.get(), readSize))
	{
		throw std::invalid_argument("Failed to parse message from stream");
	}
	return ret;
}
