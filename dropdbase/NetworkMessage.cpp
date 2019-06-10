#include "NetworkMessage.h"
#include <boost/endian/conversion.hpp>
#include <stdexcept>

/// <summary>
/// Write protobuffer message to the network.
/// </summary>
/// <param name="message">Protobuffer message.</param>
/// <param name="socket">Socket to which the message will be written.</param>
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

/// <summary>
/// Read message from network.
/// </summary>
/// <param name="socket">Socket from which the message will be read.</param>
/// <returns>Any protobuffer message.</param>
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

void NetworkMessage::WriteRaw(boost::asio::ip::tcp::socket& socket, char* dataBuffer, int32_t elementCount, DataType dataType)
{
	int32_t elementSize = 0;
	switch(dataType)
	{
		case COLUMN_INT:
		case CONST_INT:
			elementSize = sizeof(int32_t);
			break;
		case COLUMN_LONG:
		case CONST_LONG:
			elementSize = sizeof(int64_t);
			break;
		case COLUMN_DOUBLE:
		case CONST_DOUBLE:
			elementSize = sizeof(double);
			break;
		case COLUMN_FLOAT:
		case CONST_FLOAT:
			elementSize = sizeof(float);
			break;
		default:
			elementSize = sizeof(int8_t);
			break;
	}

	int32_t totalSize = elementCount * elementSize;
	boost::asio::write(socket, boost::asio::buffer(dataBuffer, totalSize));
}

void NetworkMessage::ReadRaw(boost::asio::ip::tcp::socket& socket, char* dataBuffer, int32_t elementCount, DataType dataType)
{
	int32_t elementSize = 0;
	switch(dataType)
	{
		case COLUMN_INT:
		case CONST_INT:
			elementSize = sizeof(int32_t);
			break;
		case COLUMN_LONG:
		case CONST_LONG:
			elementSize = sizeof(int64_t);
			break;
		case COLUMN_DOUBLE:
		case CONST_DOUBLE:
			elementSize = sizeof(double);
			break;
		case COLUMN_FLOAT:
		case CONST_FLOAT:
			elementSize = sizeof(float);
			break;
		default:
			elementSize = sizeof(int8_t);
			break;
	}

	int32_t totalSize = elementCount * elementSize;
	boost::asio::read(socket, boost::asio::buffer(dataBuffer, totalSize));

}