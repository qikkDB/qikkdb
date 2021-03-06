// qikkDB_test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "gtest/gtest.h"
#include "../qikkDB/messages/InfoMessage.pb.h"
#include <google/protobuf/message.h>
#include <google/protobuf/any.pb.h>

TEST(ProtocolMessageTests, ProtobufSerializeAndDeserialize)
{
	QikkDB::NetworkClient::Message::InfoMessage testMsg;
	testMsg.set_code(QikkDB::NetworkClient::Message::InfoMessage::OK);
	testMsg.set_message("Serialization test");
	google::protobuf::Any packedMsg;
	packedMsg.PackFrom(testMsg);
	int size = packedMsg.ByteSize();
	std::vector<char> serializedMessage;
	serializedMessage.reserve(size);
	packedMsg.SerializeToArray(serializedMessage.data(), size);
	google::protobuf::Any ret;
	if (!ret.ParseFromArray(serializedMessage.data(), size))
	{
		throw std::invalid_argument("Failed to parse message from stream");
	}
	QikkDB::NetworkClient::Message::InfoMessage infoMessage;
	ret.UnpackTo(&infoMessage);
	ASSERT_EQ(infoMessage.code(), QikkDB::NetworkClient::Message::InfoMessage::OK);
	ASSERT_EQ(infoMessage.message(), "Serialization test");
}