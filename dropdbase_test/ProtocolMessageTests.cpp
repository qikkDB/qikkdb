// dropdbase_test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "gtest/gtest.h"
#include "../dropdbase/messages/InfoMessage.pb.h"
#include <google/protobuf/message.h>
#include <google/protobuf/any.pb.h>

TEST(dropdbaseServer, protobufSerializeAndDeserialize)
{
	ColmnarDB::NetworkClient::Message::InfoMessage testMsg;
	testMsg.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
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
	ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
	ret.UnpackTo(&infoMessage);
	ASSERT_EQ(infoMessage.code(), ColmnarDB::NetworkClient::Message::InfoMessage::OK);
	ASSERT_EQ(infoMessage.message(), "Serialization test");
}