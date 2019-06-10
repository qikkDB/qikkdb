#include "gtest/gtest.h"
#include "../dropdbase/TCPClientHandler.h"
#include "../dropdbase/ClientPoolWorker.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "../dropdbase/Types/Point.pb.h"

TEST(ClientHandlerTests, TestHandlerInfo)
{
	std::unique_ptr<IClientHandler> handler = std::make_unique<TCPClientHandler>();
	boost::asio::io_context context;
	IClientHandler* handlerPtr = handler.get();
	ClientPoolWorker tempWorker(std::move(handler), boost::asio::ip::tcp::socket(context), 60000);
	ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
	infoMessage.set_code(ColmnarDB::NetworkClient::Message::InfoMessage_StatusCode_CONN_END);
	infoMessage.set_message("");
	handlerPtr->HandleInfoMessage(tempWorker, infoMessage);
	ASSERT_TRUE(tempWorker.HasStopped());
}

TEST(ClientHandlerTests, TestHandlerSetDB)
{
	std::shared_ptr<Database> db = std::make_shared<Database>("test");
	Database::AddToInMemoryDatabaseList(db);
	std::unique_ptr<IClientHandler> handler = std::make_unique<TCPClientHandler>();
	boost::asio::io_context context;
	IClientHandler* handlerPtr = handler.get();
	ClientPoolWorker tempWorker(std::move(handler), boost::asio::ip::tcp::socket(context), 60000);
	ColmnarDB::NetworkClient::Message::SetDatabaseMessage setDatabaseMessage;
	setDatabaseMessage.set_databasename("test");
	handlerPtr->HandleSetDatabase(tempWorker,setDatabaseMessage);
	auto dbPtr = tempWorker.currentDatabase_.lock();
	ASSERT_EQ(std::string("test"),dbPtr->GetName());
	Database::RemoveFromInMemoryDatabaseList("test");
}

TEST(ClientHandlerTests, TestHandlerCSV)
{
	std::unique_ptr<IClientHandler> handler = std::make_unique<TCPClientHandler>();
	boost::asio::io_context context;
	IClientHandler* handlerPtr = handler.get();
	ClientPoolWorker tempWorker(std::move(handler), boost::asio::ip::tcp::socket(context), 60000);
	ColmnarDB::NetworkClient::Message::CSVImportMessage csvMessage;
	csvMessage.set_csvname("test");
	csvMessage.set_databasename("testCSV");
	csvMessage.set_payload("test1,test2,test3\n1,2,3");
	handlerPtr->HandleCSVImport(tempWorker, csvMessage);
	auto db = Database::GetDatabaseByName("testCSV");
	ASSERT_NE(db.get(), static_cast<Database*>(nullptr));
	auto& column1 = db->GetTables().at("test").GetColumns().at("test1");
	ASSERT_EQ(1, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column1).GetBlocksList()[0]).GetData()[0]);
	auto& column2 = db->GetTables().at("test").GetColumns().at("test2");
	ASSERT_EQ(2, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column2).GetBlocksList()[0]).GetData()[0]);
	auto& column3 = db->GetTables().at("test").GetColumns().at("test3");
	ASSERT_EQ(3, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column3).GetBlocksList()[0]).GetData()[0]);
	Database::RemoveFromInMemoryDatabaseList("testCSV");
}

TEST(ClientHandlerTests, TestHandlerCSVWithTypes)
{
	std::unique_ptr<IClientHandler> handler = std::make_unique<TCPClientHandler>();
	boost::asio::io_context context;
	IClientHandler* handlerPtr = handler.get();
	ClientPoolWorker tempWorker(std::move(handler), boost::asio::ip::tcp::socket(context), 60000);
	ColmnarDB::NetworkClient::Message::CSVImportMessage csvMessage;
	csvMessage.set_csvname("test");
	csvMessage.set_databasename("testCSV");
	csvMessage.set_payload("test1,test2,test3\n1,2,3");
	csvMessage.add_columntypes(ColmnarDB::NetworkClient::Message::COLUMN_INT);
	csvMessage.add_columntypes(ColmnarDB::NetworkClient::Message::COLUMN_INT);
	csvMessage.add_columntypes(ColmnarDB::NetworkClient::Message::COLUMN_INT);
	handlerPtr->HandleCSVImport(tempWorker, csvMessage);
	auto db = Database::GetDatabaseByName("testCSV");
	ASSERT_NE(db.get(), static_cast<Database*>(nullptr));
	auto& column1 = db->GetTables().at("test").GetColumns().at("test1");
	ASSERT_EQ(1, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column1).GetBlocksList()[0]).GetData()[0]);
	auto& column2 = db->GetTables().at("test").GetColumns().at("test2");
	ASSERT_EQ(2, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column2).GetBlocksList()[0]).GetData()[0]);
	auto& column3 = db->GetTables().at("test").GetColumns().at("test3");
	ASSERT_EQ(3, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column3).GetBlocksList()[0]).GetData()[0]);
	Database::RemoveFromInMemoryDatabaseList("testCSV");
}

TEST(ClientHandlerTests, TestHandlerQuery)
{
	Context::getInstance();
	std::shared_ptr<Database> db = std::make_shared<Database>("test");
	Database::AddToInMemoryDatabaseList(db);
	db->CreateTable(std::unordered_map<std::string, DataType>{ {"test", COLUMN_INT} }, "test");
	db->GetTables().at("test").InsertData(std::unordered_map <std::string, std::any>{ {"test", std::make_any<std::vector<int>>({ 1,2,3 })}});
	std::unique_ptr<IClientHandler> handler = std::make_unique<TCPClientHandler>();
	boost::asio::io_context context;
	IClientHandler* handlerPtr = handler.get();
	ClientPoolWorker tempWorker(std::move(handler), boost::asio::ip::tcp::socket(context), 60000);
	tempWorker.currentDatabase_ = db;
	ColmnarDB::NetworkClient::Message::QueryMessage queryMessage;
	queryMessage.set_query("SELECT test FROM test;");
	auto responsePtr = handlerPtr->HandleQuery(tempWorker, queryMessage);
	ASSERT_NE(dynamic_cast<ColmnarDB::NetworkClient::Message::InfoMessage*>(responsePtr.get()), static_cast<ColmnarDB::NetworkClient::Message::InfoMessage*>(nullptr));
	ASSERT_EQ(dynamic_cast<ColmnarDB::NetworkClient::Message::InfoMessage*>(responsePtr.get())->code(), ColmnarDB::NetworkClient::Message::InfoMessage_StatusCode_WAIT);
	ColmnarDB::NetworkClient::Message::InfoMessage getNextResultMessage;
	getNextResultMessage.set_code(ColmnarDB::NetworkClient::Message::InfoMessage_StatusCode_GET_NEXT_RESULT);
	getNextResultMessage.set_message("");
	auto queryResponsePtr = handlerPtr->HandleInfoMessage(tempWorker, getNextResultMessage);
	ASSERT_NE(dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(queryResponsePtr.get()), static_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(nullptr));
	auto& payload = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(queryResponsePtr.get())->payloads().at("test.test").intpayload().intdata();
	ASSERT_EQ(payload.size(), 3);
	ASSERT_EQ(payload[0], 1);
	ASSERT_EQ(payload[1], 2);
	ASSERT_EQ(payload[2], 3);
	Database::RemoveFromInMemoryDatabaseList("test");
}

TEST(ClientHandlerTests, TestHandlerBulkImportSingleColumn)
{
	std::shared_ptr<Database> db = std::make_shared<Database>("test");
	Database::AddToInMemoryDatabaseList(db);
	std::unique_ptr<IClientHandler> handler = std::make_unique<TCPClientHandler>();
	boost::asio::io_context context;
	IClientHandler* handlerPtr = handler.get();
	ClientPoolWorker tempWorker(std::move(handler), boost::asio::ip::tcp::socket(context), 60000);
	ColmnarDB::NetworkClient::Message::SetDatabaseMessage setDatabaseMessage;
	setDatabaseMessage.set_databasename("test");
	handlerPtr->HandleSetDatabase(tempWorker,setDatabaseMessage);
	ColmnarDB::NetworkClient::Message::BulkImportMessage bulkImportMessage;
	bulkImportMessage.set_tablename("test");
	bulkImportMessage.set_columnname("test");
	bulkImportMessage.set_columntype(static_cast<ColmnarDB::NetworkClient::Message::DataType>(DataType::COLUMN_INT));
	bulkImportMessage.set_elemcount(5);
	int32_t dataBuff[] = {1,2,3,4,5};
	handlerPtr->HandleBulkImport(tempWorker,bulkImportMessage,reinterpret_cast<char*>(dataBuff));
	auto& column1 = db->GetTables().at("test").GetColumns().at("test");
	ASSERT_EQ(1, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column1).GetBlocksList()[0]).GetData()[0]);
	ASSERT_EQ(5, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column1).GetBlocksList()[0]).GetData()[4]);
	Database::RemoveFromInMemoryDatabaseList("test");
}


TEST(ClientHandlerTests, TestHandlerBulkImportMultiColumn)
{
	std::shared_ptr<Database> db = std::make_shared<Database>("test");
	Database::AddToInMemoryDatabaseList(db);
	std::unique_ptr<IClientHandler> handler = std::make_unique<TCPClientHandler>();
	boost::asio::io_context context;
	IClientHandler* handlerPtr = handler.get();
	ClientPoolWorker tempWorker(std::move(handler), boost::asio::ip::tcp::socket(context), 60000);
	ColmnarDB::NetworkClient::Message::SetDatabaseMessage setDatabaseMessage;
	setDatabaseMessage.set_databasename("test");
	handlerPtr->HandleSetDatabase(tempWorker,setDatabaseMessage);
	ColmnarDB::NetworkClient::Message::BulkImportMessage bulkImportMessage;
	bulkImportMessage.set_tablename("test");
	bulkImportMessage.set_columnname("test");
	bulkImportMessage.set_columntype(static_cast<ColmnarDB::NetworkClient::Message::DataType>(DataType::COLUMN_INT));
	bulkImportMessage.set_elemcount(5);
	int32_t dataBuff[] = {1,2,3,4,5};
	handlerPtr->HandleBulkImport(tempWorker,bulkImportMessage,reinterpret_cast<char*>(dataBuff));
	bulkImportMessage.set_tablename("test");
	bulkImportMessage.set_columnname("test2");
	bulkImportMessage.set_columntype(static_cast<ColmnarDB::NetworkClient::Message::DataType>(DataType::COLUMN_INT));
	bulkImportMessage.set_elemcount(5);
	int32_t dataBuff2[] = {6,7,8,9,10};
	handlerPtr->HandleBulkImport(tempWorker,bulkImportMessage,reinterpret_cast<char*>(dataBuff2));
	auto& column1 = db->GetTables().at("test").GetColumns().at("test");
	auto& column2 = db->GetTables().at("test").GetColumns().at("test2");
	ASSERT_EQ(1, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column1).GetBlocksList()[0]).GetData()[0]);
	ASSERT_EQ(5, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column1).GetBlocksList()[0]).GetData()[4]);
	ASSERT_EQ(6, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column2).GetBlocksList()[0]).GetData()[0]);
	ASSERT_EQ(10, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column2).GetBlocksList()[0]).GetData()[4]);
	Database::RemoveFromInMemoryDatabaseList("test");
}

TEST(ClientHandlerTests, TestHandlerBulkImportMultiSplitColumn)
{
	std::shared_ptr<Database> db = std::make_shared<Database>("test");
	Database::AddToInMemoryDatabaseList(db);
	std::unique_ptr<IClientHandler> handler = std::make_unique<TCPClientHandler>();
	boost::asio::io_context context;
	IClientHandler* handlerPtr = handler.get();
	ClientPoolWorker tempWorker(std::move(handler), boost::asio::ip::tcp::socket(context), 60000);
	ColmnarDB::NetworkClient::Message::SetDatabaseMessage setDatabaseMessage;
	setDatabaseMessage.set_databasename("test");
	handlerPtr->HandleSetDatabase(tempWorker,setDatabaseMessage);
	ColmnarDB::NetworkClient::Message::BulkImportMessage bulkImportMessage;
	bulkImportMessage.set_tablename("test");
	bulkImportMessage.set_columnname("test");
	bulkImportMessage.set_columntype(static_cast<ColmnarDB::NetworkClient::Message::DataType>(DataType::COLUMN_INT));
	bulkImportMessage.set_elemcount(5);
	int32_t dataBuff[] = {1,2,3,4,5};
	handlerPtr->HandleBulkImport(tempWorker,bulkImportMessage,reinterpret_cast<char*>(dataBuff));
	ColmnarDB::NetworkClient::Message::BulkImportMessage bulkImportMessage2;
	bulkImportMessage2.set_tablename("test");
	bulkImportMessage2.set_columnname("test2");
	bulkImportMessage2.set_columntype(static_cast<ColmnarDB::NetworkClient::Message::DataType>(DataType::COLUMN_INT));
	bulkImportMessage2.set_elemcount(5);
	int32_t dataBuff2[] = {6,7,8,9,10};
	handlerPtr->HandleBulkImport(tempWorker,bulkImportMessage2,reinterpret_cast<char*>(dataBuff2));
	handlerPtr->HandleBulkImport(tempWorker,bulkImportMessage,reinterpret_cast<char*>(dataBuff2));
	handlerPtr->HandleBulkImport(tempWorker,bulkImportMessage2,reinterpret_cast<char*>(dataBuff));
	auto& column1 = db->GetTables().at("test").GetColumns().at("test");
	auto& column2 = db->GetTables().at("test").GetColumns().at("test2");
	ASSERT_EQ(1, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column1).GetBlocksList()[0]).GetData()[0]);
	ASSERT_EQ(5, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column1).GetBlocksList()[0]).GetData()[4]);
	ASSERT_EQ(6, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column1).GetBlocksList()[0]).GetData()[5]);
	ASSERT_EQ(10, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column1).GetBlocksList()[0]).GetData()[9]);
	ASSERT_EQ(6, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column2).GetBlocksList()[0]).GetData()[0]);
	ASSERT_EQ(10, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column2).GetBlocksList()[0]).GetData()[4]);
	ASSERT_EQ(1, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column2).GetBlocksList()[0]).GetData()[5]);
	ASSERT_EQ(5, dynamic_cast<BlockBase<int>&>(*dynamic_cast<ColumnBase<int>&>(*column2).GetBlocksList()[0]).GetData()[9]);
	Database::RemoveFromInMemoryDatabaseList("test");
}

TEST(ClientHandlerTests, TestHandlerBulkImportProtobufColumn)
{
	std::shared_ptr<Database> db = std::make_shared<Database>("test");
	Database::AddToInMemoryDatabaseList(db);
	std::unique_ptr<IClientHandler> handler = std::make_unique<TCPClientHandler>();
	boost::asio::io_context context;
	IClientHandler* handlerPtr = handler.get();
	ClientPoolWorker tempWorker(std::move(handler), boost::asio::ip::tcp::socket(context), 60000);
	ColmnarDB::NetworkClient::Message::SetDatabaseMessage setDatabaseMessage;
	setDatabaseMessage.set_databasename("test");
	handlerPtr->HandleSetDatabase(tempWorker,setDatabaseMessage);
	ColmnarDB::NetworkClient::Message::BulkImportMessage bulkImportMessage;
	bulkImportMessage.set_tablename("test");
	bulkImportMessage.set_columnname("test");
	bulkImportMessage.set_columntype(static_cast<ColmnarDB::NetworkClient::Message::DataType>(DataType::COLUMN_POINT));
	bulkImportMessage.set_elemcount(5);
	ColmnarDB::Types::Point aPoint;
	aPoint.mutable_geopoint()->set_latitude(5);
	aPoint.mutable_geopoint()->set_longitude(5);
	int32_t arraySize = aPoint.ByteSize() * 5 + sizeof(int32_t) * 5;
	std::unique_ptr<char[]> dataBuff = std::make_unique<char[]>(arraySize);
	for(int i = 0; i < arraySize; i += aPoint.ByteSize())
	{
		*reinterpret_cast<int32_t*>(dataBuff.get() + i)  = aPoint.ByteSize();
		i += 4;
		aPoint.SerializeToArray(dataBuff.get() + i, aPoint.ByteSize());
	}
	handlerPtr->HandleBulkImport(tempWorker,bulkImportMessage, dataBuff.get());
	auto& column1 = db->GetTables().at("test").GetColumns().at("test");
	ASSERT_EQ(aPoint.geopoint().latitude(), dynamic_cast<BlockBase<ColmnarDB::Types::Point>&>(*dynamic_cast<ColumnBase<ColmnarDB::Types::Point>&>(*column1).GetBlocksList()[0]).GetData()[0].geopoint().latitude());
	ASSERT_EQ(aPoint.geopoint().longitude(), dynamic_cast<BlockBase<ColmnarDB::Types::Point>&>(*dynamic_cast<ColumnBase<ColmnarDB::Types::Point>&>(*column1).GetBlocksList()[0]).GetData()[0].geopoint().longitude());
	Database::RemoveFromInMemoryDatabaseList("test");
}