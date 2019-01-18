#include "TCPClientHandler.h"
#include "ITCPWorker.h"
#include "CSVInMemoryImporter.h"
#include "Configuration.h"
#include <functional>
#include <stdexcept>
#include <boost/timer/timer.hpp>
#include "messages/QueryResponseMessage.pb.h"

std::unique_ptr<google::protobuf::Message> TCPClientHandler::GetNextQueryResult()
{
	if (lastResultMessage_ == nullptr)
	{
		if (lastQueryResult_.valid())
		{
			lastResultMessage_ = std::move(lastQueryResult_.get());
		}
		else
		{
			auto infoMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
			infoMessage->set_message("");
			infoMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
			return infoMessage;
		}
	}
	auto* resultMessage = lastResultMessage_.get();
	if (dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultMessage) == nullptr)
	{
		return std::move(lastResultMessage_);
	}
	auto* completeResult = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultMessage);
	if (lastResultLen_ == 0)
	{
		for (const auto& payload : completeResult->payloads())
		{
			switch (payload.second.payload_case())
			{
			case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kIntPayload:
				lastResultLen_ = std::max(payload.second.intpayload().intdata().size(), lastResultLen_);
				break;
			case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kFloatPayload:
				lastResultLen_ = std::max(payload.second.floatpayload().floatdata().size(), lastResultLen_);
				break;
			case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kInt64Payload:
				lastResultLen_ = std::max(payload.second.int64payload().int64data().size(), lastResultLen_);
				break;
			case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kDoublePayload:
				lastResultLen_ = std::max(payload.second.doublepayload().doubledata().size(), lastResultLen_);
				break;
			case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kPointPayload:
				lastResultLen_ = std::max(payload.second.doublepayload().doubledata().size(), lastResultLen_);
				break;
			case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kPolygonPayload:
				lastResultLen_ = std::max(payload.second.polygonpayload().polygondata().size(), lastResultLen_);
				break;
			case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kStringPayload:
				lastResultLen_ = std::max(payload.second.stringpayload().stringdata().size(), lastResultLen_);
				break;
			}
		}
		if (lastResultLen_ < FRAGMENT_SIZE)
		{
			lastResultLen_ = 0;
			return std::move(lastResultMessage_);
		}
	}
	std::unique_ptr<ColmnarDB::NetworkClient::Message::QueryResponseMessage> smallPayload;
	if (sentRecords_ == 0)
	{
		for (const auto& timing : completeResult->timing()) 
		{
			smallPayload->mutable_timing()->insert(timing);
		}
	}

	for(const auto& payload : completeResult->payloads())
	{
		int bufferSize = FRAGMENT_SIZE > (lastResultLen_ - sentRecords_) ? (lastResultLen_ - sentRecords_) : FRAGMENT_SIZE;
		ColmnarDB::NetworkClient::Message::QueryResponsePayload finalPayload;
		switch (payload.second.payload_case())
		{
		case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kIntPayload:
			for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
			{
				finalPayload.mutable_intpayload()->add_intdata(payload.second.intpayload().intdata()[i]);
			}
			break;
		case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kFloatPayload:
			for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
			{
				finalPayload.mutable_floatpayload()->add_floatdata(payload.second.floatpayload().floatdata()[i]);
			}
			break;
		case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kInt64Payload:
			for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
			{
				finalPayload.mutable_int64payload()->add_int64data(payload.second.int64payload().int64data()[i]);
			}
			break;
		case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kDoublePayload:
			for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
			{
				finalPayload.mutable_doublepayload()->add_doubledata(payload.second.doublepayload().doubledata()[i]);
			}
			break;
		case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kPointPayload:
			for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
			{
				*finalPayload.mutable_pointpayload()->add_pointdata() = payload.second.pointpayload().pointdata()[i];
			}
			break;
		case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kPolygonPayload:
			for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
			{
				*finalPayload.mutable_polygonpayload()->add_polygondata() = payload.second.polygonpayload().polygondata()[i];
			}
			break;
		case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kStringPayload:
			for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
			{
				finalPayload.mutable_stringpayload()->add_stringdata(payload.second.stringpayload().stringdata()[i]);
			}
			break;
		}
		smallPayload->mutable_payloads()->insert({ payload.first,finalPayload });
	}
	sentRecords_ += FRAGMENT_SIZE;
	if (sentRecords_ >= lastResultLen_)
	{
		sentRecords_ = 0;
		lastResultLen_ = 0;
		lastResultMessage_.reset();
	}
	return std::move(smallPayload);
	
}

std::unique_ptr<google::protobuf::Message> TCPClientHandler::RunQuery(const std::weak_ptr<Database>& database, const ColmnarDB::NetworkClient::Message::QueryMessage & queryMessage)
{
	try
	{
		boost::timer::cpu_timer queryTimer;
		//parser
		auto elapsed = queryTimer.elapsed();
		auto parserTime = elapsed.system + elapsed.user;
		//executor
		{
			std::lock_guard<std::mutex> queryLock(queryMutex_);
			//execute
		}
		return std::make_unique<ColmnarDB::NetworkClient::Message::QueryResponseMessage>();
	}
	catch (std::exception& e)
	{
		auto infoMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
		infoMessage->set_message(e.what());
		infoMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
		return infoMessage;
	}
}

std::unique_ptr<google::protobuf::Message> TCPClientHandler::HandleInfoMessage(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::InfoMessage & infoMessage)
{
	if (infoMessage.code() == ColmnarDB::NetworkClient::Message::InfoMessage::CONN_END)
	{
		worker.Abort();
	}
	else if (infoMessage.code() == ColmnarDB::NetworkClient::Message::InfoMessage::GET_NEXT_RESULT)
	{
		return GetNextQueryResult();
	}
	else
	{
		//Log.WarnFormat("Invalid InfoMessage received, Code = {0}", infoMessage.Code);
	}
	return nullptr;
}

std::unique_ptr<google::protobuf::Message> TCPClientHandler::HandleQuery(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::QueryMessage & queryMessage)
{
	sentRecords_ = 0;
	lastResultLen_ = 0;
	lastQueryResult_ = std::async(std::launch::async, std::bind(&TCPClientHandler::RunQuery, this, worker.currentDatabase_, queryMessage));
	auto resultMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
	resultMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::WAIT);
	resultMessage->set_message("");
	return resultMessage;
}

std::unique_ptr<google::protobuf::Message> TCPClientHandler::HandleCSVImport(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::CSVImportMessage & csvImportMessage)
{
	CSVInMemoryImporter dataImporter(csvImportMessage.csvname(), csvImportMessage.payload());
	auto resultMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
	try
	{
		auto importDB = Database::GetDatabaseByName(csvImportMessage.databasename());
		auto importDBShared = importDB.lock();
		if (importDBShared == nullptr)
		{
			std::shared_ptr<Database> newImportDB = std::make_shared<Database>(csvImportMessage.databasename(), Configuration::BlockSize());
			dataImporter.ImportTables(*newImportDB);
			Database::AddToInMemoryDatabaseList(newImportDB);
		}
		else
		{
			dataImporter.ImportTables(*importDBShared);
		}
	}
	catch (std::exception& e)
	{
		std::cerr << "CSVImport: " << e.what() << std::endl;
	}
	return resultMessage;
}

std::unique_ptr<google::protobuf::Message> TCPClientHandler::HandleSetDatabase(ITCPWorker & worker, const ColmnarDB::NetworkClient::Message::SetDatabaseMessage & setDatabaseMessage)
{
	auto resultMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
	worker.currentDatabase_ = Database::GetDatabaseByName(setDatabaseMessage.databasename());
	if (!worker.currentDatabase_.expired())
	{
		resultMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
		resultMessage->set_message("");
	}
	else
	{
		resultMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
		resultMessage->set_message("No such database");
	}
	return resultMessage;
}
