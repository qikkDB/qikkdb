#include "CSVDataImporter.h"
#include "TCPClientHandler.h"
#include "ITCPWorker.h"
#include "Configuration.h"
#include "Database.h"
#include <functional>
#include <stdexcept>
#include <chrono>
#include "messages/QueryResponseMessage.pb.h"
#include "ConstraintViolationError.h"
#include <boost/log/trivial.hpp>
#include <boost/asio.hpp>

std::mutex TCPClientHandler::queryMutex_;
std::mutex TCPClientHandler::importMutex_;

std::unique_ptr<google::protobuf::Message> TCPClientHandler::GetNextQueryResult()
{
    BOOST_LOG_TRIVIAL(debug) << "TCPClientHandler: GetNextQueryResult()";
    if (lastQueryResult_.valid())
    {
        lastResultMessage_ = lastQueryResult_.get();
    }
    if (lastResultMessage_ == nullptr)
    {
        auto infoMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
        infoMessage->set_message("");
        infoMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
        return infoMessage;
    }
    auto* resultMessage = lastResultMessage_.get();
    if (dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultMessage) == nullptr)
    {
        return std::move(lastResultMessage_);
    }
    auto* completeResult = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultMessage);
    BOOST_LOG_TRIVIAL(debug) << "TCPClientHandler: LastResultLen: " << lastResultLen_;
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
            case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kDateTimePayload:
                lastResultLen_ =
                    std::max(payload.second.datetimepayload().datetimedata().size(), lastResultLen_);
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
            default:
                break;
            }
        }
        BOOST_LOG_TRIVIAL(debug) << "TCPClientHandler: New LastResultLen: " << lastResultLen_;
        if (lastResultLen_ < FRAGMENT_SIZE)
        {
            lastResultLen_ = 0;
            return std::move(lastResultMessage_);
        }
    }
    std::unique_ptr<ColmnarDB::NetworkClient::Message::QueryResponseMessage> smallPayload =
        std::make_unique<ColmnarDB::NetworkClient::Message::QueryResponseMessage>();
    for (const auto& column : completeResult->columnorder())
    {
        smallPayload->add_columnorder(column);
    }
    if (sentRecords_ == 0)
    {
        for (const auto& timing : completeResult->timing())
        {
            smallPayload->mutable_timing()->insert(timing);
        }
    }
    BOOST_LOG_TRIVIAL(debug) << "TCPClientHandler: Sent Records: " << sentRecords_;
    BOOST_LOG_TRIVIAL(debug) << "TCPClientHandler: Inserting payloads...\n";
    for (const auto& payload : completeResult->payloads())
    {
        int bufferSize =
            FRAGMENT_SIZE > (lastResultLen_ - sentRecords_) ? (lastResultLen_ - sentRecords_) : FRAGMENT_SIZE;
        BOOST_LOG_TRIVIAL(debug) << "TCPClientHandler: bufferSize: " << bufferSize;
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
                finalPayload.mutable_floatpayload()->add_floatdata(
                    payload.second.floatpayload().floatdata()[i]);
            }
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kInt64Payload:
            for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
            {
                finalPayload.mutable_int64payload()->add_int64data(
                    payload.second.int64payload().int64data()[i]);
            }
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kDateTimePayload:
            for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
            {
                finalPayload.mutable_datetimepayload()->add_datetimedata(
                    payload.second.datetimepayload().datetimedata()[i]);
            }
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kDoublePayload:
            for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
            {
                finalPayload.mutable_doublepayload()->add_doubledata(
                    payload.second.doublepayload().doubledata()[i]);
            }
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kPointPayload:
            for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
            {
                *finalPayload.mutable_pointpayload()->add_pointdata() =
                    payload.second.pointpayload().pointdata()[i];
            }
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kPolygonPayload:
            for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
            {
                *finalPayload.mutable_polygonpayload()->add_polygondata() =
                    payload.second.polygonpayload().polygondata()[i];
            }
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kStringPayload:
            for (int i = sentRecords_; i < sentRecords_ + bufferSize; i++)
            {
                finalPayload.mutable_stringpayload()->add_stringdata(
                    payload.second.stringpayload().stringdata()[i]);
            }
            break;
        default:
            break;
        }
        smallPayload->mutable_payloads()->insert({payload.first, finalPayload});
        if (completeResult->nullbitmasks().find(payload.first) != completeResult->nullbitmasks().end())
        {
            if ((FRAGMENT_SIZE % (sizeof(nullmask_t) * 8)) != 0)
            {
                throw std::runtime_error("TCPClientHandler::FRAGMENT_SIZE (" +
                                         std::to_string(FRAGMENT_SIZE) + ") must be a multiple of " +
                                         std::to_string(sizeof(nullmask_t) * 8));
            }
            int start = NullValues::GetNullBitMaskSize(sentRecords_);
            int nullMaskBufferSize = NullValues::GetNullBitMaskSize(bufferSize);
            ColmnarDB::NetworkClient::Message::QueryNullmaskPayload nullMasks;
            std::vector<nullmask_t> nullMaskBuffer(
                completeResult->nullbitmasks().at(payload.first).nullmask().begin() + start,
                completeResult->nullbitmasks().at(payload.first).nullmask().begin() + start + nullMaskBufferSize);

            for (size_t i = 0; i <= nullMaskBufferSize; i++)
            {
                nullMasks.add_nullmask(nullMaskBuffer[i]);
            }

            smallPayload->mutable_nullbitmasks()->insert({payload.first, nullMasks});
        }
    }
    sentRecords_ += FRAGMENT_SIZE;
    if (sentRecords_ >= lastResultLen_)
    {
        BOOST_LOG_TRIVIAL(debug) << "TCPClientHandler: Last Block, cleaning up";
        sentRecords_ = 0;
        lastResultLen_ = 0;
        lastResultMessage_.reset();
    }
    BOOST_LOG_TRIVIAL(debug) << "TCPClientHandler: Returning small payload";
    return std::move(smallPayload);
}

std::unique_ptr<google::protobuf::Message>
TCPClientHandler::RunQuery(const std::weak_ptr<Database>& database,
                           const ColmnarDB::NetworkClient::Message::QueryMessage& queryMessage,
                           std::function<void(std::unique_ptr<google::protobuf::Message>)> handler)
{
    std::unique_lock<std::mutex> dbLock{Database::dbMutex_};
    std::lock_guard<std::mutex> queryLock(queryMutex_);
    try
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto sharedDb = database.lock();
        parser_ = std::make_unique<GpuSqlCustomParser>(sharedDb, queryMessage.query());
        auto ret = parser_->Parse();
        auto end = std::chrono::high_resolution_clock::now();
        BOOST_LOG_TRIVIAL(info) << "TCPClientHandler: Elapsed: " << std::chrono::duration<float>(end - start).count() << " sec.";
        std::unique_ptr<google::protobuf::Message> notifyMessage = nullptr;
        if (auto response = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(ret.get()))
        {
            response->mutable_timing()->insert(
                {"Elapsed", std::chrono::duration<float>(end - start).count() * 1000});
            notifyMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
            dynamic_cast<ColmnarDB::NetworkClient::Message::InfoMessage*>(notifyMessage.get())->set_message("");
            dynamic_cast<ColmnarDB::NetworkClient::Message::InfoMessage*>(notifyMessage.get())
                ->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::GET_NEXT_RESULT);
        }
        else
        {
            notifyMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
            dynamic_cast<ColmnarDB::NetworkClient::Message::InfoMessage*>(notifyMessage.get())->set_message("");
            dynamic_cast<ColmnarDB::NetworkClient::Message::InfoMessage*>(notifyMessage.get())
                ->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
        }
        parser_ = nullptr;
        handler(std::move(notifyMessage));
        return ret;
    }
    catch (const std::exception& e)
    {
        auto notifyMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
        dynamic_cast<ColmnarDB::NetworkClient::Message::InfoMessage*>(notifyMessage.get())->set_message("");
        dynamic_cast<ColmnarDB::NetworkClient::Message::InfoMessage*>(notifyMessage.get())
            ->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
        handler(std::move(notifyMessage));
        auto infoMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
        infoMessage->set_message(e.what());
        infoMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
        return infoMessage;
    }
}

std::unique_ptr<google::protobuf::Message>
TCPClientHandler::HandleInfoMessage(ITCPWorker& worker,
                                    const ColmnarDB::NetworkClient::Message::InfoMessage& infoMessage)
{
    if (infoMessage.code() == ColmnarDB::NetworkClient::Message::InfoMessage::CONN_END)
    {
        worker.Abort();
    }
    else if (infoMessage.code() == ColmnarDB::NetworkClient::Message::InfoMessage::GET_NEXT_RESULT)
    {
        return GetNextQueryResult();
    }
    else if (infoMessage.code() == ColmnarDB::NetworkClient::Message::InfoMessage::HEARTBEAT)
    {
        auto infoMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
        infoMessage->set_message("");
        infoMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
        return infoMessage;
    }
    else
    {
        BOOST_LOG_TRIVIAL(warning) << "TCPClientHandler: Invalid InfoMessage received, Code = " << infoMessage.code();
    }
    return nullptr;
}

std::unique_ptr<google::protobuf::Message>
TCPClientHandler::HandleQuery(ITCPWorker& worker,
                              const ColmnarDB::NetworkClient::Message::QueryMessage& queryMessage,
                              std::function<void(std::unique_ptr<google::protobuf::Message> notifyMessage)> handler)
{
    sentRecords_ = 0;
    lastResultLen_ = 0;
    BOOST_LOG_TRIVIAL(info) << "TCPClientHandler: " << queryMessage.query();
    lastQueryResult_ =
        std::async(std::launch::async, std::bind(&TCPClientHandler::RunQuery, this,
                                                 worker.currentDatabase_, queryMessage, handler));
    auto resultMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
    resultMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::WAIT);
    resultMessage->set_message("");
    return resultMessage;
}

std::unique_ptr<google::protobuf::Message>
TCPClientHandler::HandleCSVImport(ITCPWorker& worker,
                                  const ColmnarDB::NetworkClient::Message::CSVImportMessage& csvImportMessage)
{
    CSVDataImporter dataImporter(csvImportMessage.payload().c_str(), csvImportMessage.csvname().c_str());
    if (csvImportMessage.columntypes_size() > 0)
    {
        std::vector<DataType> types;
        std::transform(csvImportMessage.columntypes().cbegin(),
                       csvImportMessage.columntypes().cend(), std::back_inserter(types),
                       [](int32_t x) -> DataType { return static_cast<DataType>(x); });
        dataImporter.SetTypes(types);
    }
    auto resultMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
    try
    {
        std::lock_guard<std::mutex> importLock(importMutex_);
        auto importDB = Database::GetDatabaseByName(csvImportMessage.databasename());
        if (importDB == nullptr)
        {
            std::shared_ptr<Database> newImportDB =
                std::make_shared<Database>(csvImportMessage.databasename().c_str(),
                                           Configuration::GetInstance().GetBlockSize());
            Database::AddToInMemoryDatabaseList(newImportDB);
            dataImporter.ImportTables(newImportDB);
        }
        else
        {
            dataImporter.ImportTables(importDB);
        }
    }
    catch (std::exception& e)
    {
        BOOST_LOG_TRIVIAL(error) << "TCPClientHandler: CSVImport has failed: " << e.what();
    }
    return resultMessage;
}

std::unique_ptr<google::protobuf::Message>
TCPClientHandler::HandleSetDatabase(ITCPWorker& worker,
                                    const ColmnarDB::NetworkClient::Message::SetDatabaseMessage& setDatabaseMessage)
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

std::unique_ptr<google::protobuf::Message>
TCPClientHandler::HandleBulkImport(ITCPWorker& worker,
                                   const ColmnarDB::NetworkClient::Message::BulkImportMessage& bulkImportMessage,
                                   const char* dataBuffer,
                                   const char* nullMask)
{
    auto resultMessage = std::make_unique<ColmnarDB::NetworkClient::Message::InfoMessage>();
    std::string tableName = bulkImportMessage.tablename();
    std::string columnName = bulkImportMessage.columnname();
    DataType columnType = static_cast<DataType>(bulkImportMessage.columntype());
    int32_t elementCount = bulkImportMessage.elemcount();
    bool isNullable = bulkImportMessage.nullmasklen() != 0;
    auto sharedDb = worker.currentDatabase_.lock();
    if (sharedDb == nullptr)
    {
        resultMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
        resultMessage->set_message("Database was not found");
        return resultMessage;
    }
    std::lock_guard<std::mutex> importLock(importMutex_);
    auto& tables = sharedDb->GetTables();
    auto search = tables.find(tableName);
    if (search == tables.end())
    {
        std::unordered_map<std::string, DataType> columns;
        columns.insert({columnName, columnType});
        sharedDb->CreateTable(columns, tableName.c_str());
    }

    auto& table = tables.at(tableName);
    if (!table.ContainsColumn(columnName.c_str()))
    {
        table.CreateColumn(columnName.c_str(), columnType);
    }

    auto& column = table.GetColumns().at(columnName);

    if (column->GetColumnType() != columnType)
    {
        resultMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
        resultMessage->set_message((std::string("Column type mismatch in column ") + columnName).c_str());
        return resultMessage;
    }
    std::unordered_map<std::string, std::any> columnData;
    if (columnType == COLUMN_INT8_T)
    {
        std::vector<int8_t> dataVector;
        std::copy(dataBuffer, dataBuffer + elementCount, std::back_inserter(dataVector));
        columnData.insert({columnName, dataVector});
    }
    else if (columnType == COLUMN_INT)
    {
        std::vector<int32_t> dataVector;
        std::copy(reinterpret_cast<const int32_t*>(dataBuffer),
                  reinterpret_cast<const int32_t*>(dataBuffer) + elementCount, std::back_inserter(dataVector));
        columnData.insert({columnName, dataVector});
    }
    else if (columnType == COLUMN_LONG)
    {
        std::vector<int64_t> dataVector;
        std::copy(reinterpret_cast<const int64_t*>(dataBuffer),
                  reinterpret_cast<const int64_t*>(dataBuffer) + elementCount, std::back_inserter(dataVector));
        columnData.insert({columnName, dataVector});
    }
    else if (columnType == COLUMN_FLOAT)
    {
        std::vector<float> dataVector;
        std::copy(reinterpret_cast<const float*>(dataBuffer),
                  reinterpret_cast<const float*>(dataBuffer) + elementCount, std::back_inserter(dataVector));
        columnData.insert({columnName, dataVector});
    }
    else if (columnType == COLUMN_DOUBLE)
    {
        std::vector<double> dataVector;
        std::copy(reinterpret_cast<const double*>(dataBuffer),
                  reinterpret_cast<const double*>(dataBuffer) + elementCount, std::back_inserter(dataVector));
        columnData.insert({columnName, dataVector});
    }
    else if (columnType == COLUMN_POINT)
    {
        std::vector<ColmnarDB::Types::Point> dataVector;
        int i = 0;
        int elemsRead = 0;
        while (elemsRead < elementCount)
        {
            ColmnarDB::Types::Point point;
            int32_t size = *reinterpret_cast<const int32_t*>(dataBuffer + i);
            i += sizeof(int32_t);
            point.ParseFromArray(dataBuffer + i, size);
            i += size;
            elemsRead++;
            dataVector.push_back(point);
        }
        columnData.insert({columnName, dataVector});
    }
    else if (columnType == COLUMN_POLYGON)
    {
        std::vector<ColmnarDB::Types::ComplexPolygon> dataVector;
        int i = 0;
        int elemsRead = 0;
        while (elemsRead < elementCount)
        {
            ColmnarDB::Types::ComplexPolygon polygon;
            int32_t size = *reinterpret_cast<const int32_t*>(dataBuffer + i);
            i += sizeof(int32_t);
            polygon.ParseFromArray(dataBuffer + i, size);
            i += size;
            elemsRead++;
            dataVector.push_back(polygon);
        }
        columnData.insert({columnName, dataVector});
    }
    else if (columnType == COLUMN_STRING)
    {
        std::vector<std::string> dataVector;
        int i = 0;
        int elemsRead = 0;
        while (elemsRead < elementCount)
        {

            int32_t size = *reinterpret_cast<const int32_t*>(dataBuffer + i);
            i += sizeof(int32_t);
            std::string str(dataBuffer + i, size);
            i += size;
            elemsRead++;
            dataVector.push_back(str);
        }
        columnData.insert({columnName, dataVector});
    }
    if (isNullable)
    {
        constexpr size_t bitMultiplier = sizeof(nullmask_t) / sizeof(uint8_t); // out:in null mask bit count ratio
        std::vector<nullmask_t> nullMaskVector(
            NullValues::GetNullBitMaskSize(bulkImportMessage.nullmasklen() * (8 * sizeof(uint8_t))));
        // Cycle per out null mask
        for (int i = 0; i < nullMaskVector.size(); i++)
        {
            nullMaskVector[i] = 0;
            // Cycle to sequentially fill in one full number of out null mask (using OR)
            for (int j = 0; j < bitMultiplier; j++)
            {
                const size_t inputIndex = bitMultiplier * i + j;
                if (inputIndex < bulkImportMessage.nullmasklen())
                {
                    nullMaskVector[i] |= static_cast<nullmask_t>(nullMask[inputIndex])
                                         << (j * sizeof(uint8_t) * 8U);
                }
                else
                {
                    break;
                }
            }
        }
        std::unordered_map<std::string, std::vector<nullmask_t>> nullMap;
        nullMap.insert({columnName, nullMaskVector});

        try
        {
            table.InsertData(columnData, false, nullMap);
        }
        catch (constraint_violation_error& e)
        {
            BOOST_LOG_TRIVIAL(warning) << "TCPClientHandler: " << e.what();
        }
        catch (const std::exception& e)
        {
            resultMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
            resultMessage->set_message(e.what());
            return resultMessage;
        }
    }
    else
    {
        try
        {
            table.InsertData(columnData, false);
        }
        catch (constraint_violation_error& e)
        {
            BOOST_LOG_TRIVIAL(warning) << "TCPClientHandler: " << e.what();
        }
        catch (const std::exception& e)
        {
            resultMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
            resultMessage->set_message(e.what());
            return resultMessage;
        }
    }
    resultMessage->set_code(ColmnarDB::NetworkClient::Message::InfoMessage::OK);
    resultMessage->set_message("");
    return resultMessage;
}

void TCPClientHandler::Abort()
{
    if (parser_)
    {
        BOOST_LOG_TRIVIAL(info) << "TCPClientHandler: Got request from client to abort parser (cancel query execution).";
        parser_->InterruptQueryExecution();
    }
}
