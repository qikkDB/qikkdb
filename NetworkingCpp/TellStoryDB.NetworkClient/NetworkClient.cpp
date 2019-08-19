#include "NetworkClient.h"
#include "messages/InfoMessage.pb.h"
#include "messages/QueryMessage.pb.h"
#include "messages/SetDatabaseMessage.pb.h"
#include "messages/QueryResponseMessage.pb.h"
#include "messages/BulkImportMessage.pb.h"
#include "NetworkMessage.h"
#include <stdexcept>
#include <numeric>
#include <execution>

namespace TellStoryDB::NetworkClient
{

Client::Client(const std::string& ip, uint16_t port)
: ioContext_(), socket_(ioContext_), ip_(ip), port_(port)
{
}

void Client::Connect()
{
    boost::asio::ip::tcp::resolver resolver(ioContext_);
    auto endpoints = resolver.resolve(ip_, std::to_string(port_));
    boost::asio::connect(socket_, endpoints);
    ColmnarDB::NetworkClient::Message::InfoMessage hello;
    hello.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::CONN_ESTABLISH);
    hello.set_message("");
    NetworkMessage::WriteToNetwork(hello, socket_);
    auto ret = NetworkMessage::ReadFromNetwork(socket_);
    ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
    if (!ret.UnpackTo(&infoMessage))
    {
        throw std::invalid_argument("Invalid message received from server");
    }
    if (infoMessage.code() != ColmnarDB::NetworkClient::Message::InfoMessage::OK)
    {
        throw std::invalid_argument("Invalid message code received from server");
    }
}

void Client::Query(const std::string& queryString)
{
    ColmnarDB::NetworkClient::Message::QueryMessage query;
    query.set_query(queryString);
    NetworkMessage::WriteToNetwork(query, socket_);
    auto response = NetworkMessage::ReadFromNetwork(socket_);
    ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
    if (!response.UnpackTo(&infoMessage))
    {
        throw std::invalid_argument("Invalid message received from server");
    }
    if (infoMessage.code() != ColmnarDB::NetworkClient::Message::InfoMessage::WAIT)
    {
        throw std::invalid_argument(infoMessage.message());
    }
}

Result Client::GetNextQueryResult()
{
    ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
    infoMessage.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::GET_NEXT_RESULT);
    infoMessage.set_message("");
    NetworkMessage::WriteToNetwork(infoMessage, socket_);
    auto response = NetworkMessage::ReadFromNetwork(socket_);
    ColmnarDB::NetworkClient::Message::QueryResponseMessage queryResponse;
    Result ret;
    if (!response.UnpackTo(&queryResponse))
    {
        ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
        if (!response.UnpackTo(&infoMessage))
        {

            throw std::domain_error("Invalid message received");
        }
        if (infoMessage.code() != ColmnarDB::NetworkClient::Message::InfoMessage::OK)
        {
            throw std::runtime_error(infoMessage.message());
		}
        return ret;
    }
    for (auto& timing : queryResponse.timing())
    {
        ret.executionTimes.insert(timing);
    }
    for (auto& payload : queryResponse.payloads())
    {
        switch (payload.second.payload_case())
        {
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::kIntPayload:
            ret.columnData.emplace(
                payload.first,
                TypedResult{std::vector<int32_t>(payload.second.intpayload().intdata().begin(),
                                                 payload.second.intpayload().intdata().end()),
                            DataType::COLUMN_INT});
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::kInt64Payload:
            ret.columnData.emplace(
                payload.first,
                TypedResult{std::vector<int64_t>(payload.second.int64payload().int64data().begin(),
                                                 payload.second.int64payload().int64data().end()),
                            DataType::COLUMN_LONG});
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::kFloatPayload:
            ret.columnData.emplace(
                payload.first,
                TypedResult{std::vector<float>(payload.second.floatpayload().floatdata().begin(),
                                               payload.second.floatpayload().floatdata().end()),
                            DataType::COLUMN_FLOAT});
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::kDoublePayload:
            ret.columnData.emplace(
                payload.first,
                TypedResult{std::vector<double>(payload.second.doublepayload().doubledata().begin(),
                                                payload.second.doublepayload().doubledata().end()),
                            DataType::COLUMN_DOUBLE});
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::kStringPayload:
            ret.columnData.emplace(
                payload.first,
                TypedResult{std::vector<std::string>(payload.second.stringpayload().stringdata().begin(),
                                                     payload.second.stringpayload().stringdata().end()),
                            DataType::COLUMN_STRING});
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::kPolygonPayload:
            ret.columnData.emplace(payload.first,
                                   TypedResult{std::vector<ColmnarDB::Types::ComplexPolygon>(
                                                   payload.second.polygonpayload().polygondata().begin(),
                                                   payload.second.polygonpayload().polygondata().end()),
                                               DataType::COLUMN_POINT});
            break;
        case ColmnarDB::NetworkClient::Message::QueryResponsePayload::kPointPayload:
            ret.columnData.emplace(payload.first,
                                   TypedResult{std::vector<ColmnarDB::Types::Point>(
                                                   payload.second.pointpayload().pointdata().begin(),
                                                   payload.second.pointpayload().pointdata().end()),
                                               DataType::COLUMN_POLYGON});
            break;
        default:
            break;
        }
    }
    for (auto& nullMask : queryResponse.nullbitmasks())
    {
        ret.nullMasks.emplace(nullMask.first, std::move(nullMask.second));
    }
    return ret;
}

void Client::BulkImport(const std::string& tableName, const std::unordered_map<std::string, TypedResult>& data)
{
    ColmnarDB::NetworkClient::Message::BulkImportMessage bulkImportMessage;
    bulkImportMessage.set_tablename(tableName);
    for (auto& column : data)
    {
        bulkImportMessage.set_columnname("test");
        bulkImportMessage.set_columntype(
            static_cast<ColmnarDB::NetworkClient::Message::DataType>(column.second.resultType));
        std::unique_ptr<char[]> dataBuff = nullptr;
        size_t dataSize = 0;
        size_t idx = 0;
        size_t typeSize = 1;
        switch (column.second.resultType)
        {
        case DataType::COLUMN_INT:
            dataSize = std::any_cast<std::vector<int32_t>>(column.second.resultData).size() * sizeof(int32_t);
            dataBuff = std::make_unique<char[]>(dataSize);
            std::copy(reinterpret_cast<char*>(
                          std::any_cast<std::vector<int32_t>>(column.second.resultData).data()),
                      reinterpret_cast<char*>(
                          std::any_cast<std::vector<int32_t>>(column.second.resultData).data()) +
                          dataSize,
                      dataBuff.get());
            typeSize = sizeof(int32_t);
            break;
        case DataType::COLUMN_LONG:
            dataSize = std::any_cast<std::vector<int64_t>>(column.second.resultData).size() * sizeof(int64_t);
            dataBuff = std::make_unique<char[]>(dataSize);
            std::copy(reinterpret_cast<char*>(
                          std::any_cast<std::vector<int64_t>>(column.second.resultData).data()),
                      reinterpret_cast<char*>(
                          std::any_cast<std::vector<int64_t>>(column.second.resultData).data()) +
                          dataSize,
                      dataBuff.get());
            typeSize = sizeof(int64_t);
            break;
        case DataType::COLUMN_FLOAT:
            dataSize = std::any_cast<std::vector<float>>(column.second.resultData).size() * sizeof(float);
            dataBuff = std::make_unique<char[]>(dataSize);
            std::copy(
                reinterpret_cast<char*>(std::any_cast<std::vector<float>>(column.second.resultData).data()),
                reinterpret_cast<char*>(std::any_cast<std::vector<float>>(column.second.resultData).data()) + dataSize,
                dataBuff.get());
            typeSize = sizeof(float);
            break;
        case DataType::COLUMN_DOUBLE:
            dataSize = std::any_cast<std::vector<double>>(column.second.resultData).size() * sizeof(double);
            dataBuff = std::make_unique<char[]>(dataSize);
            std::copy(reinterpret_cast<char*>(
                          std::any_cast<std::vector<double>>(column.second.resultData).data()),
                      reinterpret_cast<char*>(
                          std::any_cast<std::vector<double>>(column.second.resultData).data()) +
                          dataSize,
                      dataBuff.get());
            typeSize = sizeof(double);
            break;
        case DataType::COLUMN_STRING:
            for (const auto& str : std::any_cast<std::vector<std::string>>(column.second.resultData))
            {
                dataSize += sizeof(int32_t) + str.length();
            }
            dataBuff = std::make_unique<char[]>(dataSize);
            for (const auto& str : std::any_cast<std::vector<std::string>>(column.second.resultData))
            {
                *reinterpret_cast<int32_t*>(dataBuff.get() + idx) = str.length();
                idx += sizeof(int32_t);
                std::copy(str.begin(), str.end(), dataBuff.get() + idx);
                idx += str.length();
            }
            break;
        case DataType::COLUMN_POLYGON:
            for (const auto& poly :
                 std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>>(column.second.resultData))
            {
                dataSize += sizeof(int32_t) + poly.ByteSize();
            }
            dataBuff = std::make_unique<char[]>(dataSize);
            for (const auto& poly :
                 std::any_cast<std::vector<ColmnarDB::Types::ComplexPolygon>>(column.second.resultData))
            {
                *reinterpret_cast<int32_t*>(dataBuff.get() + idx) = poly.ByteSize();
                idx += sizeof(int32_t);
                poly.SerializeToArray(dataBuff.get() + idx, poly.ByteSize());
                idx += poly.ByteSize();
            }
            break;
        case DataType::COLUMN_POINT:
            for (const auto& point :
                 std::any_cast<std::vector<ColmnarDB::Types::Point>>(column.second.resultData))
            {
                dataSize += sizeof(int32_t) + point.ByteSize();
            }
            dataBuff = std::make_unique<char[]>(dataSize);
            for (const auto& point :
                 std::any_cast<std::vector<ColmnarDB::Types::Point>>(column.second.resultData))
            {
                *reinterpret_cast<int32_t*>(dataBuff.get() + idx) = point.ByteSize();
                idx += sizeof(int32_t);
                point.SerializeToArray(dataBuff.get() + idx, point.ByteSize());
                idx += point.ByteSize();
            }
            break;
        default:
            break;
        }

        for (idx = 0; idx < dataSize; idx += BULK_IMPORT_FRAGMENT_SIZE)
        {
            int32_t fragmentSize =
                (dataSize - idx) < BULK_IMPORT_FRAGMENT_SIZE ? dataSize - idx : BULK_IMPORT_FRAGMENT_SIZE;
            bulkImportMessage.set_elemcount(fragmentSize / typeSize);
            NetworkMessage::WriteToNetwork(bulkImportMessage, socket_);
            NetworkMessage::WriteRaw(socket_, dataBuff.get() + idx, fragmentSize, column.second.resultType);
            idx += fragmentSize;
            auto response = NetworkMessage::ReadFromNetwork(socket_);
            ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
            if (!response.UnpackTo(&infoMessage))
            {
                throw std::invalid_argument("Invalid message received from server");
            }
            if (infoMessage.code() != ColmnarDB::NetworkClient::Message::InfoMessage::OK)
            {
                throw std::invalid_argument(infoMessage.message());
            }
        }
    }
}

void Client::UseDatabase(const std::string& databaseName)
{
    ColmnarDB::NetworkClient::Message::SetDatabaseMessage setDbMsg;
    setDbMsg.set_databasename(databaseName);
    NetworkMessage::WriteToNetwork(setDbMsg, socket_);
    auto response = NetworkMessage::ReadFromNetwork(socket_);
    ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
    if (!response.UnpackTo(&infoMessage))
    {
        throw std::invalid_argument("Invalid message received from server");
    }
    if (infoMessage.code() != ColmnarDB::NetworkClient::Message::InfoMessage::OK)
    {
        throw std::invalid_argument(infoMessage.message());
    }
}

void Client::Heartbeat()
{
    ColmnarDB::NetworkClient::Message::InfoMessage heartbeat;
    heartbeat.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::HEARTBEAT);
    heartbeat.set_message("");
    NetworkMessage::WriteToNetwork(heartbeat, socket_);
    auto ret = NetworkMessage::ReadFromNetwork(socket_);
    ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
    if (!ret.UnpackTo(&infoMessage))
    {
        throw std::invalid_argument("Invalid message received from server");
    }
    if (infoMessage.code() != ColmnarDB::NetworkClient::Message::InfoMessage::OK)
    {
        throw std::invalid_argument("Invalid message code received from server");
    }
}

void Client::Close()
{
    ColmnarDB::NetworkClient::Message::InfoMessage infoMessage;
    infoMessage.set_code(ColmnarDB::NetworkClient::Message::InfoMessage::CONN_END);
    infoMessage.set_message("");
    NetworkMessage::WriteToNetwork(infoMessage, socket_);
}
} // namespace TellStoryDB::NetworkClient