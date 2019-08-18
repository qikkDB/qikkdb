#pragma once
#include <string>
#include <any>
#include <unordered_map>
#include "DataType.h"
#include <boost/asio.hpp>

namespace TellStoryDB::NetworkClient
{
struct TypedResult
{
    std::any resultData;
    DataType resultType;
};

struct Result
{
    std::unordered_map<std::string, TypedResult> columnData;
    std::unordered_map<std::string, float> executionTimes;

    bool IsValueNull(const std::string& columnName, size_t valueIndex)
    {
        size_t byteIdx = (valueIndex + sizeof(int8_t) * 8 - 1) / (sizeof(int8_t) * 8);
        int32_t shiftIdx = (valueIndex + sizeof(int8_t) * 8 - 1) % (sizeof(int8_t) * 8);
        return (nullMasks.at(columnName).at(byteIdx) >> shiftIdx) & 1;
    }

    bool IsColumnNullable(const std::string& columnName)
    {
        return nullMasks.find(columnName) != nullMasks.end();
    }

private:
    std::unordered_map<std::string, std::string> nullMasks;
    friend class Client;
};

class Client
{
public:
    Client(const std::string& ip, uint16_t port);
    void Connect();
    void Query(const std::string& query);
    Result GetNextQueryResult();
    void BulkImport(const std::string& tableName, const std::unordered_map<std::string, TypedResult>& data);
    void UseDatabase(const std::string& databaseName);
    void Heartbeat();
    void Close();

private:
    boost::asio::io_context ioContext_;
    boost::asio::ip::tcp::socket socket_;
    std::string ip_;
    uint16_t port_;
	constexpr inline static int BULK_IMPORT_FRAGMENT_SIZE = 8192 * 1024;
};

} // namespace TellStoryDB::NetworkClient