#pragma once
#include "GpuSqlParser/GpuSqlCustomParser.h"
#include <thread>
#include <mutex>
#include <google/protobuf/message.h>
#include <future>
#include "IClientHandler.h"
#include "Database.h"
#include <memory>


class TCPClientHandler final : public IClientHandler
{
private:
    const int FRAGMENT_SIZE = 1024; // Must be a multiple of sizeof(nullmask_t)*8
    int sentRecords_;
    int lastResultLen_;
    static std::mutex importMutex_;
    std::unique_ptr<google::protobuf::Message> GetNextQueryResult();
    std::future<std::unique_ptr<google::protobuf::Message>> lastQueryResult_;
    std::unique_ptr<google::protobuf::Message> lastResultMessage_;
    std::unique_ptr<google::protobuf::Message>
    RunQuery(const std::weak_ptr<Database>& database,
             const QikkDB::NetworkClient::Message::QueryMessage& queryMessage,
             std::function<void(std::unique_ptr<google::protobuf::Message> notifyMessage)> handler);
    std::unique_ptr<GpuSqlCustomParser> parser_;

public:
    TCPClientHandler(){};

    // Inherited via IClientHandler
    virtual std::unique_ptr<google::protobuf::Message>
    HandleInfoMessage(ITCPWorker& worker, const QikkDB::NetworkClient::Message::InfoMessage& infoMessage) override;
    virtual std::unique_ptr<google::protobuf::Message>
    HandleQuery(ITCPWorker& worker,
                const QikkDB::NetworkClient::Message::QueryMessage& queryMessage,
                std::function<void(std::unique_ptr<google::protobuf::Message>)> handler) override;
    virtual std::unique_ptr<google::protobuf::Message>
    HandleCSVImport(ITCPWorker& worker,
                    const QikkDB::NetworkClient::Message::CSVImportMessage& csvImportMessage) override;
    virtual std::unique_ptr<google::protobuf::Message>
    HandleSetDatabase(ITCPWorker& worker,
                      const QikkDB::NetworkClient::Message::SetDatabaseMessage& setDatabaseMessage) override;
    virtual std::unique_ptr<google::protobuf::Message>
    HandleBulkImport(ITCPWorker& worker,
                     const QikkDB::NetworkClient::Message::BulkImportMessage& bulkImportMessage,
                     const char* dataBuffer,
                     const char* nullMask = nullptr) override;
    virtual void Abort() override;
};
