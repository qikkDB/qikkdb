#pragma once
#include <memory>
#include <functional>
#include <google/protobuf/message.h>
#include "messages/InfoMessage.pb.h"
#include "messages/QueryMessage.pb.h"
#include "messages/CSVImportMessage.pb.h"
#include "messages/BulkImportMessage.pb.h"
#include "messages/SetDatabaseMessage.pb.h"

class ITCPWorker;
/// <summary>
/// Interface for handling client requests
/// </summary>
class IClientHandler
{
public:
    /// <summary>
    /// Handle InfoMessage
    /// </summary>
    /// <param name="worker">Worker that requested handling</param>
    /// <param name="infoMessage">Message to handle</param>
    virtual std::unique_ptr<google::protobuf::Message>
    HandleInfoMessage(ITCPWorker& worker, const QikkDB::NetworkClient::Message::InfoMessage& infoMessage) = 0;
    /// <summary>
    /// Start execution of a query
    /// </summary>
    /// <param name="worker">Worker that requested handling</param>
    /// <param name="queryMessage">Message to handle</param>
    /// <returns>InfoMessage telling client to wait for execution to finish</returns>
    virtual std::unique_ptr<google::protobuf::Message>
    HandleQuery(ITCPWorker& worker,
                const QikkDB::NetworkClient::Message::QueryMessage& queryMessage,
                std::function<void(std::unique_ptr<google::protobuf::Message>)> handler) = 0;
    /// <summary>
    /// Import sent CSV
    /// </summary>
    /// <param name="worker">Worker that requested handling</param>
    /// <param name="csvImportMessage">Message to handle</param>
    /// <returns>InfoMessage representing success state of the operation</returns>
    virtual std::unique_ptr<google::protobuf::Message>
    HandleCSVImport(ITCPWorker& worker,
                    const QikkDB::NetworkClient::Message::CSVImportMessage& csvImportMessage) = 0;
    /// <summary>
    /// Bulk import data
    /// </summary>
    /// <param name="worker">Worker that requested handling</param>
    /// <param name="bulkImportMessage">Message to handle</param>
    /// <returns>InfoMessage representing success state of the operation</returns>
    virtual std::unique_ptr<google::protobuf::Message>
    HandleBulkImport(ITCPWorker& worker,
                     const QikkDB::NetworkClient::Message::BulkImportMessage& bulkImportMessage,
                     const char* dataBuffer,
                     const char* nullMask = nullptr) = 0;
    /// <summary>
    /// Set working database
    /// </summary>
    /// <param name="worker">Worker that requested handling</param>
    /// <param name="setDatabaseMessage">Message to handle</param>
    /// <returns>InfoMessage representing success state of the operation</returns>
    virtual std::unique_ptr<google::protobuf::Message>
    HandleSetDatabase(ITCPWorker& worker,
                      const QikkDB::NetworkClient::Message::SetDatabaseMessage& SetDatabaseMessage) = 0;
    virtual void Abort() = 0;
    virtual ~IClientHandler(){};
};
