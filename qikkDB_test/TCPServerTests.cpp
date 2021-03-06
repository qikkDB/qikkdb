#include "gtest/gtest.h"
#include "../qikkDB/IClientHandler.h"
#include "../qikkDB/ClientPoolWorker.h"
#include "../qikkDB/messages/QueryResponseMessage.pb.h"
#include "../qikkDB/messages/BulkImportMessage.pb.h"
#include "../qikkDB/messages/SetDatabaseMessage.pb.h"
#include "../qikkDB/TCPServer.h"
#include "../qikkDB/NetworkMessage.h"
#include <boost/asio.hpp>
#include <thread>
#include <stdexcept>

class DummyClientHandler : public IClientHandler
{
    // Inherited via IClientHandler
    virtual std::unique_ptr<google::protobuf::Message>
    HandleInfoMessage(ITCPWorker& worker, const QikkDB::NetworkClient::Message::InfoMessage& infoMessage) override
    {
        std::unique_ptr<QikkDB::NetworkClient::Message::QueryResponseMessage> ret =
            std::make_unique<QikkDB::NetworkClient::Message::QueryResponseMessage>();
        if (infoMessage.code() == QikkDB::NetworkClient::Message::InfoMessage::GET_NEXT_RESULT)
        {
            QikkDB::NetworkClient::Message::QueryResponsePayload qrp;
            (*qrp.mutable_stringpayload()->add_stringdata()) = "test";
            ret->mutable_payloads()->insert({"test", qrp});
            ret->mutable_timing()->insert({"aaa", 2});
            ret->add_columnorder("test");
            return ret;
        }
        else if (infoMessage.code() == QikkDB::NetworkClient::Message::InfoMessage::HEARTBEAT)
        {
            std::unique_ptr<QikkDB::NetworkClient::Message::InfoMessage> ret =
                std::make_unique<QikkDB::NetworkClient::Message::InfoMessage>();
            ret->set_code(QikkDB::NetworkClient::Message::InfoMessage::OK);
            ret->set_message("");
            return ret;
        }
        else
        {
            worker.Abort();
            return nullptr;
        }
    }
    virtual std::unique_ptr<google::protobuf::Message>
    HandleQuery(ITCPWorker& worker,
                const QikkDB::NetworkClient::Message::QueryMessage& queryMessage,
                std::function<void(std::unique_ptr<google::protobuf::Message>)> handler) override
    {
        std::unique_ptr<QikkDB::NetworkClient::Message::InfoMessage> ret =
            std::make_unique<QikkDB::NetworkClient::Message::InfoMessage>();
        ret->set_code(QikkDB::NetworkClient::Message::InfoMessage::WAIT);
        ret->set_message("");
        return ret;
    }
    virtual std::unique_ptr<google::protobuf::Message>
    HandleCSVImport(ITCPWorker& worker, const QikkDB::NetworkClient::Message::CSVImportMessage& csvImportMessage) override
    {
        std::unique_ptr<QikkDB::NetworkClient::Message::InfoMessage> ret =
            std::make_unique<QikkDB::NetworkClient::Message::InfoMessage>();
        ret->set_code(QikkDB::NetworkClient::Message::InfoMessage::OK);
        ret->set_message("");
        return ret;
    }
    virtual std::unique_ptr<google::protobuf::Message>
    HandleSetDatabase(ITCPWorker& worker,
                      const QikkDB::NetworkClient::Message::SetDatabaseMessage& SetDatabaseMessage) override
    {
        std::unique_ptr<QikkDB::NetworkClient::Message::InfoMessage> ret =
            std::make_unique<QikkDB::NetworkClient::Message::InfoMessage>();
        ret->set_code(QikkDB::NetworkClient::Message::InfoMessage::OK);
        ret->set_message("");
        return ret;
    }
    virtual std::unique_ptr<google::protobuf::Message>
    HandleBulkImport(ITCPWorker& worker,
                     const QikkDB::NetworkClient::Message::BulkImportMessage& bulkImportMessage,
                     const char* dataBuffer,
                     const uint8_t* nullMask) override
    {
        std::unique_ptr<QikkDB::NetworkClient::Message::InfoMessage> ret =
            std::make_unique<QikkDB::NetworkClient::Message::InfoMessage>();
        if (std::string(bulkImportMessage.columnname()) != "test" ||
            std::string(bulkImportMessage.tablename()) != "test" ||
            bulkImportMessage.columntype() != QikkDB::NetworkClient::Message::DataType::COLUMN_INT ||
            bulkImportMessage.elemcount() != 5)
        {
            printf("Something wrong.\n");
            ret->set_code(QikkDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
        }
        else
        {
            ret->set_code(QikkDB::NetworkClient::Message::InfoMessage::OK);
            const int32_t* intData = reinterpret_cast<const int32_t*>(dataBuffer);
            for (int i = 0; i < 5; i++)
            {
                if (intData[i] != i + 1)
                {
                    printf("Data mismatch %d != %d.\n", intData[i], i + 1);
                    ret->set_code(QikkDB::NetworkClient::Message::InfoMessage::QUERY_ERROR);
                    break;
                }
            }
        }

        ret->set_message("");
        return ret;
    }

    virtual void Abort() override
    {
    }
};

boost::asio::ip::tcp::socket connectSocketToTestServer(boost::asio::io_context& context)
{
    boost::asio::ip::tcp::socket sock(context);
    boost::asio::ip::tcp::resolver resolver(context);
    auto endpoints = resolver.resolve("127.0.0.1", "12345");
    boost::asio::connect(sock, endpoints);
    return sock;
}

void connect(boost::asio::ip::tcp::socket& sock, boost::asio::io_context& context)
{
    try
    {
        NetworkMessage networkMessage;
        QikkDB::NetworkClient::Message::InfoMessage hello;
        hello.set_code(QikkDB::NetworkClient::Message::InfoMessage::CONN_ESTABLISH);
        hello.set_message("");
        std::promise<QikkDB::NetworkClient::Message::InfoMessage> promise;
        networkMessage.WriteToNetwork(hello, sock,
                                      [&sock, &promise, &networkMessage]() {
                                          networkMessage.ReadFromNetwork(
                                              sock,
                                              [&promise](google::protobuf::Any ret) {
                                                  QikkDB::NetworkClient::Message::InfoMessage infoMessage;
                                                  if (!ret.UnpackTo(&infoMessage))
                                                  {
                                                      promise.set_exception(std::make_exception_ptr(std::domain_error(
                                                          "Invalid message received")));
                                                  }
                                                  else
                                                  {
                                                      promise.set_value(infoMessage);
                                                  }
                                              },
                                              []() {});
                                      },
                                      []() {});
        auto future = promise.get_future();
        while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        {
            context.poll();
            context.restart();
        }
        ASSERT_EQ(future.get().code(), QikkDB::NetworkClient::Message::InfoMessage::OK);
    }
    catch (const std::exception& e)
    {
        std::string what = e.what();
        std::cout << what << "\n";
        throw e;
    }
}

void disconnect(boost::asio::ip::tcp::socket& sock, boost::asio::io_context& context)
{
    NetworkMessage networkMessage;
    QikkDB::NetworkClient::Message::InfoMessage infoMessage;
    infoMessage.set_code(QikkDB::NetworkClient::Message::InfoMessage::CONN_END);
    infoMessage.set_message("");
    networkMessage.WriteToNetwork(infoMessage, sock, []() {}, []() {});
    context.poll();
    context.restart();
}

void query(boost::asio::ip::tcp::socket& sock, const char* queryString, boost::asio::io_context& context)
{
    NetworkMessage networkMessage;
    QikkDB::NetworkClient::Message::QueryMessage query;
    query.set_query(queryString);
    std::promise<QikkDB::NetworkClient::Message::InfoMessage> promise;
    networkMessage.WriteToNetwork(query, sock,
                                  [&sock, &promise, &networkMessage]() {
                                      networkMessage.ReadFromNetwork(
                                          sock,
                                          [&promise](google::protobuf::Any ret) {
                                              QikkDB::NetworkClient::Message::InfoMessage infoMessage;
                                              if (!ret.UnpackTo(&infoMessage))
                                              {
                                                  promise.set_exception(std::make_exception_ptr(std::domain_error(
                                                      "Invalid message received")));
                                              }
                                              else
                                              {
                                                  promise.set_value(infoMessage);
                                              }
                                          },
                                          []() {});
                                  },
                                  []() {});
    auto future = promise.get_future();
    while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
        context.poll();
        context.restart();
    }
    ASSERT_EQ(future.get().code(), QikkDB::NetworkClient::Message::InfoMessage::WAIT);
}

QikkDB::NetworkClient::Message::QueryResponseMessage
getNextQueryResult(boost::asio::ip::tcp::socket& sock, boost::asio::io_context& context)
{
    NetworkMessage networkMessage;
    QikkDB::NetworkClient::Message::InfoMessage infoMessage;
    infoMessage.set_code(QikkDB::NetworkClient::Message::InfoMessage_StatusCode_GET_NEXT_RESULT);
    infoMessage.set_message("");
    std::promise<QikkDB::NetworkClient::Message::QueryResponseMessage> retPromise;
    networkMessage.WriteToNetwork(infoMessage, sock,
                                  [&sock, &retPromise, &networkMessage]() {
                                      networkMessage.ReadFromNetwork(
                                          sock,
                                          [&retPromise](google::protobuf::Any response) {
                                              QikkDB::NetworkClient::Message::QueryResponseMessage ret;
                                              if (!response.UnpackTo(&ret))
                                              {
                                                  retPromise.set_exception(std::make_exception_ptr(std::domain_error(
                                                      "Invalid message received")));
                                              }
                                              else
                                              {
                                                  retPromise.set_value(ret);
                                              }
                                          },
                                          []() {});
                                  },
                                  []() {});
    auto future = retPromise.get_future();
    while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
        context.poll();
        context.restart();
    }
    return future.get();
}

void importCSV(boost::asio::ip::tcp::socket& sock, const char* name, const char* data, boost::asio::io_context& context)
{
    NetworkMessage networkMessage;
    QikkDB::NetworkClient::Message::CSVImportMessage csvImport;
    csvImport.set_databasename(name);
    csvImport.set_payload(data);
    csvImport.set_csvname(name);
    std::promise<QikkDB::NetworkClient::Message::InfoMessage> promise;
    networkMessage.WriteToNetwork(csvImport, sock,
                                  [&sock, &promise, &networkMessage]() {
                                      networkMessage.ReadFromNetwork(
                                          sock,
                                          [&sock, &promise](google::protobuf::Any ret) {
                                              QikkDB::NetworkClient::Message::InfoMessage infoMessage;
                                              if (!ret.UnpackTo(&infoMessage))
                                              {
                                                  promise.set_exception(std::make_exception_ptr(std::domain_error(
                                                      "Invalid message received")));
                                              }
                                              else
                                              {
                                                  promise.set_value(infoMessage);
                                              }
                                          },
                                          []() {});
                                  },
                                  []() {});
    auto future = promise.get_future();
    while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
        context.poll();
        context.restart();
    }
    ASSERT_EQ(future.get().code(), QikkDB::NetworkClient::Message::InfoMessage::OK);
}

void setDatabase(boost::asio::ip::tcp::socket& sock, const char* name, boost::asio::io_context& context)
{
    NetworkMessage networkMessage;
    QikkDB::NetworkClient::Message::SetDatabaseMessage setDbMsg;
    setDbMsg.set_databasename(name);
    std::promise<QikkDB::NetworkClient::Message::InfoMessage> promise;
    networkMessage.WriteToNetwork(setDbMsg, sock,
                                  [&sock, &promise, &networkMessage]() {
                                      networkMessage.ReadFromNetwork(
                                          sock,
                                          [&sock, &promise](google::protobuf::Any ret) {
                                              QikkDB::NetworkClient::Message::InfoMessage infoMessage;
                                              if (!ret.UnpackTo(&infoMessage))
                                              {
                                                  promise.set_exception(std::make_exception_ptr(std::domain_error(
                                                      "Invalid message received")));
                                              }
                                              else
                                              {
                                                  promise.set_value(infoMessage);
                                              }
                                          },
                                          []() {});
                                  },
                                  []() {});
    auto future = promise.get_future();
    while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
        context.poll();
        context.restart();
    }
    ASSERT_EQ(future.get().code(), QikkDB::NetworkClient::Message::InfoMessage::OK);
}

void bulkImport(boost::asio::ip::tcp::socket& sock, boost::asio::io_context& context)
{
    NetworkMessage networkMessage;
    QikkDB::NetworkClient::Message::BulkImportMessage bulkImportMessage;
    bulkImportMessage.set_tablename("test");
    bulkImportMessage.set_columnname("test");
    bulkImportMessage.set_columntype(static_cast<QikkDB::NetworkClient::Message::DataType>(DataType::COLUMN_INT));
    bulkImportMessage.set_elemcount(5);
    bulkImportMessage.set_datalength(5 * sizeof(int32_t));
    int32_t dataBuff[] = {1, 2, 3, 4, 5};
    std::promise<QikkDB::NetworkClient::Message::InfoMessage> promise;
    networkMessage.WriteToNetwork(
        bulkImportMessage, sock,
        [&sock, &promise, dataBuff, &networkMessage]() mutable {
            networkMessage.WriteRaw(sock, reinterpret_cast<char*>(dataBuff), 5, DataType::COLUMN_INT,
                                    [&sock, &promise, &networkMessage]() {
                                        networkMessage.ReadFromNetwork(
                                            sock,
                                            [&promise](google::protobuf::Any response) {
                                                QikkDB::NetworkClient::Message::InfoMessage infoMessage;
                                                if (!response.UnpackTo(&infoMessage))
                                                {
                                                    promise.set_exception(std::make_exception_ptr(std::domain_error(
                                                        "Invalid message received")));
                                                }
                                                else
                                                {
                                                    promise.set_value(infoMessage);
                                                }
                                            },
                                            []() {});
                                    },
                                    []() {});
        },
        []() {});
    auto future = promise.get_future();
    while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
        context.poll();
        context.restart();
    }
    ASSERT_EQ(future.get().code(), QikkDB::NetworkClient::Message::InfoMessage::OK);
}

void heartbeat(boost::asio::ip::tcp::socket& sock, boost::asio::io_context& context)
{
    NetworkMessage networkMessage;
    QikkDB::NetworkClient::Message::InfoMessage heartbeat;
    heartbeat.set_code(QikkDB::NetworkClient::Message::InfoMessage::HEARTBEAT);
    heartbeat.set_message("");
    std::promise<QikkDB::NetworkClient::Message::InfoMessage> promise;
    networkMessage.WriteToNetwork(heartbeat, sock,
                                  [&sock, &promise, &networkMessage]() {
                                      networkMessage.ReadFromNetwork(
                                          sock,
                                          [&promise](google::protobuf::Any ret) {
                                              QikkDB::NetworkClient::Message::InfoMessage infoMessage;
                                              if (!ret.UnpackTo(&infoMessage))
                                              {
                                                  promise.set_exception(std::make_exception_ptr(std::domain_error(
                                                      "Invalid message received")));
                                              }
                                              else
                                              {
                                                  promise.set_value(infoMessage);
                                              }
                                          },
                                          []() {});
                                  },
                                  []() {});
    auto future = promise.get_future();
    while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
        context.poll();
        context.restart();
    }
    ASSERT_EQ(future.get().code(), QikkDB::NetworkClient::Message::InfoMessage::OK);
}

TEST(TCPServer, ServerMessageInfo)
{
    try
    {
        printf("\nServerMessageInfo\n");
        TCPServer<DummyClientHandler, ClientPoolWorker> testServer("127.0.0.1", 12345, false);
        auto future = std::thread([&testServer]() { testServer.Run(); });
        boost::asio::io_context context;
        auto sock = connectSocketToTestServer(context);
        ASSERT_NO_THROW(connect(sock, context));
        ASSERT_NO_THROW(disconnect(sock, context));
        testServer.Abort();
        future.join();
        printf("\nServerMessageInfoEnd\n");
    }
    catch (...)
    {
        ASSERT_TRUE(false);
    }
}

TEST(TCPServer, ServerMessageInfoHeartbeat)
{
    try
    {
        printf("\nServerMessageInfoHeartbeat\n");
        TCPServer<DummyClientHandler, ClientPoolWorker> testServer("127.0.0.1", 12345, false);
        auto future = std::thread([&testServer]() { testServer.Run(); });
        boost::asio::io_context context;
        auto sock = connectSocketToTestServer(context);
        ASSERT_NO_THROW(connect(sock, context));
        ASSERT_NO_THROW(heartbeat(sock, context));
        ASSERT_NO_THROW(disconnect(sock, context));
        testServer.Abort();
        future.join();
        printf("\nServerMessageInfoHeartbeat\n");
    }
    catch (...)
    {
        ASSERT_TRUE(false);
    }
}

TEST(TCPServer, ServerMessageSetDB)
{
    try
    {
        printf("\nServerMessageSetDB\n");
        TCPServer<DummyClientHandler, ClientPoolWorker> testServer("127.0.0.1", 12345, false);
        auto future = std::thread([&testServer]() { testServer.Run(); });
        boost::asio::io_context context;
        auto sock = connectSocketToTestServer(context);
        ASSERT_NO_THROW(connect(sock, context));
        ASSERT_NO_THROW(setDatabase(sock, "test", context));
        ASSERT_NO_THROW(disconnect(sock, context));
        testServer.Abort();
        future.join();
        printf("\nServerMessageSetDBEnd\n");
    }
    catch (...)
    {
        ASSERT_TRUE(false);
    }
}

TEST(TCPServer, ServerMessageQuery)
{
    try
    {
        printf("\nServerMessageQuery\n");
        TCPServer<DummyClientHandler, ClientPoolWorker> testServer("127.0.0.1", 12345, false);
        auto future = std::thread([&testServer]() { testServer.Run(); });
        boost::asio::io_context context;
        auto sock = connectSocketToTestServer(context);
        ASSERT_NO_THROW(connect(sock, context));
        ASSERT_NO_THROW(query(sock, "test", context));
        QikkDB::NetworkClient::Message::QueryResponseMessage resp;
        ASSERT_NO_THROW(resp = getNextQueryResult(sock, context));
        ASSERT_EQ(resp.payloads().at("test").stringpayload().stringdata()[0], "test");
        ASSERT_EQ(resp.columnorder().Get(0), "test");
        ASSERT_NO_THROW(disconnect(sock, context));
        testServer.Abort();
        future.join();
        printf("\nServerMessageQueryEnd\n");
    }
    catch (...)
    {
        ASSERT_TRUE(false);
    }
}

TEST(TCPServer, ServerMessageCSV)
{
    try
    {
        printf("\nServerMessageCSV\n");
        TCPServer<DummyClientHandler, ClientPoolWorker> testServer("127.0.0.1", 12345, false);
        auto future = std::thread([&testServer]() { testServer.Run(); });
        boost::asio::io_context context;
        auto sock = connectSocketToTestServer(context);
        ASSERT_NO_THROW(connect(sock, context));
        ASSERT_NO_THROW(importCSV(sock, "test", "test", context));
        ASSERT_NO_THROW(disconnect(sock, context));
        testServer.Abort();
        future.join();
        printf("\nServerMessageCSVEnd\n");
    }
    catch (...)
    {
        ASSERT_TRUE(false);
    }
}


TEST(TCPServer, ServerMessageBulkImport)
{
    try
    {
        printf("\nServerMessageBulkImport\n");
        TCPServer<DummyClientHandler, ClientPoolWorker> testServer("127.0.0.1", 12345, false);
        auto future = std::thread([&testServer]() { testServer.Run(); });
        boost::asio::io_context context;
        auto sock = connectSocketToTestServer(context);
        ASSERT_NO_THROW(connect(sock, context));
        ASSERT_NO_THROW(bulkImport(sock, context));
        ASSERT_NO_THROW(disconnect(sock, context));
        context.stop();
        testServer.Abort();
        future.join();
        printf("\nServerMessageBulkImportEnd\n");
    }
    catch (...)
    {
        ASSERT_TRUE(false);
    }
}