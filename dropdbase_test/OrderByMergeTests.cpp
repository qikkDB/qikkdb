#include "gtest/gtest.h"
#include "../dropdbase/DatabaseGenerator.h"
#include "../dropdbase/ColumnBase.h"
#include "../dropdbase/BlockBase.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/Table.h"
#include "../dropdbase/QueryEngine/Context.h"
#include "../dropdbase/GpuSqlParser/GpuSqlCustomParser.h"
#include "../dropdbase/messages/QueryResponseMessage.pb.h"
#include "DispatcherObjs.h"

// Pls dont delete, this is Nikolas' sandbox
TEST(OrderByMergeTests, OrderByMergeTest)
{
    // Set random generator
    int32_t SEED = 42;
    int32_t NUMERIC_LIMIT = 100;
    srand(SEED);

    // Get instance
    Context::getInstance();

    // Create custom database
    const std::string databaseName = "OrderByDatabase";
	const std::string tableName = "OrderByTable";
	const int32_t blockSize = 8;  
    const int32_t columnSize = 16;  
    const std::vector<int32_t> testData;

    std::shared_ptr<Database> database = std::make_shared<Database>(databaseName.c_str(), blockSize);
    Database::AddToInMemoryDatabaseList(database);

    // Create table with columns in database
    auto columns = std::unordered_map<std::string, DataType>();
    columns.insert(std::make_pair<std::string, DataType>("colInteger1", DataType::COLUMN_INT));
    columns.insert(std::make_pair<std::string, DataType>("colInteger2", DataType::COLUMN_INT));
    database->CreateTable(columns, tableName.c_str());

    // Create columns with int data
    std::vector<int32_t> colInteger1;
    std::vector<int32_t> colInteger2;
    for (int i = 0; i < columnSize; i++)
    {
        colInteger1.push_back(rand() % NUMERIC_LIMIT);
        colInteger2.push_back(rand() % NUMERIC_LIMIT);
    }

    // Insert the columns
    reinterpret_cast<ColumnBase<int32_t>*>(database->GetTables().at("OrderByTable").
        GetColumns().at("colInteger1").get())->InsertData(colInteger1);

    reinterpret_cast<ColumnBase<int32_t>*>(database->GetTables().at("OrderByTable").
        GetColumns().at("colInteger2").get())->InsertData(colInteger2);
    
    // Execute the query
    GpuSqlCustomParser parser(database, 
                              "SELECT colInteger1, colInteger2 FROM OrderByTable ORDER BY colInteger1 ASC, colInteger2 DESC;");
    auto resultPtr = parser.parse();
    auto result = dynamic_cast<ColmnarDB::NetworkClient::Message::QueryResponseMessage*>(resultPtr.get());

	//auto columnInt1 = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().at("OrderByTable").GetColumns().at("colInteger1").get());
	//auto columnInt2 = dynamic_cast<ColumnBase<int32_t>*>(database->GetTables().at("OrderByTable").GetColumns().at("colInteger2").get());

    auto &payload1 = result->payloads().at("OrderByTable.colInteger1");
    auto &payload2 = result->payloads().at("OrderByTable.colInteger2");

    for (int i = 0; i < payload1.intpayload().intdata_size(); i++)
    {

        std::printf("%5d: %5d %5d\n", i, payload1.intpayload().intdata()[i], payload2.intpayload().intdata()[i]);
    }
    
    Database::RemoveFromInMemoryDatabaseList(databaseName.c_str());
}