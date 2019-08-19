#include <iostream>
#include "NetworkClient.h"

int main()
{
    std::cout << "Driver test\n";
    TellStoryDB::NetworkClient::Client client("127.0.0.1", 12345);
    client.Connect();
    client.UseDatabase("TargetLocator");
    client.Query("SELECT ageId FROM TargetLoc10M LIMIT 2010;");
	auto result = client.GetNextQueryResult();
    while (!result.columnData.empty())
    {
        for (auto& result : result.columnData)
        {
            std::cout << result.first << "\n";
            auto& resultBuff = std::any_cast<std::vector<int32_t>>(result.second.resultData);
            for (auto& integer : resultBuff)
            {
                std::cout << integer << "\n";
            }
        }
        result = client.GetNextQueryResult();
    }
    client.Close();
    return 0;
}