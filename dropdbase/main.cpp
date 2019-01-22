#include "Table.h"
#include "Database.h"
#include <memory>
#include <vector>
#include "ColumnBase.h"
#include "DatabaseGenerator.h"

int main(int argc, char** argv)
{
	auto db = DatabaseGenerator::GenerateDatabase("test", 1, 20, false);
	auto& colInt = dynamic_cast<const ColumnBase<int>&>(*(db->GetTables().at("TableA").GetColumns().at("colInteger")));
	auto& blockTest = dynamic_cast<BlockBase<int>&>(*(colInt.GetBlocksList()[0]));
	for (auto& datum : blockTest.GetData())
	{
		std::cout << datum << '\n';
	}
	return 0;
}