#include "Table.h"
#include "Database.h"
#include <memory>
#include <vector>
#include "ColumnBase.h"


int main(int argc, char** argv)
{
	std::shared_ptr<Database> db = std::make_shared<Database>();
	Table tbl(db,"TestTbl");
	tbl.CreateColumn("TestCol",COLUMN_INT);
	std::unordered_map<std::string,std::any> data;
	data.insert({"TestCol",std::vector<int>({1,2,3,4,5})});
	tbl.InsertData(data);
	auto& cols = tbl.GetColumns();
	auto& column = cols.at("TestCol");
	ColumnBase<int>* testCol = dynamic_cast<ColumnBase<int>*>(column.get());
	auto& block = testCol->GetBlocksList().front();
	BlockBase<int>* blockTest = dynamic_cast<BlockBase<int>*>(block.get());
	for (auto& datum : blockTest->GetData())
	{
		std::cout << datum << '\n';
	}
	return 0;
}