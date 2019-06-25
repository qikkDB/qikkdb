#pragma once

#include "GpuSqlJoinDispatcher.h"
#include "../QueryEngine/GPUCore/GPUJoin.cuh"
#include "../QueryEngine/GPUCore/GPUFilterConditions.cuh"

template<typename OP, typename T>
int32_t GpuSqlJoinDispatcher::joinCol()
{
	std::string colNameLeft = arguments.read<std::string>();
	std::string colNameRight = arguments.read<std::string>();
	JoinType joinType = static_cast<JoinType>(arguments.read<int32_t>());

	std::cout << "JoinCol: " << colNameLeft << " " << colNameRight << std::endl;

	std::string leftTable;
	std::string leftColumn;

	std::tie(leftTable, leftColumn) = splitColumnName(colNameLeft);

	std::string rightTable;
	std::string rightColumn;

	std::tie(rightTable, rightColumn) = splitColumnName(colNameRight);

	auto colBaseLeft = dynamic_cast<ColumnBase<T>*>(database->GetTables().at(leftTable).GetColumns().at(leftColumn).get());
	auto colBaseRight = dynamic_cast<ColumnBase<T>*>(database->GetTables().at(rightTable).GetColumns().at(rightColumn).get());

	std::vector<std::vector<int32_t>> leftJoinIndices;
	std::vector<std::vector<int32_t>> rightJoinIndices;

	switch (joinType)
	{
	case JoinType::INNER_JOIN:
		GPUJoin::JoinTableRonS<OP, T>(leftJoinIndices, rightJoinIndices, *colBaseLeft, *colBaseRight, database->GetBlockSize());
		//joinIndices.emplace(leftTable, std::move(leftJoinIndices));
		//joinIndices.emplace(rightTable, std::move(rightJoinIndices));

		std::cout << leftTable << std::endl;
		for (int32_t i = 0; i < leftJoinIndices.size(); i++)
		{
			for (int32_t j = 0; j < leftJoinIndices[i].size(); j++)
			{
				T dataLeft = dynamic_cast<BlockBase<T>*>(dynamic_cast<ColumnBase<T>*>(database->GetTables().at(leftTable).GetColumns().at(leftColumn).get())->GetBlocksList()[leftJoinIndices[i][j] / database->GetBlockSize()])->GetData()[leftJoinIndices[i][j] % database->GetBlockSize()];
				T dataRight = dynamic_cast<BlockBase<T>*>(dynamic_cast<ColumnBase<T>*>(database->GetTables().at(rightTable).GetColumns().at(rightColumn).get())->GetBlocksList()[rightJoinIndices[i][j] / database->GetBlockSize()])->GetData()[rightJoinIndices[i][j] % database->GetBlockSize()];
				std::cout << leftJoinIndices[i][j] << " " << dataLeft << std::endl;
				std::cout << rightJoinIndices[i][j] << " " << dataRight << std::endl;
				std::cout << std::endl;
			}
		}
		
		break;
	case JoinType::LEFT_JOIN:
	case JoinType::RIGHT_JOIN:
	case JoinType::FULL_OUTER_JOIN:
	default:
		break;
	}

	return 0;
}

template<typename OP, typename T>
int32_t GpuSqlJoinDispatcher::joinConst()
{
	return 0;
}
