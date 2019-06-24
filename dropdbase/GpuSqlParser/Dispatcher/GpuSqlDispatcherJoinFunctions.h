#pragma once

#include "../GpuSqlDispatcher.h"
#include "../JoinType.h"
#include "../../Database.h"
#include "../../Table.h"
#include "../../ColumnBase.h"
#include "../../QueryEngine/GPUCore/GPUJoin.cuh"
#include "../../QueryEngine/GPUCore/GPUFilterConditions.cuh"

template<typename OP, typename T>
int32_t GpuSqlDispatcher::joinCol()
{
	std::string colNameLeft = arguments.read<std::string>();
	std::string colNameRight = arguments.read<std::string>();
	JoinType joinType = static_cast<JoinType>(arguments.read<int32_t>());

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
	case JoinType::INNER_JOIN :
		GPUJoin::JoinTableRonS<OP, T>(leftJoinIndices, rightJoinIndices, *colBaseLeft, *colBaseRight, database->GetBlockSize());
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
int32_t GpuSqlDispatcher::joinConst()
{
	return 0;
}