#pragma once

#include "GpuSqlJoinDispatcher.h"
#include "../QueryEngine/GPUCore/GPUJoin.cuh"
#include "../QueryEngine/GPUCore/GPUMergeJoin.cuh"
#include "../QueryEngine/GPUCore/GPUFilterConditions.cuh"

template <typename OP, typename T>
int32_t GpuSqlJoinDispatcher::JoinCol()
{
    std::string colNameLeft = arguments_.Read<std::string>();
    std::string colNameRight = arguments_.Read<std::string>();
    JoinType joinType = static_cast<JoinType>(arguments_.Read<int32_t>());

    CudaLogBoost::getInstance(CudaLogBoost::info) << "JoinCol: " << colNameLeft << " " << colNameRight << '\n';

    std::string leftTable;
    std::string leftColumn;

    std::tie(leftTable, leftColumn) = SplitColumnName(colNameLeft);

    std::string rightTable;
    std::string rightColumn;

    std::tie(rightTable, rightColumn) = SplitColumnName(colNameRight);

    auto colBaseLeft = dynamic_cast<ColumnBase<T>*>(
        database_->GetTables().at(leftTable).GetColumns().at(leftColumn).get());
    auto colBaseRight = dynamic_cast<ColumnBase<T>*>(
        database_->GetTables().at(rightTable).GetColumns().at(rightColumn).get());

    std::vector<std::vector<int32_t>> leftJoinIndices;
    std::vector<std::vector<int32_t>> rightJoinIndices;

    switch (joinType)
    {
    case JoinType::INNER_JOIN:
		{
			bool colLUnique = colBaseLeft->GetIsUnique();
			bool colRUnique = colBaseRight->GetIsUnique();

			if (colLUnique && colRUnique)
			{
				MergeJoin::JoinUnique(leftJoinIndices, rightJoinIndices, *colBaseLeft, *colBaseRight);
			}
			if (colLUnique && !colRUnique)
			{
				MergeJoin::JoinUnique(leftJoinIndices, rightJoinIndices, *colBaseRight, *colBaseLeft);
			}
			if (!colLUnique && colRUnique)
			{
				MergeJoin::JoinUnique(leftJoinIndices, rightJoinIndices, *colBaseLeft, *colBaseRight);
			}
			if (!colLUnique && !colRUnique)
			{
				GPUJoin::JoinTableRonS<OP, T>(leftJoinIndices, rightJoinIndices, *colBaseLeft,
											  *colBaseRight, database_->GetBlockSize(), aborted_);
			}

			joinIndices_.emplace(leftTable, std::move(leftJoinIndices));
			joinIndices_.emplace(rightTable, std::move(rightJoinIndices));
		}
        break;
    case JoinType::LEFT_JOIN:
        break;
    case JoinType::RIGHT_JOIN:
        break;
    case JoinType::FULL_OUTER_JOIN:
        break;
    default:
        break;
    }

    return 0;
}

template <typename OP, typename T>
int32_t GpuSqlJoinDispatcher::JoinConst()
{
    return 0;
}
