#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUOrderBy.cuh"
#include "../../QueryEngine/GPUCore/GPUJoin.cuh"
#include "../../IVariantArray.h"
#include "../../VariantArray.h"
#include "../../Database.h"
#include "../../Table.h"
#include "../../ColumnBase.h"
#include "../../BlockBase.h"

template<typename T>
int32_t GpuSqlDispatcher::retConst()
{
	T cnst = arguments.read<T>();
	std::cout << "RET: cnst" << typeid(T).name() << std::endl;
	return 0;
}

/// Implementation of column return from SELECT clause
/// If GROUP BY clause is not present each column block is reconstructed based on the filter mask 
/// (generated from WHERE clause) and merged to response message
/// If GROUP BY is present nothing is reconstructed as the filtering was done prior to GROUP BY (in aggregation)
/// If GROUP BY is present the results are only coppied from GPU and merged to response message
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename T>
int32_t GpuSqlDispatcher::retCol()
{
	auto colName = arguments.read<std::string>();
	auto alias = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "RetCol: " << colName << ", thread: " << dispatcherThreadId << std::endl;

	int32_t outSize;
	std::unique_ptr<T[]> outData;

	if (usingGroupBy)
	{
		if (isOverallLastBlock)
		{
			std::tuple<uintptr_t, int32_t, bool> col = allocatedPointers.at(getAllocatedRegisterName(colName) + (std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) != groupByColumns.end()? "_keys" : ""));
			outSize = std::get<1>(col);

			if (usingOrderBy)
			{
				std::cout << "Reordering result block." << std::endl;
				std::tuple<uintptr_t, int32_t, bool> orderByIndices = allocatedPointers.at("$orderByIndices");
				GPUOrderBy::ReOrderByIdxInplace(reinterpret_cast<T*>(std::get<0>(col)), reinterpret_cast<int32_t*>(std::get<0>(orderByIndices)), outSize);
			}

			outData = std::make_unique<T[]>(outSize);
			GPUMemory::copyDeviceToHost(outData.get(), reinterpret_cast<T*>(std::get<0>(col)), outSize);
		}
		else
		{
			return 0;
		}
	}
	else
	{
		if (usingOrderBy)
		{
			if (isOverallLastBlock)
			{
				VariantArray<T>* reconstructedColumn = dynamic_cast<VariantArray<T>*>(reconstructedOrderByColumnsMerged.at(colName).get());
				outData = std::move(reconstructedColumn->getDataRef());
				outSize = reconstructedColumn->GetSize();
			}
			else
			{
				return 0;
			}
		}
		else
		{
			std::tuple<uintptr_t, int32_t, bool> col = allocatedPointers.at(getAllocatedRegisterName(colName));
			int32_t inSize = std::get<1>(col);
			outData = std::make_unique<T[]>(inSize);
			GPUReconstruct::reconstructCol(outData.get(), &outSize, reinterpret_cast<T*>(std::get<0>(col)), reinterpret_cast<int8_t*>(filter_), inSize);
			std::cout << "dataSize: " << outSize << std::endl;
		}
	}

	ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
	insertIntoPayload(payload, outData, outSize);
	MergePayloadToSelfResponse(alias, payload);
	return 0;
}

/// Implementation of the LOAD operation
/// Loads the current block of given column
/// Sets the last block (for current dispatcher instance and overall) flags
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename T>
int32_t GpuSqlDispatcher::loadCol(std::string& colName)
{
	if (allocatedPointers.find(colName) == allocatedPointers.end() && !colName.empty() && colName.front() != '$')
	{
		std::cout << "Load: " << colName << " " << typeid(T).name() << std::endl;

		std::string table;
		std::string column;

		std::tie(table, column) = splitColumnName(colName);

		const int32_t blockCount = usingJoin ? joinIndices->at(table).size() : database->GetTables().at(table).GetColumns().at(column).get()->GetBlockCount();
		GpuSqlDispatcher::deviceCountLimit_ = std::min(Context::getInstance().getDeviceCount() - 1, blockCount - 1);
		if (blockIndex >= blockCount)
		{
			return 1;
		}
		if (blockIndex >= blockCount - Context::getInstance().getDeviceCount())
		{
			isLastBlockOfDevice = true;
		}
		if (blockIndex == blockCount - 1)
		{
			isOverallLastBlock = true;
		}

		noLoad = false;

		if (loadNecessary == 0)
		{
			instructionPointer = jmpInstuctionPosition;
			return 12;
		}

		auto col = dynamic_cast<const ColumnBase<T>*>(database->GetTables().at(table).GetColumns().at(column).get());

		if (!usingJoin)
		{
			auto block = dynamic_cast<BlockBase<T>*>(col->GetBlocksList()[blockIndex]);

			if (block->IsCompressed())
			{
				size_t uncompressedSize = Compression::GetUncompressedDataElementsCount(block->GetData());
				size_t compressedSize = block->GetSize();
				auto cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<T>(
					database->GetName(), colName, blockIndex, uncompressedSize);
				if (!std::get<2>(cacheEntry))
				{
					T* deviceCompressed;
					GPUMemory::alloc(&deviceCompressed, compressedSize);
					GPUMemory::copyHostToDevice(deviceCompressed, block->GetData(), compressedSize);
					bool isDecompressed;
					Compression::Decompress(
						col->GetColumnType(),
						deviceCompressed,
						Compression::GetCompressedDataElementsCount(block->GetData()),
						std::get<0>(cacheEntry),
						Compression::GetUncompressedDataElementsCount(block->GetData()),
						Compression::GetCompressionBlocksCount(block->GetData()),
						block->GetMin(),
						block->GetMax(),
						isDecompressed,
						true
					);
					GPUMemory::free(deviceCompressed);
				}
				addCachedRegister(colName, std::get<0>(cacheEntry), uncompressedSize);
			}
			else
			{
				auto cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<T>(
					database->GetName(), colName, blockIndex, block->GetSize());
				if (!std::get<2>(cacheEntry))
				{
					GPUMemory::copyHostToDevice(std::get<0>(cacheEntry), block->GetData(), block->GetSize());
				}
				addCachedRegister(colName, std::get<0>(cacheEntry), block->GetSize());
			}
			noLoad = false;
		}

		else
		{
			std::cout << "Loading joined block." << std::endl;
			int32_t loadSize = joinIndices->at(table)[blockIndex].size();
			std::string joinCacheId = colName + "_join";
			for (auto& joinTable : *joinIndices)
			{
				joinCacheId += "_" + joinTable.first;
			}

			auto cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<T>(
				database->GetName(), joinCacheId, blockIndex, loadSize);
			if (!std::get<2>(cacheEntry))
			{
				int32_t outDataSize;
				GPUJoin::reorderByJoinTableCPU<T>(std::get<0>(cacheEntry), outDataSize, *col, blockIndex, joinIndices->at(table), database->GetBlockSize());
			}
			addCachedRegister(joinCacheId, std::get<0>(cacheEntry), loadSize);
			noLoad = false;
		}
	}
	return 0;
}