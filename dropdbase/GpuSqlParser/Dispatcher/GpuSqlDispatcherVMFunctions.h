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
	std::string nullMaskString = "";
	if (usingGroupBy)
	{
		if (isOverallLastBlock)
		{
			PointerAllocation col = allocatedPointers.at(getAllocatedRegisterName(colName) + (std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) != groupByColumns.end()? KEYS_SUFFIX : ""));
			outSize = col.elementCount;
			if (usingOrderBy)
			{
				std::cout << "Reordering result block." << std::endl;
				PointerAllocation orderByIndices = allocatedPointers.at("$orderByIndices");
				GPUOrderBy::ReOrderByIdxInplace(reinterpret_cast<T*>(col.gpuPtr), reinterpret_cast<int32_t*>(orderByIndices.gpuPtr), outSize);
			}

			outData = std::make_unique<T[]>(outSize);
			GPUMemory::copyDeviceToHost(outData.get(), reinterpret_cast<T*>(col.gpuPtr), outSize);
			if(col.gpuNullMaskPtr)
			{
				size_t bitMaskSize = (outSize + sizeof(char)*8 - 1) / (sizeof(char)*8);
				std::unique_ptr<int8_t[]> nullMask = std::unique_ptr<int8_t[]>(new int8_t[bitMaskSize]);
				GPUMemory::copyDeviceToHost(nullMask.get(), reinterpret_cast<int8_t*>(col.gpuNullMaskPtr), bitMaskSize);
				nullMaskString = std::string(reinterpret_cast<char*>(nullMask.get()), bitMaskSize);
			}
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

				size_t bitMaskSize = (outSize + sizeof(char) * 8 - 1) / (sizeof(char) * 8);
				nullMaskString = std::string(reinterpret_cast<char*>(reconstructedOrderByColumnsNullMerged.at(colName).get()), bitMaskSize);
			}
			else
			{
				return 0;
			}
		}
		else
		{
			PointerAllocation col = allocatedPointers.at(getAllocatedRegisterName(colName));
			int32_t inSize = col.elementCount;
			outData = std::make_unique<T[]>(inSize);
			//ToDo: Podmienene zapnut podla velkost buffera
			//GPUMemory::hostPin(outData.get(), inSize);
			if(col.gpuNullMaskPtr)
			{
				size_t bitMaskSize = (database->GetBlockSize() + sizeof(char)*8 - 1) / (sizeof(char)*8);
				std::unique_ptr<int8_t[]> nullMask = std::unique_ptr<int8_t[]>(new int8_t[bitMaskSize]);
				GPUReconstruct::reconstructCol(outData.get(), &outSize, reinterpret_cast<T*>(col.gpuPtr), reinterpret_cast<int8_t*>(filter_), col.elementCount, nullMask.get(), reinterpret_cast<int8_t*>(col.gpuNullMaskPtr));
				bitMaskSize = (outSize + sizeof(char)*8 - 1) / (sizeof(char)*8);
				nullMaskString = std::string(reinterpret_cast<char*>(nullMask.get()), bitMaskSize);
			}
			else
			{
				GPUReconstruct::reconstructCol(outData.get(), &outSize, reinterpret_cast<T*>(col.gpuPtr), reinterpret_cast<int8_t*>(filter_), col.elementCount);
			}
			//GPUMemory::hostUnregister(outData.get());
			std::cout << "dataSize: " << outSize << std::endl;
		}
	}

	ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
	insertIntoPayload(payload, outData, outSize);
	MergePayloadToSelfResponse(alias, payload, nullMaskString);
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

		auto col = dynamic_cast<const ColumnBase<T>*>(database->GetTables().at(table).GetColumns().at(column).get());

		if (!usingJoin)
		{
			int8_t* nullMaskPtr = nullptr;
			auto block = dynamic_cast<BlockBase<T>*>(col->GetBlocksList()[blockIndex]);
			size_t realSize;
			std::tuple<T*, size_t, bool> cacheEntry;
			if (block->IsCompressed())
			{
				size_t uncompressedSize = Compression::GetUncompressedDataElementsCount(block->GetData());
				size_t compressedSize = block->GetSize();
				cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<T>(
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
				
				realSize = uncompressedSize;
			}
			else
			{
				cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<T>(
					database->GetName(), colName, blockIndex, block->GetSize());
				if (!std::get<2>(cacheEntry))
				{
					GPUMemory::copyHostToDevice(std::get<0>(cacheEntry), block->GetData(), block->GetSize());
				}
				
				realSize = block->GetSize();
				
			}

			if(block->GetNullBitmask())
			{
				int32_t bitMaskCapacity = ((realSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
				auto cacheMaskEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<int8_t>(
					database->GetName(), colName + NULL_SUFFIX, blockIndex, bitMaskCapacity);
				nullMaskPtr = std::get<0>(cacheMaskEntry);
				if (!std::get<2>(cacheMaskEntry))
				{
					GPUMemory::copyHostToDevice(std::get<0>(cacheMaskEntry), block->GetNullBitmask(), bitMaskCapacity);
				}
				addCachedRegister(colName + NULL_SUFFIX, std::get<0>(cacheMaskEntry), bitMaskCapacity);
			}
			addCachedRegister(colName, std::get<0>(cacheEntry), realSize, nullMaskPtr);
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