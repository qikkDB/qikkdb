#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
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
			std::tuple<uintptr_t, int32_t, bool> col = allocatedPointers.at(colName + (groupByColumns.find(colName) != groupByColumns.end()? "_keys" : ""));
			outSize = std::get<1>(col);
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
		std::tuple<uintptr_t, int32_t, bool> col = allocatedPointers.at(colName);
		int32_t inSize = std::get<1>(col);
		outData = std::make_unique<T[]>(inSize);
		//ToDo: Podmienene zapnut podla velkost buffera
		//GPUMemory::hostPin(outData.get(), inSize);
		GPUReconstruct::reconstructCol(outData.get(), &outSize, reinterpret_cast<T*>(std::get<0>(col)), reinterpret_cast<int8_t*>(filter_), inSize);
		//GPUMemory::hostUnregister(outData.get());
		std::cout << "dataSize: " << outSize << std::endl;
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

		// split colName to table and column name
		const size_t endOfPolyIdx = colName.find(".");
		const std::string table = colName.substr(0, endOfPolyIdx);
		const std::string column = colName.substr(endOfPolyIdx + 1);

		const int32_t blockCount = database->GetTables().at(table).GetColumns().at(column).get()->GetBlockCount();
		GpuSqlDispatcher::groupByDoneLimit_ = std::min(Context::getInstance().getDeviceCount() - 1, blockCount - 1);
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

			noLoad = false;
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

			noLoad = false;
		}
	}
	return 0;
}