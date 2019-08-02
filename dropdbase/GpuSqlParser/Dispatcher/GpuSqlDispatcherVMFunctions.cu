#include "GpuSqlDispatcherVMFunctions.h"
#include <array>
#include "../ParserExceptions.h"
#include "../../PointFactory.h"
#include "../../CudaLogBoost.h"

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::retFunctions = { &GpuSqlDispatcher::retConst<int32_t>, &GpuSqlDispatcher::retConst<int64_t>, &GpuSqlDispatcher::retConst<float>, &GpuSqlDispatcher::retConst<double>, &GpuSqlDispatcher::retConst<ColmnarDB::Types::Point>, &GpuSqlDispatcher::retConst<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::retConst<std::string>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<int8_t>, &GpuSqlDispatcher::retCol<int32_t>, &GpuSqlDispatcher::retCol<int64_t>, &GpuSqlDispatcher::retCol<float>, &GpuSqlDispatcher::retCol<double>, &GpuSqlDispatcher::retCol<ColmnarDB::Types::Point>, &GpuSqlDispatcher::retCol<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::retCol<std::string>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<int8_t> };
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::lockRegisterFunction = &GpuSqlDispatcher::lockRegister;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::filFunction = &GpuSqlDispatcher::fil;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::whereEvaluationFunction = &GpuSqlDispatcher::whereEvaluation;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::jmpFunction = &GpuSqlDispatcher::jmp;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::doneFunction = &GpuSqlDispatcher::done;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::showDatabasesFunction = &GpuSqlDispatcher::showDatabases;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::showTablesFunction = &GpuSqlDispatcher::showTables;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::showColumnsFunction = &GpuSqlDispatcher::showColumns;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::insertIntoDoneFunction = &GpuSqlDispatcher::insertIntoDone;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::createDatabaseFunction = &GpuSqlDispatcher::createDatabase;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::dropDatabaseFunction = &GpuSqlDispatcher::dropDatabase;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::createTableFunction = &GpuSqlDispatcher::createTable;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::dropTableFunction = &GpuSqlDispatcher::dropTable;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::alterTableFunction = &GpuSqlDispatcher::alterTable;
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::createIndexFunction = &GpuSqlDispatcher::createIndex;

template <>
int32_t GpuSqlDispatcher::loadCol<ColmnarDB::Types::ComplexPolygon>(std::string& colName)
{
	if (allocatedPointers.find(colName + "_polyPoints") == allocatedPointers.end() && !colName.empty() && colName.front() != '$')
	{
		CudaLogBoost::getInstance(CudaLogBoost::Severity::info) << "Loaded Column: " << colName << " " << typeid(ColmnarDB::Types::ComplexPolygon).name();
		std::cout << "Load: " << colName << " " << typeid(ColmnarDB::Types::ComplexPolygon).name() << std::endl;

		std::string table;
		std::string column;

		std::tie(table, column) = splitColumnName(colName);

		const int32_t blockCount = database->GetTables().at(table).GetColumns().at(column).get()->GetBlockCount();
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

		auto col = dynamic_cast<const ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(database->GetTables().at(table).GetColumns().at(column).get());


		if (!usingJoin)
		{
			auto block = dynamic_cast<BlockBase<ColmnarDB::Types::ComplexPolygon>*>(col->GetBlocksList()[blockIndex]);
			int8_t* nullMaskPtr = nullptr;
			if(block->GetNullBitmask())
			{
				if(allocatedPointers.find(colName + NULL_SUFFIX) == allocatedPointers.end())
				{
					int32_t bitMaskCapacity = ((block->GetSize() + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
					nullMaskPtr = allocateRegister<int8_t>(colName + NULL_SUFFIX, bitMaskCapacity);
					GPUMemory::copyHostToDevice(nullMaskPtr, block->GetNullBitmask(), bitMaskCapacity);
				}
				else
				{
					nullMaskPtr = reinterpret_cast<int8_t*>(allocatedPointers.at(colName + NULL_SUFFIX).gpuPtr);
				}
			}
			insertComplexPolygon(database->GetName(), colName,
				std::vector<ColmnarDB::Types::ComplexPolygon>(block->GetData(),
					block->GetData() + block->GetSize()),
				block->GetSize(), false, nullMaskPtr);
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

			std::vector<ColmnarDB::Types::ComplexPolygon> joinedPolygons;
			int8_t* nullMaskPtr = nullptr;

			int32_t outDataSize;
			GPUJoin::reorderByJoinTableCPU<ColmnarDB::Types::ComplexPolygon>(joinedPolygons, outDataSize, *col, blockIndex, joinIndices->at(table), database->GetBlockSize());
			
			if (col->GetIsNullable())
			{
				if(allocatedPointers.find(colName + NULL_SUFFIX) == allocatedPointers.end())
				{
					int32_t bitMaskCapacity = ((loadSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
					auto cacheMaskEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<int8_t>(
						database->GetName(), joinCacheId + NULL_SUFFIX, blockIndex, bitMaskCapacity);
					nullMaskPtr = std::get<0>(cacheMaskEntry);
					if (!std::get<2>(cacheMaskEntry))
					{
						int32_t outMaskSize;
						GPUJoin::reorderNullMaskByJoinTableCPU<ColmnarDB::Types::ComplexPolygon>(std::get<0>(cacheMaskEntry), outMaskSize, *col, blockIndex, joinIndices->at(table), database->GetBlockSize());
					}
				}
				else
				{
					nullMaskPtr = reinterpret_cast<int8_t*>(allocatedPointers.at(colName + NULL_SUFFIX).gpuPtr);
				}
			}

			insertComplexPolygon(database->GetName(), colName, joinedPolygons, loadSize, nullMaskPtr);
			noLoad = false;
		}
	}
	return 0;
}

template <>
int32_t GpuSqlDispatcher::loadCol<ColmnarDB::Types::Point>(std::string& colName)
{
	if (allocatedPointers.find(colName) == allocatedPointers.end() && !colName.empty() && colName.front() != '$')
	{
		std::cout << "Load: " << colName << " " << typeid(ColmnarDB::Types::Point).name() << std::endl;

		std::string table;
		std::string column;

		std::tie(table, column) = splitColumnName(colName);

		const int32_t blockCount = database->GetTables().at(table).GetColumns().at(column).get()->GetBlockCount();
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

		auto col = dynamic_cast<const ColumnBase<ColmnarDB::Types::Point>*>(database->GetTables().at(table).GetColumns().at(column).get());
		
		if (!usingJoin)
		{
			auto block = dynamic_cast<BlockBase<ColmnarDB::Types::Point>*>(col->GetBlocksList()[blockIndex]);

			std::vector<NativeGeoPoint> nativePoints;
			std::transform(block->GetData(), block->GetData() + block->GetSize(), std::back_inserter(nativePoints), [](const ColmnarDB::Types::Point& point) -> NativeGeoPoint { return NativeGeoPoint{ point.geopoint().latitude(), point.geopoint().longitude() }; });

			auto cacheEntry =
				Context::getInstance().getCacheForCurrentDevice().getColumn<NativeGeoPoint>(database->GetName(), colName, blockIndex,
					nativePoints.size());
			if (!std::get<2>(cacheEntry))
			{
				GPUMemory::copyHostToDevice(std::get<0>(cacheEntry),
					reinterpret_cast<NativeGeoPoint*>(nativePoints.data()),
					nativePoints.size());
			}
			int8_t* nullMaskPtr = nullptr;
			if(block->GetNullBitmask())
			{
				if(allocatedPointers.find(colName + NULL_SUFFIX) == allocatedPointers.end())
				{
					int32_t bitMaskCapacity = ((block->GetSize() + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
					auto cacheMaskEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<int8_t>(
						database->GetName(), colName + NULL_SUFFIX, blockIndex, bitMaskCapacity);
					nullMaskPtr = std::get<0>(cacheMaskEntry);
					if (!std::get<2>(cacheMaskEntry))
					{
						GPUMemory::copyHostToDevice(std::get<0>(cacheMaskEntry), block->GetNullBitmask(), bitMaskCapacity);
					}
					addCachedRegister(colName + NULL_SUFFIX, std::get<0>(cacheMaskEntry), bitMaskCapacity);
				}
				else
				{
					nullMaskPtr = reinterpret_cast<int8_t*>(allocatedPointers.at(colName + NULL_SUFFIX).gpuPtr);
				}
			}
			addCachedRegister(colName, std::get<0>(cacheEntry), nativePoints.size(), nullMaskPtr);
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

			std::vector<ColmnarDB::Types::Point> joinedPoints;
			int8_t* nullMaskPtr = nullptr;
			int32_t outDataSize;
			GPUJoin::reorderByJoinTableCPU<ColmnarDB::Types::Point>(joinedPoints, outDataSize, *col, blockIndex, joinIndices->at(table), database->GetBlockSize());

			std::vector<NativeGeoPoint> nativePoints;
			std::transform(joinedPoints.data(), joinedPoints.data() + loadSize, std::back_inserter(nativePoints), [](const ColmnarDB::Types::Point& point) -> NativeGeoPoint { return NativeGeoPoint{ point.geopoint().latitude(), point.geopoint().longitude() }; });

			auto cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<NativeGeoPoint>(
				database->GetName(), joinCacheId, blockIndex, loadSize);
			if (!std::get<2>(cacheEntry))
			{
				GPUMemory::copyHostToDevice(std::get<0>(cacheEntry),
					reinterpret_cast<NativeGeoPoint*>(nativePoints.data()),
					nativePoints.size());
			}

			if (col->GetIsNullable())
			{
				if(allocatedPointers.find(colName + NULL_SUFFIX) == allocatedPointers.end())
				{
					int32_t bitMaskCapacity = ((loadSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
					auto cacheMaskEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<int8_t>(
						database->GetName(), joinCacheId + NULL_SUFFIX, blockIndex, bitMaskCapacity);
					nullMaskPtr = std::get<0>(cacheMaskEntry);
					if (!std::get<2>(cacheMaskEntry))
					{
						int32_t outMaskSize;
						GPUJoin::reorderNullMaskByJoinTableCPU<ColmnarDB::Types::Point>(std::get<0>(cacheMaskEntry), outMaskSize, *col, blockIndex, joinIndices->at(table), database->GetBlockSize());
					}
					addCachedRegister(colName + NULL_SUFFIX, std::get<0>(cacheMaskEntry), bitMaskCapacity);
				}
				else
				{
					nullMaskPtr = reinterpret_cast<int8_t*>(allocatedPointers.at(colName + NULL_SUFFIX).gpuPtr);
				}
			}

			addCachedRegister(colName, std::get<0>(cacheEntry), loadSize, nullMaskPtr);
			noLoad = false;
		}
	}
	return 0;
}


template <>
int32_t GpuSqlDispatcher::loadCol<std::string>(std::string& colName)
{
	if (allocatedPointers.find(colName + "_allChars") == allocatedPointers.end() && !colName.empty() && colName.front() != '$')
	{
		std::cout << "Load: " << colName << " " << typeid(std::string).name() << std::endl;

		std::string table;
		std::string column;

		std::tie(table, column) = splitColumnName(colName);

		const int32_t blockCount = database->GetTables().at(table).GetColumns().at(column).get()->GetBlockCount();
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

		auto col = dynamic_cast<const ColumnBase<std::string>*>(database->GetTables().at(table).GetColumns().at(column).get());

		if (!usingJoin)
		{
			auto block = dynamic_cast<BlockBase<std::string>*>(col->GetBlocksList()[blockIndex]);
			int8_t* nullMaskPtr = nullptr;
			if(block->GetNullBitmask())
			{
				if(allocatedPointers.find(colName + NULL_SUFFIX) == allocatedPointers.end())
				{
					int32_t bitMaskCapacity = ((block->GetSize() + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
					nullMaskPtr = allocateRegister<int8_t>(colName + NULL_SUFFIX, bitMaskCapacity);
					GPUMemory::copyHostToDevice(nullMaskPtr, block->GetNullBitmask(), bitMaskCapacity);
				}
				else
				{
					nullMaskPtr = reinterpret_cast<int8_t*>(allocatedPointers.at(colName + NULL_SUFFIX).gpuPtr);
				}
			}
			insertString(database->GetName(), colName, std::vector<std::string>(block->GetData(), 
				block->GetData() + block->GetSize()),
				block->GetSize(), false, nullMaskPtr);
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

			std::vector<std::string> joinedStrings;
			int8_t* nullMaskPtr = nullptr;

			int32_t outDataSize;
			GPUJoin::reorderByJoinTableCPU<std::string>(joinedStrings, outDataSize, *col, blockIndex, joinIndices->at(table), database->GetBlockSize());

			if (col->GetIsNullable())
			{
				if(allocatedPointers.find(colName + NULL_SUFFIX) == allocatedPointers.end())
				{
					int32_t bitMaskCapacity = ((loadSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
					auto cacheMaskEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<int8_t>(
						database->GetName(), joinCacheId + NULL_SUFFIX, blockIndex, bitMaskCapacity);
					nullMaskPtr = std::get<0>(cacheMaskEntry);
					if (!std::get<2>(cacheMaskEntry))
					{
						int32_t outMaskSize;
						GPUJoin::reorderNullMaskByJoinTableCPU<std::string>(std::get<0>(cacheMaskEntry), outMaskSize, *col, blockIndex, joinIndices->at(table), database->GetBlockSize());
					}
				}
				else
				{
					nullMaskPtr = reinterpret_cast<int8_t*>(allocatedPointers.at(colName + NULL_SUFFIX).gpuPtr);
				}
			}

			insertString(database->GetName(), colName, joinedStrings, loadSize, nullMaskPtr);
			noLoad = false;
		}

	}
	return 0;
}

template <>
int32_t GpuSqlDispatcher::retCol<ColmnarDB::Types::ComplexPolygon>()
{
	if (usingGroupBy)
	{
		throw RetPolygonGroupByException();
	}
	else
	{
		auto col = arguments.read<std::string>();
		auto alias = arguments.read<std::string>();

		int32_t loadFlag = loadCol<ColmnarDB::Types::ComplexPolygon>(col);
		if (loadFlag)
		{
			return loadFlag;
		}
		std::cout << "RetPolygonCol: " << col << ", thread: " << dispatcherThreadId << std::endl;

		std::unique_ptr<std::string[]> outData(new std::string[database->GetBlockSize()]);
		std::tuple<GPUMemory::GPUPolygon, int32_t, int8_t*> ACol = findComplexPolygon(col);
		int32_t outSize;
		std::string nullMaskString = "";
		if(std::get<2>(ACol))
		{
			size_t bitMaskSize = (database->GetBlockSize() + sizeof(char)*8 - 1) / (sizeof(char)*8);
			std::unique_ptr<int8_t[]> nullMask = std::unique_ptr<int8_t[]>(new int8_t[bitMaskSize]);
			GPUReconstruct::ReconstructPolyColToWKT(outData.get(), &outSize,
			std::get<0>(ACol), reinterpret_cast<int8_t*>(filter_), std::get<1>(ACol),nullMask.get(), std::get<2>(ACol));
			bitMaskSize = (outSize + sizeof(char)*8 - 1) / (sizeof(char)*8);
			nullMaskString = std::string(reinterpret_cast<char*>(nullMask.get()), bitMaskSize);
		}
		else
		{
			GPUReconstruct::ReconstructPolyColToWKT(outData.get(), &outSize,
			std::get<0>(ACol), reinterpret_cast<int8_t*>(filter_), std::get<1>(ACol));
		}
		std::cout << "dataSize: " << outSize << std::endl;

		if (outSize > 0)
		{
			ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
			insertIntoPayload(payload, outData, outSize);
			MergePayloadToSelfResponse(alias, payload, nullMaskString);
		}
	}
	return 0;
}

template<>
int32_t GpuSqlDispatcher::retCol<ColmnarDB::Types::Point>()
{
	if (usingGroupBy)
	{
		throw RetPointGroupByException();
	}
	else
	{
		auto colName = arguments.read<std::string>();
		auto alias = arguments.read<std::string>();

		int32_t loadFlag = loadCol<ColmnarDB::Types::Point>(colName);
		if (loadFlag)
		{
			return loadFlag;
		}

		std::cout << "RetPointCol: " << colName << ", thread: " << dispatcherThreadId << std::endl;

		std::unique_ptr<std::string[]> outData(new std::string[database->GetBlockSize()]);
		PointerAllocation ACol = allocatedPointers.at(colName);
		int32_t outSize;
		//ToDo: Podmienene zapnut podla velkost buffera
		//GPUMemory::hostPin(outData.get(), database->GetBlockSize());
		
		std::string nullMaskString = "";
		if(ACol.gpuNullMaskPtr)
		{
			size_t bitMaskSize = (database->GetBlockSize() + sizeof(char)*8 - 1) / (sizeof(char)*8);
			std::unique_ptr<int8_t[]> nullMask = std::unique_ptr<int8_t[]>(new int8_t[bitMaskSize]);
			GPUReconstruct::ReconstructPointColToWKT(outData.get(), &outSize,
				reinterpret_cast<NativeGeoPoint*>(ACol.gpuPtr), reinterpret_cast<int8_t*>(filter_), ACol.elementCount,
				nullMask.get(), reinterpret_cast<int8_t*>(ACol.gpuNullMaskPtr));
		    bitMaskSize = (outSize + sizeof(char)*8 - 1) / (sizeof(char)*8); 
			nullMaskString = std::string(reinterpret_cast<char*>(nullMask.get()), bitMaskSize);
		}
		else
		{
			GPUReconstruct::ReconstructPointColToWKT(outData.get(), &outSize,
				reinterpret_cast<NativeGeoPoint*>(ACol.gpuPtr), reinterpret_cast<int8_t*>(filter_), ACol.elementCount);
		}
		//GPUMemory::hostUnregister(outData.get());

		std::cout << "dataSize: " << outSize << std::endl;

		if (outSize > 0)
		{
			ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
			insertIntoPayload(payload, outData, outSize);
			MergePayloadToSelfResponse(alias, payload, nullMaskString);
		}
	}
	return 0;
}

template <>
int32_t GpuSqlDispatcher::retCol<std::string>()
{
	auto colName = arguments.read<std::string>();
	auto alias = arguments.read<std::string>();

	int32_t loadFlag = loadCol<std::string>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "RetStringCol: " << colName << ", thread: " << dispatcherThreadId << std::endl;
	
	int32_t outSize;
	std::unique_ptr<std::string[]> outData;
	std::string nullMaskString = "";
	if (usingGroupBy)
	{
		if (isOverallLastBlock)
		{
			// Return key or value col (key if groupByColumns contains colName)
			auto col = findStringColumn(colName + (std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) != groupByColumns.end() ? KEYS_SUFFIX : ""));
			outSize = std::get<1>(col);
			outData = std::make_unique<std::string[]>(outSize);
			if(std::get<2>(col))
			{
				size_t bitMaskSize = (database->GetBlockSize() + sizeof(char)*8 - 1) / (sizeof(char)*8);
				std::unique_ptr<int8_t[]> nullMask = std::unique_ptr<int8_t[]>(new int8_t[bitMaskSize]);
				GPUReconstruct::ReconstructStringCol(outData.get(), &outSize,
					std::get<0>(col), nullptr, std::get<1>(col), nullMask.get(), std::get<2>(col));
					bitMaskSize = (outSize + sizeof(char)*8 - 1) / (sizeof(char)*8);
				nullMaskString = std::string(reinterpret_cast<char*>(nullMask.get()), bitMaskSize);
			}
			else
			{
				GPUReconstruct::ReconstructStringCol(outData.get(), &outSize,
					std::get<0>(col), nullptr, std::get<1>(col));
			}
		}
		else
		{
			return 0;
		}
	}
	else
	{
		auto col = findStringColumn(colName);
		outSize = std::get<1>(col);
		outData = std::make_unique<std::string[]>(outSize);
		if(std::get<2>(col))
		{
			size_t bitMaskSize = (database->GetBlockSize() + sizeof(char)*8 - 1) / (sizeof(char)*8);
			std::unique_ptr<int8_t[]> nullMask = std::unique_ptr<int8_t[]>(new int8_t[bitMaskSize]);
			GPUReconstruct::ReconstructStringCol(outData.get(), &outSize,
				std::get<0>(col), reinterpret_cast<int8_t*>(filter_), std::get<1>(col), nullMask.get(), std::get<2>(col));
				bitMaskSize = (outSize + sizeof(char)*8 - 1) / (sizeof(char)*8);
			nullMaskString = std::string(reinterpret_cast<char*>(nullMask.get()), bitMaskSize);
		}
		else
		{
			GPUReconstruct::ReconstructStringCol(outData.get(), &outSize,
				std::get<0>(col), reinterpret_cast<int8_t*>(filter_), std::get<1>(col));
		}
		std::cout << "dataSize: " << outSize << std::endl;
	}

	if (outSize > 0)
	{
		ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
		insertIntoPayload(payload, outData, outSize);
		MergePayloadToSelfResponse(alias, payload, nullMaskString);
	}
	return 0;
}


int32_t GpuSqlDispatcher::lockRegister()
{
	std::string reg = arguments.read<std::string>();
	std::cout << "Locked register: " << reg << std::endl;
	registerLockList.insert(reg);
	return 0;
}