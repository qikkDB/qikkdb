#include "GpuSqlDispatcherVMFunctions.h"
#include <array>
#include "../ParserExceptions.h"
#include "../../PointFactory.h"

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::retFunctions = { &GpuSqlDispatcher::retConst<int32_t>, &GpuSqlDispatcher::retConst<int64_t>, &GpuSqlDispatcher::retConst<float>, &GpuSqlDispatcher::retConst<double>, &GpuSqlDispatcher::retConst<ColmnarDB::Types::Point>, &GpuSqlDispatcher::retConst<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::retConst<std::string>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<int8_t>, &GpuSqlDispatcher::retCol<int32_t>, &GpuSqlDispatcher::retCol<int64_t>, &GpuSqlDispatcher::retCol<float>, &GpuSqlDispatcher::retCol<double>, &GpuSqlDispatcher::retCol<ColmnarDB::Types::Point>, &GpuSqlDispatcher::retCol<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::retCol<std::string>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<int8_t> };
GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::filFunction = &GpuSqlDispatcher::fil;
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
	if (allocatedPointers.find(colName) == allocatedPointers.end() && !colName.empty() && colName.front() != '$')
	{
		std::cout << "Load: " << colName << " " << typeid(ColmnarDB::Types::ComplexPolygon).name() << std::endl;

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

		auto col = dynamic_cast<const ColumnBase<ColmnarDB::Types::ComplexPolygon>*>(database->GetTables().at(table).GetColumns().at(column).get());
		auto block = dynamic_cast<BlockBase<ColmnarDB::Types::ComplexPolygon>*>(col->GetBlocksList()[blockIndex]);
		insertComplexPolygon(database->GetName(), colName,
			std::vector<ColmnarDB::Types::ComplexPolygon>(block->GetData(),
				block->GetData() + block->GetSize()),
			block->GetSize(), block->GetNullBitmask());
		noLoad = false;
	}
	return 0;
}

template <>
int32_t GpuSqlDispatcher::loadCol<ColmnarDB::Types::Point>(std::string& colName)
{
	if (allocatedPointers.find(colName) == allocatedPointers.end() && !colName.empty() && colName.front() != '$')
	{
		std::cout << "Load: " << colName << " " << typeid(ColmnarDB::Types::Point).name() << std::endl;

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

		auto col = dynamic_cast<const ColumnBase<ColmnarDB::Types::Point>*>(database->GetTables().at(table).GetColumns().at(column).get());
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
		addCachedRegister(colName, std::get<0>(cacheEntry), nativePoints.size(), block->GetNullBitmask());
		noLoad = false;
	}
	return 0;
}


template <>
int32_t GpuSqlDispatcher::loadCol<std::string>(std::string& colName)
{
	if (allocatedPointers.find(colName) == allocatedPointers.end() && !colName.empty() && colName.front() != '$')
	{
		std::cout << "Load: " << colName << " " << typeid(std::string).name() << std::endl;

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

		auto col = dynamic_cast<const ColumnBase<std::string>*>(database->GetTables().at(table).GetColumns().at(column).get());
		auto block = dynamic_cast<BlockBase<std::string>*>(col->GetBlocksList()[blockIndex]);

		insertString(database->GetName(), colName, std::vector<std::string>(block->GetData(), 
			block->GetData() + block->GetSize()),
			block->GetSize(), block->GetNullBitmask());
		noLoad = false;
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
		std::tuple<GPUMemory::GPUPolygon, int32_t> ACol = findComplexPolygon(col);
		int32_t outSize;
		GPUReconstruct::ReconstructPolyColToWKT(outData.get(), &outSize,
			std::get<0>(ACol), reinterpret_cast<int8_t*>(filter_), std::get<1>(ACol));

		std::cout << "dataSize: " << outSize << std::endl;
		ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
		insertIntoPayload(payload, outData, outSize);
		MergePayloadToSelfResponse(alias, payload);
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

		std::unique_ptr<NativeGeoPoint[]> outPoints(new NativeGeoPoint[database->GetBlockSize()]);
		int32_t outSize;
		//ToDo: Podmienene zapnut podla velkost buffera
		//GPUMemory::hostPin(outData.get(), database->GetBlockSize());
		PointerAllocation ACol = allocatedPointers.at(colName);
		GPUReconstruct::reconstructCol(outPoints.get(), &outSize, reinterpret_cast<NativeGeoPoint*>(ACol.gpuPtr), reinterpret_cast<int8_t*>(filter_), ACol.elementCount);
		//GPUMemory::hostUnregister(outData.get());

		std::unique_ptr<std::string[]> outData(new std::string[outSize]);

		for (int i = 0; i < outSize; i++)
		{
			outData[i] = PointFactory::WktFromPoint(outPoints[i]);
		}

		std::cout << "dataSize: " << outSize << std::endl;
		ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
		insertIntoPayload(payload, outData, outSize);
		MergePayloadToSelfResponse(alias, payload);
	}
	return 0;
}

template <>
int32_t GpuSqlDispatcher::retCol<std::string>()
{
	if (usingGroupBy)
	{
		throw RetStringGroupByException();
	}
	else
	{
		auto colName = arguments.read<std::string>();
		auto alias = arguments.read<std::string>();

		int32_t loadFlag = loadCol<std::string>(colName);
		if (loadFlag)
		{
			return loadFlag;
		}

		std::cout << "RetStringCol: " << colName << ", thread: " << dispatcherThreadId << std::endl;

		std::unique_ptr<std::string[]> outData(new std::string[database->GetBlockSize()]);
		std::tuple<GPUMemory::GPUString, int32_t> ACol = findStringColumn(colName);
		int32_t outSize;
		GPUReconstruct::ReconstructStringCol(outData.get(), &outSize,
			std::get<0>(ACol), reinterpret_cast<int8_t*>(filter_), std::get<1>(ACol));

		std::cout << "dataSize: " << outSize << std::endl;
		ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
		insertIntoPayload(payload, outData, outSize);
		MergePayloadToSelfResponse(alias, payload);
	}
	return 0;
}