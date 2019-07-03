//
// Created by Martin Staňo on 2019-01-15.
//

#include "GpuSqlDispatcher.h"
#include "../QueryEngine/Context.h"
#include "../Types/ComplexPolygon.pb.h"
#include "../Types/Point.pb.h"
#include "ParserExceptions.h"
#include "../QueryEngine/Context.h"
#include "../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../ComplexPolygonFactory.h"
#include "../StringFactory.h"
#include "../QueryEngine/GPUCore/GPUOrderBy.cuh"
#include "../Database.h"
#include "../Table.h"

int32_t GpuSqlDispatcher::groupByDoneCounter_ = 0;
std::mutex GpuSqlDispatcher::groupByMutex_;
std::condition_variable GpuSqlDispatcher::groupByCV_;
int32_t GpuSqlDispatcher::groupByDoneLimit_;
std::unordered_map<std::string, int32_t> GpuSqlDispatcher::linkTable;

#ifndef NDEBUG
void AssertDeviceMatchesCurrentThread(int dispatcherThreadId)
{
	int device;
	cudaGetDevice(&device);
	assert(device == dispatcherThreadId);
}
#endif

GpuSqlDispatcher::GpuSqlDispatcher(const std::shared_ptr<Database> &database, std::vector<std::unique_ptr<IGroupBy>>& groupByTables, int dispatcherThreadId) :
	database(database),
	blockIndex(dispatcherThreadId),
	instructionPointer(0),
	constPointCounter(0),
	constPolygonCounter(0),
	constStringCounter(0),
	filter_(0),
	usedRegisterMemory(0),
	maxRegisterMemory(0), // TODO value from config e.g.
	groupByTables(groupByTables),
	dispatcherThreadId(dispatcherThreadId),
	usingGroupBy(false),
	usingOrderBy(false),
	usingJoin(false),
	isLastBlockOfDevice(false),
	isOverallLastBlock(false),
	noLoad(true),
	joinIndices(nullptr),
	orderByTable(nullptr)
{
}

GpuSqlDispatcher::~GpuSqlDispatcher()
{
	if (dispatcherThreadId != -1)
	{
		Context& context = Context::getInstance();
		context.bindDeviceToContext(dispatcherThreadId);
#ifndef NDEBUG
		AssertDeviceMatchesCurrentThread(dispatcherThreadId);
#endif
		cleanUpGpuPointers();
	}
}


void GpuSqlDispatcher::copyExecutionDataTo(GpuSqlDispatcher & other)
{
	other.dispatcherFunctions = dispatcherFunctions;
	other.arguments = arguments;
}

void GpuSqlDispatcher::setJoinIndices(std::unordered_map<std::string, std::vector<std::vector<int32_t>>>* joinIdx)
{
	if (!joinIdx->empty())
	{
		joinIndices = joinIdx;
		usingJoin = true;
	}
}

/// Main execution loop of dispatcher
/// Iterates through all dispatcher functions in the operations array (filled from GpuSqlListener) and executes them
/// until running out of blocks
/// <param name="result">Response message to the SQL statement</param>
void GpuSqlDispatcher::execute(std::unique_ptr<google::protobuf::Message>& result, std::exception_ptr& exception)
{
	try
	{
		Context& context = Context::getInstance();
		context.bindDeviceToContext(dispatcherThreadId);
		int32_t err = 0;

		while (err == 0)
		{

			err = (this->*dispatcherFunctions[instructionPointer++])();
#ifndef NDEBUG
			printf("tid:%d ip: %d \n", dispatcherThreadId, instructionPointer - 1);
			AssertDeviceMatchesCurrentThread(dispatcherThreadId);
#endif
			if (err)
			{
				if (err == 1)
				{
					std::cout << "Out of blocks." << std::endl;
				}
				if (err == 2)
				{
					std::cout << "Show databases completed sucessfully" << std::endl;
				}
				if (err == 3)
				{
					std::cout << "Show tables completed sucessfully" << std::endl;
				}
				if (err == 4)
				{
					std::cout << "Show columns completed sucessfully" << std::endl;
				}
				if (err == 5)
				{
					std::cout << "Insert into completed sucessfully" << std::endl;
				}
				if (err == 6)
				{
					std::cout << "Create database completed sucessfully" << std::endl;
				}
				if (err == 7)
				{
					std::cout << "Drop database completed sucessfully" << std::endl;
				}
				if (err == 8)
				{
					std::cout << "Create table completed sucessfully" << std::endl;				
				}
				if (err == 9)
				{
					std::cout << "Drop table completed sucessfully" << std::endl;
				}
				if (err == 10)
				{
					std::cout << "Alter table completed sucessfully" << std::endl;
				}
				if (err == 11)
				{
					std::cout << "Create index completed sucessfully" << std::endl;
				}
				break;
			}
		}
		result = std::make_unique<ColmnarDB::NetworkClient::Message::QueryResponseMessage>(std::move(responseMessage));
	}
	catch (...)
	{
		exception = std::current_exception();
	}
}

const ColmnarDB::NetworkClient::Message::QueryResponseMessage &GpuSqlDispatcher::getQueryResponseMessage()
{
	return responseMessage;
}

void GpuSqlDispatcher::addRetFunction(DataType type)
{
    dispatcherFunctions.push_back(retFunctions[type]);
}

void GpuSqlDispatcher::addOrderByFunction(DataType type)
{
	dispatcherFunctions.push_back(orderByFunctions[type]);
}

void GpuSqlDispatcher::addOrderByReconstructFunction(DataType type)
{
	dispatcherFunctions.push_back(orderByReconstructFunctions[type]);
}

void GpuSqlDispatcher::addFreeOrderByTableFunction()
{
	dispatcherFunctions.push_back(freeOrderByTableFunction);
}

void GpuSqlDispatcher::addFilFunction()
{
    dispatcherFunctions.push_back(filFunction);
}

void GpuSqlDispatcher::addJmpInstruction()
{
	dispatcherFunctions.push_back(jmpFunction);
}

void GpuSqlDispatcher::addDoneFunction()
{
    dispatcherFunctions.push_back(doneFunction);
}

void GpuSqlDispatcher::addShowDatabasesFunction()
{
	dispatcherFunctions.push_back(showDatabasesFunction);
}

void GpuSqlDispatcher::addShowTablesFunction()
{
	dispatcherFunctions.push_back(showTablesFunction);
}

void GpuSqlDispatcher::addShowColumnsFunction()
{
	dispatcherFunctions.push_back(showColumnsFunction);
}

void GpuSqlDispatcher::addCreateDatabaseFunction()
{
	dispatcherFunctions.push_back(createDatabaseFunction);
}

void GpuSqlDispatcher::addDropDatabaseFunction()
{
	dispatcherFunctions.push_back(dropDatabaseFunction);
}

void GpuSqlDispatcher::addCreateTableFunction()
{
	dispatcherFunctions.push_back(createTableFunction);
}

void GpuSqlDispatcher::addDropTableFunction()
{
	dispatcherFunctions.push_back(dropTableFunction);
}

void GpuSqlDispatcher::addAlterTableFunction()
{
	dispatcherFunctions.push_back(alterTableFunction);
}

void GpuSqlDispatcher::addCreateIndexFunction()
{
	dispatcherFunctions.push_back(createIndexFunction);
}

void GpuSqlDispatcher::addInsertIntoFunction(DataType type)
{
	dispatcherFunctions.push_back(insertIntoFunctions[type]);
}

void GpuSqlDispatcher::addInsertIntoDoneFunction()
{
	dispatcherFunctions.push_back(insertIntoDoneFunction);
}

void GpuSqlDispatcher::addGreaterFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(greaterFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::addLessFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(lessFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::addGreaterEqualFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(greaterEqualFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::addLessEqualFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(lessEqualFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::addEqualFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(equalFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::addNotEqualFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(notEqualFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::addLogicalAndFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(logicalAndFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::addLogicalOrFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(logicalOrFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::addMulFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(mulFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::addDivFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(divFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::addAddFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(addFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}


void GpuSqlDispatcher::addSubFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(subFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addModFunction(DataType left, DataType right)
{
	dispatcherFunctions.push_back(modFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addBitwiseOrFunction(DataType left, DataType right)
{
	dispatcherFunctions.push_back(bitwiseOrFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addBitwiseAndFunction(DataType left, DataType right)
{
	dispatcherFunctions.push_back(bitwiseAndFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addBitwiseXorFunction(DataType left, DataType right)
{
	dispatcherFunctions.push_back(bitwiseXorFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addBitwiseLeftShiftFunction(DataType left, DataType right)
{
	dispatcherFunctions.push_back(bitwiseLeftShiftFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addBitwiseRightShiftFunction(DataType left, DataType right)
{
	dispatcherFunctions.push_back(bitwiseRightShiftFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addPointFunction(DataType left, DataType right)
{
	dispatcherFunctions.push_back(pointFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addLogarithmFunction(DataType number, DataType base)
{
	dispatcherFunctions.push_back(logarithmFunctions[DataType::DATA_TYPE_SIZE * number + base]);
}

void GpuSqlDispatcher::addArctangent2Function(DataType y, DataType x)
{
	dispatcherFunctions.push_back(arctangent2Functions[DataType::DATA_TYPE_SIZE * y + x]);
}

void GpuSqlDispatcher::addConcatFunction(DataType left, DataType right)
{
	dispatcherFunctions.push_back(concatFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addLeftFunction(DataType left, DataType right)
{
	dispatcherFunctions.push_back(leftFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addRightFunction(DataType left, DataType right)
{
	dispatcherFunctions.push_back(rightFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addPowerFunction(DataType base, DataType exponent)
{
	dispatcherFunctions.push_back(powerFunctions[DataType::DATA_TYPE_SIZE * base + exponent]);
}

void GpuSqlDispatcher::addRootFunction(DataType base, DataType exponent)
{
	dispatcherFunctions.push_back(rootFunctions[DataType::DATA_TYPE_SIZE * base + exponent]);
}

void GpuSqlDispatcher::addContainsFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(containsFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addIntersectFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(intersectFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addUnionFunction(DataType left, DataType right)
{
    dispatcherFunctions.push_back(unionFunctions[DataType::DATA_TYPE_SIZE * left + right]);
}

void GpuSqlDispatcher::addLogicalNotFunction(DataType type)
{
    dispatcherFunctions.push_back(logicalNotFunctions[type]);
}

void GpuSqlDispatcher::addMinusFunction(DataType type)
{
    dispatcherFunctions.push_back(minusFunctions[type]);
}

void GpuSqlDispatcher::addYearFunction(DataType type)
{
	dispatcherFunctions.push_back(yearFunctions[type]);
}

void GpuSqlDispatcher::addMonthFunction(DataType type)
{
	dispatcherFunctions.push_back(monthFunctions[type]);
}

void GpuSqlDispatcher::addDayFunction(DataType type)
{
	dispatcherFunctions.push_back(dayFunctions[type]);
}

void GpuSqlDispatcher::addHourFunction(DataType type)
{
	dispatcherFunctions.push_back(hourFunctions[type]);
}

void GpuSqlDispatcher::addMinuteFunction(DataType type)
{
	dispatcherFunctions.push_back(minuteFunctions[type]);
}

void GpuSqlDispatcher::addSecondFunction(DataType type)
{
	dispatcherFunctions.push_back(secondFunctions[type]);
}

void GpuSqlDispatcher::addAbsoluteFunction(DataType type)
{
	dispatcherFunctions.push_back(absoluteFunctions[type]);
}

void GpuSqlDispatcher::addSineFunction(DataType type)
{
	dispatcherFunctions.push_back(sineFunctions[type]);
}

void GpuSqlDispatcher::addCosineFunction(DataType type)
{
	dispatcherFunctions.push_back(cosineFunctions[type]);
}

void GpuSqlDispatcher::addTangentFunction(DataType type)
{
	dispatcherFunctions.push_back(tangentFunctions[type]);
}

void GpuSqlDispatcher::addCotangentFunction(DataType type)
{
	dispatcherFunctions.push_back(cotangentFunctions[type]);
}

void GpuSqlDispatcher::addArcsineFunction(DataType type)
{
	dispatcherFunctions.push_back(arcsineFunctions[type]);
}

void GpuSqlDispatcher::addArccosineFunction(DataType type)
{
	dispatcherFunctions.push_back(arccosineFunctions[type]);
}

void GpuSqlDispatcher::addArctangentFunction(DataType type)
{
	dispatcherFunctions.push_back(arctangentFunctions[type]);
}

void GpuSqlDispatcher::addLogarithm10Function(DataType type)
{
	dispatcherFunctions.push_back(logarithm10Functions[type]);
}

void GpuSqlDispatcher::addLogarithmNaturalFunction(DataType type)
{
	dispatcherFunctions.push_back(logarithmNaturalFunctions[type]);
}

void GpuSqlDispatcher::addExponentialFunction(DataType type)
{
	dispatcherFunctions.push_back(exponentialFunctions[type]);
}

void GpuSqlDispatcher::addSquareFunction(DataType type)
{
	dispatcherFunctions.push_back(squareFunctions[type]);
}

void GpuSqlDispatcher::addSquareRootFunction(DataType type)
{
	dispatcherFunctions.push_back(squareRootFunctions[type]);
}

void GpuSqlDispatcher::addSignFunction(DataType type)
{
	dispatcherFunctions.push_back(signFunctions[type]);
}

void GpuSqlDispatcher::addRoundFunction(DataType type)
{
	dispatcherFunctions.push_back(roundFunctions[type]);
}

void GpuSqlDispatcher::addFloorFunction(DataType type)
{
	dispatcherFunctions.push_back(floorFunctions[type]);
}

void GpuSqlDispatcher::addCeilFunction(DataType type)
{
	dispatcherFunctions.push_back(ceilFunctions[type]);
}

void GpuSqlDispatcher::addLtrimFunction(DataType type)
{
	dispatcherFunctions.push_back(ltrimFunctions[type]);
}

void GpuSqlDispatcher::addRtrimFunction(DataType type)
{
	dispatcherFunctions.push_back(rtrimFunctions[type]);
}

void GpuSqlDispatcher::addLowerFunction(DataType type)
{
	dispatcherFunctions.push_back(lowerFunctions[type]);
}

void GpuSqlDispatcher::addUpperFunction(DataType type)
{
	dispatcherFunctions.push_back(upperFunctions[type]);
}

void GpuSqlDispatcher::addReverseFunction(DataType type)
{
	dispatcherFunctions.push_back(reverseFunctions[type]);
}

void GpuSqlDispatcher::addLenFunction(DataType type)
{
	dispatcherFunctions.push_back(lenFunctions[type]);
}

void GpuSqlDispatcher::addMinFunction(DataType key, DataType value, bool usingGroupBy)
{
    dispatcherFunctions.push_back((usingGroupBy ? minGroupByFunctions : minAggregationFunctions)
		[DataType::DATA_TYPE_SIZE * key + value]);
}

void GpuSqlDispatcher::addMaxFunction(DataType key, DataType value, bool usingGroupBy)
{
    dispatcherFunctions.push_back((usingGroupBy ? maxGroupByFunctions : maxAggregationFunctions)
		[DataType::DATA_TYPE_SIZE * key + value]);
}

void GpuSqlDispatcher::addSumFunction(DataType key, DataType value, bool usingGroupBy)
{
    dispatcherFunctions.push_back((usingGroupBy ? sumGroupByFunctions : sumAggregationFunctions)
		[DataType::DATA_TYPE_SIZE * key + value]);
}

void GpuSqlDispatcher::addCountFunction(DataType key, DataType value, bool usingGroupBy)
{
    dispatcherFunctions.push_back((usingGroupBy ? countGroupByFunctions : countAggregationFunctions)
		[DataType::DATA_TYPE_SIZE * key + value]);
}

void GpuSqlDispatcher::addAvgFunction(DataType key, DataType value, bool usingGroupBy)
{
    dispatcherFunctions.push_back((usingGroupBy ? avgGroupByFunctions : avgAggregationFunctions)
		[DataType::DATA_TYPE_SIZE * key + value]);
}

void GpuSqlDispatcher::addGroupByFunction(DataType type)
{
    dispatcherFunctions.push_back(groupByFunctions[type]);
}

void GpuSqlDispatcher::addBetweenFunction(DataType op1, DataType op2, DataType op3)
{
    //TODO: Between
}

std::string GpuSqlDispatcher::getAllocatedRegisterName(const std::string & reg)
{
	if (usingJoin && reg.front() != '$')
	{
		std::string joinReg = reg + "_join";
		for (auto& joinTable : *joinIndices)
		{
			joinReg += "_" + joinTable.first;
		}
		return joinReg;
	}
	return reg;
}

void GpuSqlDispatcher::fillPolygonRegister(GPUMemory::GPUPolygon& polygonColumn, const std::string & reg, int32_t size, bool useCache)
{
	allocatedPointers.insert({ reg + "_polyPoints", std::make_tuple(reinterpret_cast<uintptr_t>(polygonColumn.polyPoints), size, !useCache) });
	allocatedPointers.insert({ reg + "_pointIdx", std::make_tuple(reinterpret_cast<uintptr_t>(polygonColumn.pointIdx), size, !useCache) });
	allocatedPointers.insert({ reg + "_pointCount", std::make_tuple(reinterpret_cast<uintptr_t>(polygonColumn.pointCount), size, !useCache) });
	allocatedPointers.insert({ reg + "_polyIdx", std::make_tuple(reinterpret_cast<uintptr_t>(polygonColumn.polyIdx), size, !useCache) });
	allocatedPointers.insert({ reg + "_polyCount", std::make_tuple(reinterpret_cast<uintptr_t>(polygonColumn.polyCount), size, !useCache) });
}

void GpuSqlDispatcher::fillStringRegister(GPUMemory::GPUString & stringColumn, const std::string & reg, int32_t size, bool useCache)
{
	allocatedPointers.insert({ reg + "_stringIndices", std::make_tuple(reinterpret_cast<uintptr_t>(stringColumn.stringIndices), size, !useCache) });
	allocatedPointers.insert({ reg + "_allChars", std::make_tuple(reinterpret_cast<uintptr_t>(stringColumn.allChars), size, !useCache) });
}

GPUMemory::GPUPolygon GpuSqlDispatcher::insertComplexPolygon(const std::string& databaseName, const std::string& colName, const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons, int32_t size, bool useCache)
{
	if (useCache)
	{
		if (Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_polyPoints", blockIndex) &&
			Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_pointIdx", blockIndex) &&
			Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_pointCount", blockIndex) &&
			Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_polyIdx", blockIndex) &&
			Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_polyCount", blockIndex))
		{
			GPUMemoryCache& cache = Context::getInstance().getCacheForCurrentDevice();
			GPUMemory::GPUPolygon polygon;
			polygon.polyPoints = std::get<0>(cache.getColumn<NativeGeoPoint>(databaseName, colName + "_polyPoints", blockIndex, size));
			polygon.pointIdx = std::get<0>(cache.getColumn<int32_t>(databaseName, colName + "_pointCount", blockIndex, size));
			polygon.pointCount = std::get<0>(cache.getColumn<int32_t>(databaseName, colName + "_pointCount", blockIndex, size));
			polygon.polyIdx = std::get<0>(cache.getColumn<int32_t>(databaseName, colName + "_polyIdx", blockIndex, size));
			polygon.polyCount = std::get<0>(cache.getColumn<int32_t>(databaseName, colName + "_polyCount", blockIndex, size));
			fillPolygonRegister(polygon, colName, size, useCache);
			return polygon;
		}
		else
		{
			GPUMemory::GPUPolygon polygon = ComplexPolygonFactory::PrepareGPUPolygon(polygons, databaseName, colName, blockIndex);
			fillPolygonRegister(polygon, colName, size, useCache);
			return polygon;
		}
	}
	else
	{
		GPUMemory::GPUPolygon polygon = ComplexPolygonFactory::PrepareGPUPolygon(polygons);
		fillPolygonRegister(polygon, colName, size, useCache);
		return polygon;
	}
}

GPUMemory::GPUString GpuSqlDispatcher::insertString(const std::string& databaseName, const std::string& colName, const std::vector<std::string>& strings, int32_t size, bool useCache)
{
	if (useCache)
	{
		if (Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_stringIndices", blockIndex) &&
			Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_allChars", blockIndex))
		{
			GPUMemoryCache& cache = Context::getInstance().getCacheForCurrentDevice();
			GPUMemory::GPUString gpuString;
			gpuString.stringIndices = std::get<0>(cache.getColumn<int64_t>(databaseName, colName + "_stringIndices", blockIndex, size));
			gpuString.allChars = std::get<0>(cache.getColumn<char>(databaseName, colName + "_allChars", blockIndex, size));
			fillStringRegister(gpuString, colName, size, useCache);
			return gpuString;
		}
		else
		{
			GPUMemory::GPUString gpuString = StringFactory::PrepareGPUString(strings, databaseName, colName, blockIndex);
			fillStringRegister(gpuString, colName, size, useCache);
			return gpuString;
		}
	}
	else
	{
		GPUMemory::GPUString gpuString = StringFactory::PrepareGPUString(strings);
		fillStringRegister(gpuString, colName, size, useCache);
		return gpuString;
	}
}

std::tuple<GPUMemory::GPUPolygon, int32_t> GpuSqlDispatcher::findComplexPolygon(std::string colName)
{
	GPUMemory::GPUPolygon polygon;
	int32_t size = std::get<1>(allocatedPointers.at(colName + "_polyPoints"));

	polygon.polyPoints = reinterpret_cast<NativeGeoPoint*>(std::get<0>(allocatedPointers.at(colName + "_polyPoints")));
	polygon.pointIdx = reinterpret_cast<int32_t*>(std::get<0>(allocatedPointers.at(colName + "_pointIdx")));
	polygon.pointCount = reinterpret_cast<int32_t*>(std::get<0>(allocatedPointers.at(colName + "_pointCount")));
	polygon.polyIdx = reinterpret_cast<int32_t*>(std::get<0>(allocatedPointers.at(colName + "_polyIdx")));
	polygon.polyCount = reinterpret_cast<int32_t*>(std::get<0>(allocatedPointers.at(colName + "_polyCount")));

	return std::make_tuple(polygon, size);
}

std::tuple<GPUMemory::GPUString, int32_t> GpuSqlDispatcher::findStringColumn(const std::string & colName)
{
	GPUMemory::GPUString gpuString;
	int32_t size = std::get<1>(allocatedPointers.at(colName + "_stringIndices"));
	gpuString.stringIndices = reinterpret_cast<int64_t*>(std::get<0>(allocatedPointers.at(colName + "_stringIndices")));
	gpuString.allChars = reinterpret_cast<char*>(std::get<0>(allocatedPointers.at(colName + "_allChars")));
	return std::make_tuple(gpuString, size);
}

NativeGeoPoint* GpuSqlDispatcher::insertConstPointGpu(ColmnarDB::Types::Point& point)
{
	NativeGeoPoint nativePoint;
	nativePoint.latitude = point.geopoint().latitude();
	nativePoint.longitude = point.geopoint().longitude();

	NativeGeoPoint *gpuPointer = allocateRegister<NativeGeoPoint>("constPoint" + std::to_string(constPointCounter), 1);
	constPointCounter++;

	GPUMemory::copyHostToDevice(gpuPointer, reinterpret_cast<NativeGeoPoint*>(&nativePoint), 1);
	return gpuPointer;
}

// TODO change to return GPUMemory::GPUPolygon struct
/// Copy polygon column to GPU memory - create polygon gpu representation temporary buffers from protobuf polygon object
/// <param name="polygon">Polygon object (protobuf type)</param>
/// <returns>Struct with GPU pointers to start of polygon arrays</returns>
GPUMemory::GPUPolygon GpuSqlDispatcher::insertConstPolygonGpu(ColmnarDB::Types::ComplexPolygon& polygon)
{
	std::string name = "constPolygon" + std::to_string(constPolygonCounter);
	constPolygonCounter++;
	return insertComplexPolygon(database->GetName(), name, { polygon }, 1);
}

GPUMemory::GPUString GpuSqlDispatcher::insertConstStringGpu(const std::string& str)
{
	std::string name = "constString" + std::to_string(constStringCounter);
	constStringCounter++;
	return insertString(database->GetName(), name, { str }, 1);
}


/// Clears all allocated buffers
/// Resets memory stream reading index to prepare for execution on the next block of data
void GpuSqlDispatcher::cleanUpGpuPointers()
{
	usingGroupBy = false;
	arguments.reset();
	for (auto& ptr : allocatedPointers)
	{
		if (std::get<0>(ptr.second) != 0 && std::get<2>(ptr.second))
		{
			GPUMemory::free(reinterpret_cast<void*>(std::get<0>(ptr.second)));
		}
	}
	usedRegisterMemory = 0;
	allocatedPointers.clear();
}


/// Implementation of FIL operation
/// Marks WHERE clause result register as the filtering register
/// <returns name="statusCode">Finish status code of the operation</returns>
int32_t GpuSqlDispatcher::fil()
{
    auto reg = arguments.read<std::string>();
    std::cout << "Filter: " << reg << std::endl;
	filter_ = std::get<0>(allocatedPointers.at(reg));
	return 0;
}


/// Implementation of JMP operation
/// Determines next block index to process by this instance of dispatcher based on CUDA device count
/// <returns name="statusCode">Finish status code of the operation</returns>
int32_t GpuSqlDispatcher::jmp()
{
	Context& context = Context::getInstance();

	if (noLoad)
	{
		cleanUpGpuPointers();
		return 0;
	}

	if (!isLastBlockOfDevice)
	{
		blockIndex += context.getDeviceCount();
		instructionPointer = 0;
		cleanUpGpuPointers();
		return 0;
	}

	std::cout << "Jump" << std::endl;
	return 0;
}


/// Implementation of DONE operation
/// Clears all allocated temporary result buffers
/// <returns name="statusCode">Finish status code of the operation</returns>
int32_t GpuSqlDispatcher::done()
{
	cleanUpGpuPointers();
	std::cout << "Done" << std::endl;
	return 1;
}

/// Implementation of SHOW DATABASES operation
/// Inserts database names to the response message
/// <returns name="statusCode">Finish status code of the operation</returns>
int32_t GpuSqlDispatcher::showDatabases()
{
	auto databases_map = Database::GetDatabaseNames();
	std::unique_ptr<std::string[]> outData(new std::string[databases_map.size()]);
	
	int i = 0;
	for (auto& database : databases_map) {
		outData[i++] = database;
	}
	
	ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
	insertIntoPayload(payload, outData, databases_map.size());
	MergePayloadToSelfResponse("Databases", payload);

	return 2;
}


/// Implementation of SHOW TABLES operation
/// Inserts table names to the response message
/// <returns name="statusCode">Finish status code of the operation</returns>
int32_t GpuSqlDispatcher::showTables()
{
	std::string db = arguments.read<std::string>();
	std::shared_ptr<Database> database = Database::GetDatabaseByName(db);

	std::unique_ptr<std::string[]> outData(new std::string[database->GetTables().size()]);
	auto& tables_map = database->GetTables();

	int i = 0;
	for (auto& tableName : tables_map) {
		outData[i++] = tableName.first;
	}

	ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
	insertIntoPayload(payload, outData, tables_map.size());
	MergePayloadToSelfResponse(db, payload);

	return 3;
}

/// Implementation of SHOW COLUMN operation
/// Inserts column names and their types to the response message
/// <returns name="statusCode">Finish status code of the operation</returns>
int32_t GpuSqlDispatcher::showColumns()
{
	std::string db = arguments.read<std::string>();
	std::string tab = arguments.read<std::string>();

	std::shared_ptr<Database> database = Database::GetDatabaseByName(db);
	auto& table = database->GetTables().at(tab);

	auto& columns_map = table.GetColumns();
	//std::vector<std::string> columns;
	std::unique_ptr<std::string[]> outDataName(new std::string[table.GetColumns().size()]);
	std::unique_ptr<std::string[]> outDataType(new std::string[table.GetColumns().size()]);

	int i = 0;
	for (auto& column : columns_map) {
		outDataName[i] = column.first;
		outDataType[i] = std::to_string(column.second.get()->GetColumnType());
		i++;
	}

	ColmnarDB::NetworkClient::Message::QueryResponsePayload payloadName;
	ColmnarDB::NetworkClient::Message::QueryResponsePayload payloadType;
	insertIntoPayload(payloadName, outDataName, columns_map.size());
	insertIntoPayload(payloadType, outDataType, columns_map.size());
	MergePayloadToSelfResponse(tab + "_columns", payloadName);
	MergePayloadToSelfResponse(tab + "_types", payloadType);
	return 4;
}

int32_t GpuSqlDispatcher::createDatabase()
{
	std::string newDbName = arguments.read<std::string>();
	int32_t newDbBlockSize = arguments.read<int32_t>();
	std::shared_ptr<Database> newDb = std::make_shared<Database>(newDbName.c_str(), newDbBlockSize);
	Database::AddToInMemoryDatabaseList(newDb);
	return 6;
}

int32_t GpuSqlDispatcher::dropDatabase()
{
	std::string dbName = arguments.read<std::string>();
	Database::RemoveFromInMemoryDatabaseList(dbName.c_str());
	database->DeleteDatabaseFromDisk();
	return 7;
}

int32_t GpuSqlDispatcher::createTable()
{
	std::unordered_map<std::string, DataType> newColumns;
	std::unordered_map<std::string, std::unordered_set<std::string>> newIndices;

	std::string newTableName = arguments.read<std::string>();

	int32_t newColumnsCount = arguments.read<int32_t>();
	for (int32_t i = 0; i < newColumnsCount; i++)
	{
		std::string newColumnName = arguments.read<std::string>();
		int32_t newColumnDataType = arguments.read<int32_t>();
		newColumns.insert({ newColumnName, static_cast<DataType>(newColumnDataType) });
	}

	std::unordered_set<std::string> allIndexColumns;

	int32_t newIndexCount = arguments.read<int32_t>();
	for (int32_t i = 0; i < newIndexCount; i++)
	{
		std::string newIndexName = arguments.read<std::string>();
		int32_t newIndexColumnCount = arguments.read<int32_t>();
		std::unordered_set<std::string> newIndexColumns;

		for (int32_t j = 0; j < newIndexColumnCount; j++)
		{
			std::string newIndexColumn = arguments.read<std::string>();
			newIndexColumns.insert(newIndexColumn);
			allIndexColumns.insert(newIndexColumn);
		}
		newIndices.insert({ newIndexName, newIndexColumns });
	}

	std::vector<std::string> allIndexColumnsVector(allIndexColumns.begin(), allIndexColumns.end());
	database->CreateTable(newColumns, newTableName.c_str()).SetSortingColumns(allIndexColumnsVector);
	return 8;
}

int32_t GpuSqlDispatcher::dropTable()
{
	std::string tableName = arguments.read<std::string>();
	database->GetTables().erase(tableName);
	database->DeleteTableFromDisk(tableName.c_str());
	return 9;
}

int32_t GpuSqlDispatcher::alterTable()
{
	std::string tableName = arguments.read<std::string>();

	int32_t addColumnsCount = arguments.read<int32_t>();
	for (int32_t i = 0; i < addColumnsCount; i++)
	{
		std::string addColumnName = arguments.read<std::string>();
		int32_t addColumnDataType = arguments.read<int32_t>();
		database->GetTables().at(tableName).CreateColumn(addColumnName.c_str(), static_cast<DataType>(addColumnDataType));
		int64_t tableSize = database->GetTables().at(tableName).GetSize();
		database->GetTables().at(tableName).GetColumns().at(addColumnName)->InsertNullData(tableSize);
	}

	int32_t dropColumnsCount = arguments.read<int32_t>();
	for (int32_t i = 0; i < dropColumnsCount; i++)
	{
		std::string dropColumnName = arguments.read<std::string>();
		database->GetTables().at(tableName).EraseColumn(dropColumnName);
		database->DeleteColumnFromDisk(tableName.c_str(), dropColumnName.c_str());
	}
	return 10;
}

int32_t GpuSqlDispatcher::createIndex()
{
	std::string	indexName = arguments.read<std::string>();
	std::string tableName = arguments.read<std::string>();
	std::unordered_set<std::string> indexColumns;
	std::unordered_set<std::string> sortingColumns;

	int32_t indexColumnCount = arguments.read<int32_t>();
	for (int i = 0; i < indexColumnCount; i++)
	{
		std::string indexColumn = arguments.read<std::string>();
		indexColumns.insert(indexColumn);
		sortingColumns.insert(indexColumn);
	}

	for (auto& column : database->GetTables().at(tableName).GetSortingColumns())
	{
		sortingColumns.insert(column);
	}

	std::vector<std::string> sortingColumnsVector(sortingColumns.begin(), sortingColumns.end());
	database->GetTables().at(tableName).SetSortingColumns(sortingColumnsVector);

	return 11;
}


void GpuSqlDispatcher::insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<int32_t[]> &data, int32_t dataSize)
{
	for (int i = 0; i < dataSize; i++)
	{
		payload.mutable_intpayload()->add_intdata(data[i]);
	}
}

void GpuSqlDispatcher::insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<int64_t[]> &data, int32_t dataSize)
{
	for (int i = 0; i < dataSize; i++)
	{
		payload.mutable_int64payload()->add_int64data(data[i]);
	}
}

void GpuSqlDispatcher::insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<float[]> &data, int32_t dataSize)
{
	for (int i = 0; i < dataSize; i++)
	{
		payload.mutable_floatpayload()->add_floatdata(data[i]);
	}
}

void GpuSqlDispatcher::insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<double[]> &data, int32_t dataSize)
{
	for (int i = 0; i < dataSize; i++)
	{
		payload.mutable_doublepayload()->add_doubledata(data[i]);
	}
}

void GpuSqlDispatcher::insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<std::string[]> &data, int32_t dataSize)
{
	for (int i = 0; i < dataSize; i++)
	{
		payload.mutable_stringpayload()->add_stringdata(data[i]);
	}
}

void GpuSqlDispatcher::MergePayload(const std::string &trimmedKey, ColmnarDB::NetworkClient::Message::QueryResponseMessage * responseMessage,
	ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload)
{
	// If there is payload with new key
	if (responseMessage->payloads().find(trimmedKey) == responseMessage->payloads().end())
	{
		responseMessage->mutable_payloads()->insert({ trimmedKey, payload });
	}
	else    // If there is payload with existing key, merge or aggregate according to key
	{
		// Find index of parenthesis (for finding out if it is aggregation function)
		size_t keyParensIndex = trimmedKey.find('(');

		bool aggregationOperationFound = false;
		// If no function is used
		if (keyParensIndex == std::string::npos)
		{
			aggregationOperationFound = false;
		}
		else
		{
			// Get operation name
			std::string operation = trimmedKey.substr(0, keyParensIndex);
			// To upper case
			for (auto &c : operation)
			{
				c = toupper(c);
			}
			// Switch according to data type of payload (=column)
			switch (payload.payload_case())
			{
			case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kIntPayload:
			{
				std::pair<bool, int32_t> result =
					AggregateOnCPU<int32_t>(operation, payload.intpayload().intdata()[0],
						responseMessage->mutable_payloads()->at(trimmedKey).intpayload().intdata()[0]);
				aggregationOperationFound = result.first;
				if (aggregationOperationFound)
				{
					responseMessage->mutable_payloads()->at(trimmedKey).mutable_intpayload()->set_intdata(0, result.second);
				}
				break;
			}
			case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kInt64Payload:
			{
				std::pair<bool, int64_t> result =
					AggregateOnCPU<int64_t>(operation, payload.int64payload().int64data()[0],
						responseMessage->payloads().at(trimmedKey).int64payload().int64data()[0]);
				aggregationOperationFound = result.first;
				if (aggregationOperationFound)
				{
					responseMessage->mutable_payloads()->at(trimmedKey).mutable_int64payload()->set_int64data(0, result.second);
				}
				break;
			}
			case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kFloatPayload:
			{
				std::pair<bool, float> result =
					AggregateOnCPU<float>(operation, payload.floatpayload().floatdata()[0],
						responseMessage->mutable_payloads()->at(trimmedKey).floatpayload().floatdata()[0]);
				aggregationOperationFound = result.first;
				if (aggregationOperationFound)
				{
					responseMessage->mutable_payloads()->at(trimmedKey).mutable_floatpayload()->set_floatdata(0, result.second);
				}
				break;
			}
			case ColmnarDB::NetworkClient::Message::QueryResponsePayload::PayloadCase::kDoublePayload:
			{
				std::pair<bool, double> result =
					AggregateOnCPU<double>(operation, payload.doublepayload().doubledata()[0],
						responseMessage->mutable_payloads()->at(trimmedKey).doublepayload().doubledata()[0]);
				aggregationOperationFound = result.first;
				if (aggregationOperationFound)
				{
					responseMessage->mutable_payloads()->at(trimmedKey).mutable_doublepayload()->set_doubledata(0, result.second);
				}
				break;
			}
			default:
				// This case is taken even without aggregation functions, because Points are considered functions 
				// for some reason
				if(aggregationOperationFound)
				{
					throw std::out_of_range("Unsupported aggregation type result");
				}
				break;
			}
		}

		if (!aggregationOperationFound)
		{
			responseMessage->mutable_payloads()->at(trimmedKey).MergeFrom(payload);
		}
	}
}

void GpuSqlDispatcher::MergePayloadToSelfResponse(const std::string& key, ColmnarDB::NetworkClient::Message::QueryResponsePayload& payload)
{
	std::string trimmedKey = key.substr(0, std::string::npos);
	if (!key.empty() && key.front() == '$')
	{
		trimmedKey = key.substr(1, std::string::npos);
	}
	MergePayload(trimmedKey, &responseMessage, payload);
}

bool GpuSqlDispatcher::isRegisterAllocated(std::string & reg)
{
	return allocatedPointers.find(reg) != allocatedPointers.end();
}

std::pair<std::string, std::string> GpuSqlDispatcher::splitColumnName(const std::string& colName)
{
	const size_t splitIdx = colName.find(".");
	const std::string table = colName.substr(0, splitIdx);
	const std::string column = colName.substr(splitIdx + 1);
	return {table, column};
}
