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
#include "../QueryEngine/GPUCore/GPUWhereInterpreter.cuh"
#include "../ComplexPolygonFactory.h"

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
	filter_(0),
	symbolTable(nullptr),
	usedRegisterMemory(0),
	maxRegisterMemory(0), // TODO value from config e.g.
	groupByTables(groupByTables),
	dispatcherThreadId(dispatcherThreadId),
	usingGroupBy(false),
	isLastBlockOfDevice(false),
	isOverallLastBlock(false),
	noLoad(true)
{
	dispatcherFunctions.push_back(ldByLinkFunction);
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
	other.linkTable = linkTable;
	other.symbolTable = std::unique_ptr<void*[]>(new void*[linkTable.size()]);
	other.gpuOpCodes = gpuOpCodes;
}

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

void GpuSqlDispatcher::addDispatcherBinaryFunction(DispatcherFunction fun, DataType left, DataType right)
{
	switch (fun)
	{
	case DispatcherFunction::GT_FUNC:
		dispatcherFunctions.push_back(greaterFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::LT_FUNC:
		dispatcherFunctions.push_back(lessFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::GTEQ_FUNC:
		dispatcherFunctions.push_back(greaterEqualFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::LTEQ_FUNC:
		dispatcherFunctions.push_back(lessEqualFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::EQ_FUNC:
		dispatcherFunctions.push_back(equalFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::NEQ_FUNC:
		dispatcherFunctions.push_back(notEqualFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::AND_FUNC:
		dispatcherFunctions.push_back(logicalAndFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::OR_FUNC:
		dispatcherFunctions.push_back(logicalOrFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::MUL_FUNC:
		dispatcherFunctions.push_back(mulFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::DIV_FUNC:
		dispatcherFunctions.push_back(divFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::ADD_FUNC:
		dispatcherFunctions.push_back(addFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::SUB_FUNC:
		dispatcherFunctions.push_back(subFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::MOD_FUNC:
		dispatcherFunctions.push_back(modFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::BIT_OR_FUNC:
		dispatcherFunctions.push_back(bitwiseOrFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::BIT_AND_FUNC:
		dispatcherFunctions.push_back(bitwiseAndFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::BIT_XOR_FUNC:
		dispatcherFunctions.push_back(bitwiseXorFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::LEFT_SHIFT_FUNC:
		dispatcherFunctions.push_back(bitwiseLeftShiftFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::RIGHT_SHIFT_FUNC:
		dispatcherFunctions.push_back(bitwiseRightShiftFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::POINT_FUNC:
		dispatcherFunctions.push_back(pointFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::GEO_CONTAINS_FUNC:
		dispatcherFunctions.push_back(containsFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::GEO_INTERSECT_FUNC:
		dispatcherFunctions.push_back(intersectFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::GEO_UNION_FUNC:
		dispatcherFunctions.push_back(unionFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::LOG_BIN_FUNC:
		dispatcherFunctions.push_back(logarithmFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::POW_BIN_FUNC:
		dispatcherFunctions.push_back(powerFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::ROOT_BIN_FUNC:
		dispatcherFunctions.push_back(rootFunctions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	case DispatcherFunction::ATAN2_FUNC:
		dispatcherFunctions.push_back(arctangent2Functions[DataType::DATA_TYPE_SIZE * left + right]);
		break;
	default:
		break;
	}
}

void GpuSqlDispatcher::addDispatcherUnaryFunction(DispatcherFunction fun, DataType type)
{
	switch (fun)
	{
	case DispatcherFunction::NOT_FUNC:
		dispatcherFunctions.push_back(logicalNotFunctions[type]);
		break;
	case DispatcherFunction::MINUS_FUNC:
		dispatcherFunctions.push_back(minusFunctions[type]);
		break;
	case DispatcherFunction::YEAR_FUNC:
		dispatcherFunctions.push_back(yearFunctions[type]);
		break;
	case DispatcherFunction::MONTH_FUNC:
		dispatcherFunctions.push_back(monthFunctions[type]);
		break;
	case DispatcherFunction::DAY_FUNC:
		dispatcherFunctions.push_back(dayFunctions[type]);
		break;
	case DispatcherFunction::HOUR_FUNC:
		dispatcherFunctions.push_back(hourFunctions[type]);
		break;
	case DispatcherFunction::MINUTE_FUNC:
		dispatcherFunctions.push_back(minuteFunctions[type]);
		break;
	case DispatcherFunction::SECOND_FUNC:
		dispatcherFunctions.push_back(secondFunctions[type]);
		break;
	case DispatcherFunction::ABS_FUNC:
		dispatcherFunctions.push_back(absoluteFunctions[type]);
		break;
	case DispatcherFunction::SIN_FUNC:
		dispatcherFunctions.push_back(sineFunctions[type]);
		break;
	case DispatcherFunction::COS_FUNC:
		dispatcherFunctions.push_back(cosineFunctions[type]);
		break;
	case DispatcherFunction::TAN_FUNC:
		dispatcherFunctions.push_back(tangentFunctions[type]);
		break;
	case DispatcherFunction::COT_FUNC:
		dispatcherFunctions.push_back(cotangentFunctions[type]);
		break;
	case DispatcherFunction::ASIN_FUNC:
		dispatcherFunctions.push_back(arcsineFunctions[type]);
		break;
	case DispatcherFunction::ACOS_FUNC:
		dispatcherFunctions.push_back(arccosineFunctions[type]);
		break;
	case DispatcherFunction::ATAN_FUNC:
		dispatcherFunctions.push_back(arctangentFunctions[type]);
		break;
	case DispatcherFunction::LOG10_FUNC:
		dispatcherFunctions.push_back(logarithm10Functions[type]);
		break;
	case DispatcherFunction::LOG_FUNC:
		dispatcherFunctions.push_back(logarithmNaturalFunctions[type]);
		break;
	case DispatcherFunction::EXP_FUNC:
		dispatcherFunctions.push_back(exponentialFunctions[type]);
		break;
	case DispatcherFunction::SQRT_FUNC:
		dispatcherFunctions.push_back(squareRootFunctions[type]);
		break;
	case DispatcherFunction::SQUARE_FUNC:
		dispatcherFunctions.push_back(squareFunctions[type]);
		break;
	case DispatcherFunction::SIGN_FUNC:
		dispatcherFunctions.push_back(signFunctions[type]);
		break;
	case DispatcherFunction::ROUND_FUNC:
		dispatcherFunctions.push_back(roundFunctions[type]);
		break;
	case DispatcherFunction::FLOOR_FUNC:
		dispatcherFunctions.push_back(floorFunctions[type]);
		break;
	case DispatcherFunction::CEIL_FUNC:
		dispatcherFunctions.push_back(ceilFunctions[type]);
		break;
	default:
		break;
	}
}

void GpuSqlDispatcher::addRetFunction(DataType type)
{
    dispatcherFunctions.push_back(retFunctions[type]);
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

void GpuSqlDispatcher::addInsertIntoFunction(DataType type)
{
	dispatcherFunctions.push_back(insertIntoFunctions[type]);
}

void GpuSqlDispatcher::addInsertIntoDoneFunction()
{
	dispatcherFunctions.push_back(insertIntoDoneFunction);
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

void GpuSqlDispatcher::addLoadFunction(DataType type)
{
	dispatcherFunctions.push_back(ldFunctions[type]);
	//dispatcherFunctions.insert(dispatcherFunctions.begin(),ldFunctions[type]);
}

void GpuSqlDispatcher::addGroupByFunction(DataType type)
{
    dispatcherFunctions.push_back(groupByFunctions[type]);
}

void GpuSqlDispatcher::addBetweenFunction(DataType op1, DataType op2, DataType op3)
{
    //TODO: Between
}

void GpuSqlDispatcher::addGpuWhereBinaryFunction(DispatcherFunction func, DataType left, DataType right)
{
	auto gpuVMFunc = reinterpret_cast<GpuVMFunction>(func * DataType::COLUMN_INT * DataType::COLUMN_INT + getConstDataType(left) * DataType::COLUMN_INT + getConstDataType(right));
	gpuOpCodes.push_back({ gpuVMFunc });
}

void GpuSqlDispatcher::addGpuWhereUnaryFunction(DispatcherFunction func, DataType type)
{
	auto gpuVMFunc = reinterpret_cast<GpuVMFunction>(func * DataType::COLUMN_INT * DataType::COLUMN_INT + getConstDataType(type));
	gpuOpCodes.push_back({ gpuVMFunc });
}

void GpuSqlDispatcher::addGpuPushWhereFunction(DataType type, const char* token)
{		
	auto gpuVMFunc = reinterpret_cast<GpuVMFunction>(DispatcherFunction::PUSH_FUNC * DataType::COLUMN_INT * DataType::COLUMN_INT + type);
	GPUOpCode opcode;
	switch (type)
	{
	case DataType::CONST_INT:
		opcode.fun_ptr = gpuVMFunc;
		*reinterpret_cast<int32_t*>(opcode.data) = std::stoi(token);
		gpuOpCodes.push_back(opcode);
		break;
	case DataType::CONST_LONG:
		opcode.fun_ptr = gpuVMFunc;
		*reinterpret_cast<int64_t*>(opcode.data) = std::stoll(token);
		gpuOpCodes.push_back(opcode);
		break;
	case DataType::CONST_FLOAT:
		opcode.fun_ptr = gpuVMFunc;
		*reinterpret_cast<float*>(opcode.data) = std::stof(token);
		gpuOpCodes.push_back(opcode);
		break;
	case DataType::CONST_DOUBLE:
		opcode.fun_ptr = gpuVMFunc;
		*reinterpret_cast<double*>(opcode.data) = std::stod(token);
		gpuOpCodes.push_back(opcode);
		break;
	case DataType::CONST_POINT:
	case DataType::CONST_POLYGON:
	case DataType::CONST_STRING:
		break;
	case DataType::COLUMN_INT:
	case DataType::COLUMN_LONG:
	case DataType::COLUMN_FLOAT:
	case DataType::COLUMN_DOUBLE:
	case DataType::COLUMN_POINT:
	case DataType::COLUMN_POLYGON:
	case DataType::COLUMN_STRING:
	case DataType::COLUMN_INT8_T:
		opcode.fun_ptr = gpuVMFunc;
		*reinterpret_cast<int8_t*>(opcode.data) = linkTable.at(token);
		gpuOpCodes.push_back(opcode);
		break;
	case DataType::DATA_TYPE_SIZE:
	case DataType::CONST_ERROR:
		break;
	}	
}

void GpuSqlDispatcher::insertComplexPolygon(const std::string& databaseName, const std::string& colName, const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons, int32_t size, bool useCache)
{
	if (useCache)
	{
		if (Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_polyPoints", blockIndex) &&
			Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_pointIdx", blockIndex) &&
			Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_pointCount", blockIndex) &&
			Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_polyIdx", blockIndex) &&
			Context::getInstance().getCacheForCurrentDevice().containsColumn(databaseName, colName + "_polyCount", blockIndex))
		{
			auto polyPoints = Context::getInstance().getCacheForCurrentDevice().getColumn<NativeGeoPoint>(databaseName, colName + "_polyPoints", blockIndex, size);
			auto pointIdx = Context::getInstance().getCacheForCurrentDevice().getColumn<int32_t>(databaseName, colName + "_pointIdx", blockIndex, size);
			auto pointCount = Context::getInstance().getCacheForCurrentDevice().getColumn<int32_t>(databaseName, colName + "_pointCount", blockIndex, size);
			auto polyIdx = Context::getInstance().getCacheForCurrentDevice().getColumn<int32_t>(databaseName, colName + "_polyIdx", blockIndex, size);
			auto polyCount = Context::getInstance().getCacheForCurrentDevice().getColumn<int32_t>(databaseName, colName + "_polyCount", blockIndex, size);
			allocatedPointers.insert({ colName + "_polyPoints", std::make_tuple(reinterpret_cast<uintptr_t>(std::get<0>(polyPoints)), size, false) });
			allocatedPointers.insert({ colName + "_pointIdx", std::make_tuple(reinterpret_cast<uintptr_t>(std::get<0>(pointIdx)), size, false) });
			allocatedPointers.insert({ colName + "_pointCount", std::make_tuple(reinterpret_cast<uintptr_t>(std::get<0>(pointCount)), size, false) });
			allocatedPointers.insert({ colName + "_polyIdx", std::make_tuple(reinterpret_cast<uintptr_t>(std::get<0>(polyIdx)), size, false) });
			allocatedPointers.insert({ colName + "_polyCount", std::make_tuple(reinterpret_cast<uintptr_t>(std::get<0>(polyCount)), size, false) });
		}
		else
		{
			GPUMemory::GPUPolygon polygon = ComplexPolygonFactory::PrepareGPUPolygon(polygons, databaseName, colName, blockIndex);
			allocatedPointers.insert({ colName + "_polyPoints", std::make_tuple(reinterpret_cast<uintptr_t>(polygon.polyPoints), size, false) });
			allocatedPointers.insert({ colName + "_pointIdx", std::make_tuple(reinterpret_cast<uintptr_t>(polygon.pointIdx), size, false) });
			allocatedPointers.insert({ colName + "_pointCount", std::make_tuple(reinterpret_cast<uintptr_t>(polygon.pointCount), size, false) });
			allocatedPointers.insert({ colName + "_polyIdx", std::make_tuple(reinterpret_cast<uintptr_t>(polygon.polyIdx), size, false) });
			allocatedPointers.insert({ colName + "_polyCount", std::make_tuple(reinterpret_cast<uintptr_t>(polygon.polyCount), size, false) });
		}
	}
	else
	{
		GPUMemory::GPUPolygon polygon = ComplexPolygonFactory::PrepareGPUPolygon(polygons);
		allocatedPointers.insert({ colName + "_polyPoints", std::make_tuple(reinterpret_cast<uintptr_t>(polygon.polyPoints), size, true) });
		allocatedPointers.insert({ colName + "_pointIdx", std::make_tuple(reinterpret_cast<uintptr_t>(polygon.pointIdx), size, true) });
		allocatedPointers.insert({ colName + "_pointCount", std::make_tuple(reinterpret_cast<uintptr_t>(polygon.pointCount), size, true) });
		allocatedPointers.insert({ colName + "_polyIdx", std::make_tuple(reinterpret_cast<uintptr_t>(polygon.polyIdx), size, true) });
		allocatedPointers.insert({ colName + "_polyCount", std::make_tuple(reinterpret_cast<uintptr_t>(polygon.polyCount), size, true) });
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

std::string GpuSqlDispatcher::insertConstPolygonGpu(ColmnarDB::Types::ComplexPolygon& polygon)
{
	std::string ret = "constPolygon" + std::to_string(constPolygonCounter);
	insertComplexPolygon(database->GetName(), ret, { polygon }, 1);
	constPolygonCounter++;
	return ret;
}

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

int32_t GpuSqlDispatcher::fil()
{
    auto reg = arguments.read<std::string>();
    std::cout << "Filter: " << reg << std::endl;
	filter_ = std::get<0>(allocatedPointers.at(reg));

	//GPU DISPATCH

	int32_t dataElementCount = std::numeric_limits<int32_t>::max();
	for (auto& column : linkTable)
	{
		symbolTable[column.second] = reinterpret_cast<void*>(std::get<0>(allocatedPointers.at(column.first)));
		if (std::get<1>(allocatedPointers.at(column.first)) < dataElementCount)
		{
			dataElementCount = std::get<1>(allocatedPointers.at(column.first));
		}
	}

	auto gpuDispatcherMask = allocateRegister<int8_t>("gpuDispatcherMask", dataElementCount);

	auto gpuDispatchSymbolTable = allocateRegister<void*>("gpuDispatchSymbolTable", linkTable.size());
	GPUMemory::copyHostToDevice(gpuDispatchSymbolTable, symbolTable.get(), linkTable.size());
	
	if (blockIndex == 0)
	{
		for (auto& gpuOpCode : gpuOpCodes)
		{
			auto idx = reinterpret_cast<std::uintptr_t>(gpuOpCode.fun_ptr);
			gpuOpCode.fun_ptr = Context::getInstance().getDispatchTableForCurrentDevice()[idx];
		}
	}

	auto gpuDispatchOpCodes = allocateRegister<GPUOpCode>("gpuDispatchOpCodes", gpuOpCodes.size());
	GPUMemory::copyHostToDevice(gpuDispatchOpCodes, gpuOpCodes.data(), gpuOpCodes.size());

	GPUWhereDispatcher::gpuWhere(gpuDispatcherMask, gpuDispatchOpCodes, gpuOpCodes.size(), gpuDispatchSymbolTable, dataElementCount);
	filter_ = reinterpret_cast<uintptr_t>(gpuDispatcherMask);
	return 0;
}

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

int32_t GpuSqlDispatcher::done()
{
	cleanUpGpuPointers();
	std::cout << "Done" << std::endl;
	return 1;
}

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