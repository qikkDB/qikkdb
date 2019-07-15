//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H
#define DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H

#include <functional>
#include <vector>
#include <iostream>
#include <memory>
#include <array>
#include <regex>
#include <string>
#include <mutex>
#include <condition_variable>
#include "../messages/QueryResponseMessage.pb.h"
#include "MemoryStream.h"
#include "../DataType.h"
#include "../QueryEngine/OrderByType.h"
#include "../IVariantArray.h"
#include "../QueryEngine/GPUCore/IGroupBy.h"
#include "../NativeGeoPoint.h"
#include "../QueryEngine/GPUCore/GPUMemory.cuh"
#include "ParserExceptions.h"

#ifndef NDEBUG
void AssertDeviceMatchesCurrentThread(int dispatcherThreadId);
#endif


struct OrderByBlocks
{
	std::unordered_map<std::string, std::vector<std::unique_ptr<IVariantArray>>> reconstructedOrderByOrderColumnBlocks;
	std::unordered_map<std::string, std::vector<std::unique_ptr<int8_t[]>>> reconstructedOrderByOrderColumnNullBlocks;
	
	std::unordered_map<std::string, std::vector<std::unique_ptr<IVariantArray>>> reconstructedOrderByRetColumnBlocks;
	std::unordered_map<std::string, std::vector<std::unique_ptr<int8_t[]>>> reconstructedOrderByRetColumnNullBlocks;
};

class Database; 
class GPUOrderBy;

class GpuSqlDispatcher
{
private:
	struct PointerAllocation
	{
		std::uintptr_t gpuPtr;
		int32_t elementCount; 
		bool shouldBeFreed;
		std::uintptr_t gpuNullMaskPtr;
	};
	
	typedef int32_t(GpuSqlDispatcher::*DispatchFunction)();
    std::vector<DispatchFunction> dispatcherFunctions;
    MemoryStream arguments;
	int32_t blockIndex;
	int64_t usedRegisterMemory;
	const int64_t maxRegisterMemory;
	int32_t dispatcherThreadId;
	int32_t instructionPointer;
	int32_t constPointCounter;
	int32_t constPolygonCounter;
	int32_t constStringCounter;
    const std::shared_ptr<Database> &database;
	std::unordered_map<std::string, PointerAllocation> allocatedPointers;
	std::unordered_map<std::string, std::vector<std::vector<int32_t>>>* joinIndices;

	ColmnarDB::NetworkClient::Message::QueryResponseMessage responseMessage;
	std::uintptr_t filter_;
	bool usingGroupBy;
	bool usingOrderBy;
	bool usingJoin;
	bool isLastBlockOfDevice;
	bool isOverallLastBlock;
	bool noLoad;
	std::unordered_set<std::string> groupByColumns;
	std::unordered_set<std::string> aggregatedRegisters;
	std::unordered_set<std::string> registerLockList;
	bool isRegisterAllocated(std::string& reg);
	std::pair<std::string, std::string> splitColumnName(const std::string& colName);
	std::vector<std::unique_ptr<IGroupBy>>& groupByTables;
	std::unique_ptr<GPUOrderBy> orderByTable;
	std::vector<OrderByBlocks>& orderByBlocks;
	
	std::unordered_map<std::string, std::unique_ptr<IVariantArray>> reconstructedOrderByColumnsMerged;
	std::unordered_map<std::string, std::unique_ptr<int8_t[]>> reconstructedOrderByColumnsNullMerged;

	std::unordered_map<int32_t, std::pair<std::string, OrderBy::Order>> orderByColumns;
	std::vector<std::vector<int32_t>> orderByIndices;
	

    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterEqualFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessEqualFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> equalFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> notEqualFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalAndFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalOrFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> mulFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> divFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> addFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> subFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> modFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseOrFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseAndFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseXorFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseLeftShiftFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseRightShiftFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logarithmFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> arctangent2Functions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> concatFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> powerFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> rootFunctions;
	static std::array<DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> pointFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> containsFunctions;
    static std::array<DispatchFunction, 
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> intersectFunctions;
    static std::array<DispatchFunction, 
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> unionFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> leftFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> rightFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE> logicalNotFunctions;
	static std::array<DispatchFunction, 
		DataType::DATA_TYPE_SIZE> yearFunctions;
	static std::array<DispatchFunction, 
		DataType::DATA_TYPE_SIZE> monthFunctions;
	static std::array<DispatchFunction, 
		DataType::DATA_TYPE_SIZE> dayFunctions;
	static std::array<DispatchFunction, 
		DataType::DATA_TYPE_SIZE> hourFunctions;
	static std::array<DispatchFunction, 
		DataType::DATA_TYPE_SIZE> minuteFunctions;
	static std::array<DispatchFunction, 
		DataType::DATA_TYPE_SIZE> secondFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> minusFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> absoluteFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> sineFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> cosineFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> tangentFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> cotangentFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> arcsineFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> arccosineFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> arctangentFunctions;
    static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> logarithm10Functions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> logarithmNaturalFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> exponentialFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> squareRootFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> squareFunctions;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> signFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
		DataType::DATA_TYPE_SIZE> roundFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
		DataType::DATA_TYPE_SIZE> ceilFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
		DataType::DATA_TYPE_SIZE> floorFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
		DataType::DATA_TYPE_SIZE> ltrimFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
		DataType::DATA_TYPE_SIZE> rtrimFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
		DataType::DATA_TYPE_SIZE> lowerFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
		DataType::DATA_TYPE_SIZE> upperFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
		DataType::DATA_TYPE_SIZE> reverseFunctions;
	static std::array<GpuSqlDispatcher::DispatchFunction,
		DataType::DATA_TYPE_SIZE> lenFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> minAggregationFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> maxAggregationFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> sumAggregationFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> countAggregationFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> avgAggregationFunctions;
	static std::array<DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> minGroupByFunctions;
	static std::array<DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> maxGroupByFunctions;
	static std::array<DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> sumGroupByFunctions;
	static std::array<DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> countGroupByFunctions;
	static std::array<DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> avgGroupByFunctions;
	static std::array<DispatchFunction,
			DataType::DATA_TYPE_SIZE> orderByFunctions;
	static std::array<DispatchFunction,
			DataType::DATA_TYPE_SIZE> orderByReconstructOrderFunctions;
	static std::array<DispatchFunction,
			DataType::DATA_TYPE_SIZE> orderByReconstructRetFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE> retFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE> groupByFunctions;
	static DispatchFunction isNullFunction;
	static DispatchFunction isNotNullFunction;
	static DispatchFunction freeOrderByTableFunction;
	static DispatchFunction orderByReconstructRetAllBlocksFunction;
    static DispatchFunction filFunction;
	static DispatchFunction lockRegisterFunction;
	static DispatchFunction jmpFunction;
    static DispatchFunction doneFunction;
	static DispatchFunction showDatabasesFunction;
	static DispatchFunction showTablesFunction;
	static DispatchFunction showColumnsFunction;
	static DispatchFunction createDatabaseFunction;
	static DispatchFunction dropDatabaseFunction;
	static DispatchFunction createTableFunction;
	static DispatchFunction dropTableFunction;
	static DispatchFunction alterTableFunction;
	static DispatchFunction createIndexFunction;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> insertIntoFunctions;
	static DispatchFunction insertIntoDoneFunction;

	static int32_t groupByDoneCounter_;
	static int32_t orderByDoneCounter_;
	static int32_t deviceCountLimit_;

	void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<int32_t[]> &data, int32_t dataSize);

	void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<int64_t[]> &data, int32_t dataSize);

	void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<float[]> &data, int32_t dataSize);

	void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<double[]> &data, int32_t dataSize);

	void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<std::string[]> &data, int32_t dataSize);
public:
	static std::mutex groupByMutex_;
	static std::mutex orderByMutex_;

	static std::condition_variable groupByCV_;
	static std::condition_variable orderByCV_;

	static void IncGroupByDoneCounter()
	{
		groupByDoneCounter_++;
	}

	static void IncOrderByDoneCounter()
	{
		orderByDoneCounter_++;
	}

	static bool IsGroupByDone()
	{
		return (groupByDoneCounter_ == deviceCountLimit_);
	}

	static bool IsOrderByDone()
	{
		return (orderByDoneCounter_ == deviceCountLimit_);
	}

	static void ResetGroupByCounters()
	{
		groupByDoneCounter_ = 0;
		deviceCountLimit_ = 0;
	}

	static void ResetOrderByCounters()
	{
		orderByDoneCounter_ = 0;
		deviceCountLimit_ = 0;
	}

	template<typename T>
	static std::pair<bool, T> AggregateOnCPU(std::string& operation, T number1, T number2)
	{
		if (operation == "MIN")
		{
			return std::make_pair(true, number1 < number2 ? number1 : number2);
		}
		else if (operation == "MAX")
		{
			return std::make_pair(true, number1 > number2 ? number1 : number2);
		}
		else if (operation == "SUM" || operation == "AVG" || operation == "COUNT")
		{
			return std::make_pair(true, number1 + number2);
		}
		else    // Other operation (e.g. datetime)
		{
			return std::make_pair(false, T{ 0 });
		}
	}

	static void MergePayload(const std::string &key, ColmnarDB::NetworkClient::Message::QueryResponseMessage * responseMessage,
		ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload);
	static void MergePayloadBitmask(const std::string &key, ColmnarDB::NetworkClient::Message::QueryResponseMessage * responseMessage, const std::string& nullMask);


    GpuSqlDispatcher(const std::shared_ptr<Database> &database, std::vector<std::unique_ptr<IGroupBy>>& groupByTables, std::vector<OrderByBlocks>& orderByBlocks, int dispatcherThreadId);
	~GpuSqlDispatcher();

	GpuSqlDispatcher(const GpuSqlDispatcher& dispatcher2) = delete;

	GpuSqlDispatcher& operator=(const GpuSqlDispatcher&) = delete;

	void copyExecutionDataTo(GpuSqlDispatcher& other);

	void setJoinIndices(std::unordered_map<std::string, std::vector<std::vector<int32_t>>>* joinIdx);

	void execute(std::unique_ptr<google::protobuf::Message>& result, std::exception_ptr& exception);

	const ColmnarDB::NetworkClient::Message::QueryResponseMessage &getQueryResponseMessage();

    void addGreaterFunction(DataType left, DataType right);

    void addLessFunction(DataType left, DataType right);

    void addGreaterEqualFunction(DataType left, DataType right);

    void addLessEqualFunction(DataType left, DataType right);

    void addEqualFunction(DataType left, DataType right);

    void addNotEqualFunction(DataType left, DataType right);

    void addLogicalAndFunction(DataType left, DataType right);

    void addLogicalOrFunction(DataType left, DataType right);

    void addMulFunction(DataType left, DataType right);

    void addDivFunction(DataType left, DataType right);

    void addAddFunction(DataType left, DataType right);

    void addSubFunction(DataType left, DataType right);

	void addModFunction(DataType left, DataType right);

    void addBitwiseOrFunction(DataType left, DataType right);

	void addBitwiseAndFunction(DataType left, DataType right);

	void addBitwiseXorFunction(DataType left, DataType right);

	void addBitwiseLeftShiftFunction(DataType left, DataType right);

	void addBitwiseRightShiftFunction(DataType left, DataType right);

	void addPointFunction(DataType left, DataType right);

    void addContainsFunction(DataType left, DataType right);

	void addIntersectFunction(DataType left, DataType right);

	void addUnionFunction(DataType left, DataType right);

    void addLogicalNotFunction(DataType type);

	void addIsNullFunction();

	void addIsNotNullFunction();

    void addMinusFunction(DataType type);

	void addYearFunction(DataType type);

	void addMonthFunction(DataType type);

	void addDayFunction(DataType type);

	void addHourFunction(DataType type);

	void addMinuteFunction(DataType type);

	void addSecondFunction(DataType type);

	void addAbsoluteFunction(DataType type);

	void addSineFunction(DataType type);

	void addCosineFunction(DataType type);

	void addTangentFunction(DataType type);

	void addCotangentFunction(DataType type);

	void addArcsineFunction(DataType type);

	void addArccosineFunction(DataType type);

	void addArctangentFunction(DataType type);

	void addLogarithm10Function(DataType type);

	void addLogarithmFunction(DataType number, DataType base);

	void addArctangent2Function(DataType y, DataType x);

	void addConcatFunction(DataType left, DataType right);

	void addLeftFunction(DataType left, DataType right);

	void addRightFunction(DataType left, DataType right);

	void addLogarithmNaturalFunction(DataType type);

	void addExponentialFunction(DataType type);

	void addPowerFunction(DataType base, DataType exponent);

	void addSquareRootFunction(DataType type);

	void addSquareFunction(DataType type);

	void addSignFunction(DataType type);

	void addRoundFunction(DataType type);

	void addFloorFunction(DataType type);

	void addCeilFunction(DataType type);

	void addLtrimFunction(DataType type);

	void addRtrimFunction(DataType type);

	void addLowerFunction(DataType type);

	void addUpperFunction(DataType type);

	void addReverseFunction(DataType type);

	void addLenFunction(DataType type);

	void addRootFunction(DataType base, DataType exponent);

    void addMinFunction(DataType key, DataType value, bool usingGroupBy);

    void addMaxFunction(DataType key, DataType value, bool usingGroupBy);

    void addSumFunction(DataType key, DataType value, bool usingGroupBy);

    void addCountFunction(DataType key, DataType value, bool usingGroupBy);

    void addAvgFunction(DataType key, DataType value, bool usingGroupBy);

    void addRetFunction(DataType type);

	void addOrderByFunction(DataType type);

	void addOrderByReconstructOrderFunction(DataType type);

	void addOrderByReconstructRetFunction(DataType type);

	void addFreeOrderByTableFunction();

	void addOrderByReconstructRetAllBlocksFunction();

	void addLockRegisterFunction();

    void addFilFunction();

	void addJmpInstruction();

    void addDoneFunction();

	void addShowDatabasesFunction();

	void addShowTablesFunction();

	void addShowColumnsFunction();

	void addCreateDatabaseFunction();

	void addDropDatabaseFunction();

	void addCreateTableFunction();

	void addDropTableFunction();

	void addAlterTableFunction();

	void addCreateIndexFunction();

	void addInsertIntoFunction(DataType type);

	void addInsertIntoDoneFunction();

    void addGroupByFunction(DataType type);

    void addBetweenFunction(DataType op1, DataType op2, DataType op3);

	static std::unordered_map<std::string, int32_t> linkTable;
	
	template<typename T>
	T* allocateRegister(const std::string& reg, int32_t size, int8_t** nullPointerMask = nullptr)
	{
		T * mask;
		GPUMemory::alloc<T>(&mask, size);
		if(nullPointerMask)
		{
			int32_t bitMaskSize = ((size + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
			GPUMemory::alloc<int8_t>(nullPointerMask, bitMaskSize);
			allocatedPointers.insert({ reg, PointerAllocation{reinterpret_cast<std::uintptr_t>(mask), size, true, reinterpret_cast<std::uintptr_t>(*nullPointerMask)}});
		}
		else
		{
			allocatedPointers.insert({ reg, PointerAllocation{reinterpret_cast<std::uintptr_t>(mask), size, true, 0}});
		}
		
		usedRegisterMemory += size * sizeof(T);
		return mask;
	}

	void fillPolygonRegister(GPUMemory::GPUPolygon& polygonColumn, const std::string& reg, int32_t size, bool useCache = false, int8_t* nullMaskPtr = nullptr);
	std::string getAllocatedRegisterName(const std::string& reg);
	


	void fillStringRegister(GPUMemory::GPUString& stringColumn, const std::string& reg, int32_t size, bool useCache = false, int8_t* nullMaskPtr = nullptr);

	template<typename T>
	void addCachedRegister(const std::string& reg, T* ptr, int32_t size, int8_t* nullMaskPtr = nullptr)
	{
		allocatedPointers.insert({ reg, {reinterpret_cast<std::uintptr_t>(ptr), size, false, reinterpret_cast<std::uintptr_t>(nullMaskPtr)} });
	}

	template<typename T>
	int32_t loadCol(std::string& colName);

	int32_t loadColNullMask(std::string& colName);

	template <typename T>
	void freeColumnIfRegister(const std::string& col)
	{
		if (usedRegisterMemory > maxRegisterMemory && !col.empty() && col.front() == '$' && registerLockList.find(col) == registerLockList.end())
		{
			GPUMemory::free(reinterpret_cast<void*>(allocatedPointers.at(col).gpuPtr));
			usedRegisterMemory -= allocatedPointers.at(col).elementCount * sizeof(T);
			allocatedPointers.erase(col);
			std::cout << "Free: " << col << std::endl;
		}
	}

	// TODO freeColumnIfRegister<std::string> laso point and polygon
	void MergePayloadToSelfResponse(const std::string &key, ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, const std::string& nullBitMaskString = "");

	GPUMemory::GPUPolygon insertComplexPolygon(const std::string& databaseName, const std::string& colName, const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons, int32_t size, bool useCache = false, int8_t* nullMaskPtr = nullptr);
	GPUMemory::GPUString insertString(const std::string& databaseName, const std::string& colName, const std::vector<std::string>& strings, int32_t size, bool useCache = false, int8_t* nullMaskPtr = nullptr);
	std::tuple<GPUMemory::GPUPolygon, int32_t, int8_t*> findComplexPolygon(std::string colName);
	std::tuple<GPUMemory::GPUString, int32_t, int8_t*> findStringColumn(const std::string &colName);
	NativeGeoPoint* insertConstPointGpu(ColmnarDB::Types::Point& point);
	GPUMemory::GPUPolygon insertConstPolygonGpu(ColmnarDB::Types::ComplexPolygon& polygon);
	GPUMemory::GPUString insertConstStringGpu(const std::string& str);

	template<typename T>
	int32_t orderByConst();

	template<typename T>
	int32_t orderByCol();

	template<typename T>
	int32_t orderByReconstructOrderConst();

	template<typename T>
	int32_t orderByReconstructOrderCol();

	template<typename T>
	int32_t orderByReconstructRetConst();

	template<typename T>
	int32_t orderByReconstructRetCol();

	int32_t orderByReconstructRetAllBlocks();

  	template<typename T>
    int32_t retConst();

    template<typename T>
    int32_t retCol();

	int32_t freeOrderByTable();

	int32_t lockRegister();

    int32_t fil();

	int32_t jmp();

    int32_t done();

	int32_t showDatabases();

	int32_t showTables();

	int32_t showColumns();

	int32_t createDatabase();

	int32_t dropDatabase();

	int32_t createTable();

	int32_t dropTable();

	int32_t alterTable();

	int32_t createIndex();

	void cleanUpGpuPointers();


	//// FILTERS WITH FUNCTORS

	template<typename OP, typename T, typename U>
	int32_t filterColConst();

	template<typename OP, typename T, typename U>
	int32_t filterConstCol();

	template<typename OP, typename T, typename U>
	int32_t filterColCol ();

	template<typename OP, typename T, typename U>
	int32_t filterConstConst();

	template<typename OP>
	int32_t filterStringColConst();

	template<typename OP>
	int32_t filterStringConstCol();

	template<typename OP>
	int32_t filterStringColCol();

	template<typename OP>
	int32_t filterStringConstConst();

	template<typename OP, typename T, typename U>
	int32_t logicalColConst();

	template<typename OP, typename T, typename U>
	int32_t logicalConstCol();

	template<typename OP, typename T, typename U>
	int32_t logicalColCol();

	template<typename OP, typename T, typename U>
	int32_t logicalConstConst();

	template<typename OP, typename T, typename U>
	int32_t arithmeticColConst();

	template<typename OP, typename T, typename U>
	int32_t arithmeticConstCol();

	template<typename OP, typename T, typename U>
	int32_t arithmeticColCol();

	template<typename OP, typename T, typename U>
	int32_t arithmeticConstConst();

	template<typename OP, typename T>
	int32_t arithmeticUnaryCol();

	template<typename OP, typename T>
	int32_t arithmeticUnaryConst();

	template<typename OP>
	int32_t stringUnaryCol();

	template<typename OP>
	int32_t stringUnaryConst();

	template<typename OP>
	int32_t stringUnaryNumericCol();

	template<typename OP>
	int32_t stringUnaryNumericConst();

	template<typename OP>
	int32_t stringBinaryColCol();

	template<typename OP>
	int32_t stringBinaryColConst();

	template<typename OP>
	int32_t stringBinaryConstCol();

	template<typename OP>
	int32_t stringBinaryConstConst();

	template<typename OP, typename T>
	int32_t stringBinaryNumericColCol();

	template<typename OP, typename T>
	int32_t stringBinaryNumericColConst();

	template<typename OP, typename T>
	int32_t stringBinaryNumericConstCol();

	template<typename OP, typename T>
	int32_t stringBinaryNumericConstConst();

	template<typename OP, typename R, typename T, typename U>
	int32_t aggregationGroupBy();

	template<typename OP, typename OUT, typename IN>
	int32_t aggregationCol();

	template<typename OP, typename T, typename U>
	int32_t aggregationConst();

	////

	// point from columns

	template<typename T, typename U>
	int32_t pointColCol();

	template<typename T, typename U>
	int32_t pointColConst();

	template<typename T, typename U>
	int32_t pointConstCol();

	//contains

    template<typename T, typename U>
    int32_t containsColConst();

    template<typename T, typename U>
    int32_t containsConstCol();

    template<typename T, typename U>
    int32_t containsColCol();

    template <typename T, typename U>
    int32_t containsConstConst();

	template <typename OP, typename T, typename U>
    int32_t polygonOperationColConst();

    template <typename OP, typename T, typename U>
    int32_t polygonOperationConstCol();

    template <typename OP, typename T, typename U>
    int32_t polygonOperationColCol();

    template <typename OP, typename T, typename U>
    int32_t polygonOperationConstConst();

    int32_t between();

    template<typename T>
    int32_t logicalNotCol();

    template<typename T>
    int32_t logicalNotConst();

	template<typename OP>
	int32_t nullMaskCol();

	template<typename OP>
	int32_t dateExtractCol();

	template<typename OP>
	int32_t dateExtractConst();

    template<typename T>
    int32_t groupByCol();

    template<typename T>
    int32_t groupByConst();

	template<typename T>
	int32_t insertInto();

	int32_t insertIntoDone();

    template<typename T, typename U>
    int32_t invalidOperandTypesErrorHandlerColConst()
	{
		U cnst = arguments.read<U>();
		auto colName = arguments.read<std::string>();

		throw InvalidOperandsException(colName, std::string("cnst"), std::string("operation"));
	}

    template<typename T, typename U>
    int32_t invalidOperandTypesErrorHandlerConstCol()
	{
		auto colName = arguments.read<std::string>();
		T cnst = arguments.read<T>();

		throw InvalidOperandsException(colName, std::string("cnst"), std::string("operation"));
	}

    template<typename T, typename U>
    int32_t invalidOperandTypesErrorHandlerColCol()
	{
		auto colNameRight = arguments.read<std::string>();
		auto colNameLeft = arguments.read<std::string>();

		throw InvalidOperandsException(colNameLeft, colNameRight, std::string("operation"));
	}

    template<typename T, typename U>
    int32_t invalidOperandTypesErrorHandlerConstConst()
	{
		U cnstRight = arguments.read<U>();
		T cnstLeft = arguments.read<T>();

		throw InvalidOperandsException(std::string("cnst"), std::string("cnst"), std::string("operation"));
	}


	//// FUNCTOR ERROR HANDLERS

	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerColConst()
	{
		U cnst = arguments.read<U>();
		auto colName = arguments.read<std::string>();

		throw InvalidOperandsException (colName, std::string("cnst"), std::string(typeid(OP).name()));
	}


	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerConstCol()
	{
		auto colName = arguments.read<std::string>();
		T cnst = arguments.read<T>();

		throw InvalidOperandsException(colName, std::string("cnst"), std::string(typeid(OP).name()));
	}


	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerColCol()
	{
		auto colNameRight = arguments.read<std::string>();
		auto colNameLeft = arguments.read<std::string>();

		throw InvalidOperandsException(colNameLeft, colNameRight, std::string(typeid(OP).name()));
	}


	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerConstConst()
	{
		U cnstRight = arguments.read<U>();
		T cnstLeft = arguments.read<T>();

		throw InvalidOperandsException(std::string("cnst"), std::string("cnst"), std::string(typeid(OP).name()));
	}

	template<typename OP, typename T>
	int32_t invalidOperandTypesErrorHandlerCol()
	{
		auto colName = arguments.read<std::string>();

		throw InvalidOperandsException(colName, std::string(""), std::string(typeid(OP).name()));
	}

	template<typename OP, typename T>
	int32_t invalidOperandTypesErrorHandlerConst()
	{
		T cnst = arguments.read<T>();

		throw InvalidOperandsException(std::string(""), std::string("cnst"), std::string(typeid(OP).name()));
	}

	////

	template<typename T>
	int32_t invalidOperandTypesErrorHandlerCol()
	{
		auto colName = arguments.read<std::string>();

		throw InvalidOperandsException(colName, std::string(""), std::string("operation"));
	}

	template<typename T>
	int32_t invalidOperandTypesErrorHandlerConst()
	{
		T cnst = arguments.read<T>();

		throw InvalidOperandsException(std::string(""), std::string("cnst"), std::string("operation"));
	}

    template<typename T>
    void addArgument(T argument)
    {
        arguments.insert<T>(argument);
    }

private:
	template<typename OP, typename O, typename K, typename V>
	class GroupByHelper;

	template<typename OP, typename O, typename V>
	class GroupByHelper<OP, O, std::string, V>;

};

template <>
int32_t GpuSqlDispatcher::retCol<ColmnarDB::Types::ComplexPolygon>();

template<>
int32_t GpuSqlDispatcher::retCol<ColmnarDB::Types::Point>();

template <>
int32_t GpuSqlDispatcher::retCol<std::string>();

template<>
int32_t GpuSqlDispatcher::groupByCol<std::string>();

template<>
int32_t GpuSqlDispatcher::insertInto<ColmnarDB::Types::ComplexPolygon>();

template<>
int32_t GpuSqlDispatcher::insertInto<ColmnarDB::Types::Point>();

template<>
int32_t GpuSqlDispatcher::loadCol<ColmnarDB::Types::ComplexPolygon>(std::string& colName);

template<>
int32_t GpuSqlDispatcher::loadCol<ColmnarDB::Types::Point>(std::string& colName);

template<>
int32_t GpuSqlDispatcher::loadCol<std::string>(std::string& colName);


#endif //DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H
