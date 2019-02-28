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
#include "../ComplexPolygonFactory.h"
#include "../PointFactory.h"
#include "../DataType.h"
#include "../Database.h"
#include "../Table.h"
#include "../ColumnBase.h"
#include "../BlockBase.h"
#include "../QueryEngine/GPUCore/IGroupBy.h"

class GpuSqlDispatcher;

template<typename T>
int32_t retConst(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t retCol(GpuSqlDispatcher &dispatcher);

int32_t fil(GpuSqlDispatcher &dispatcher);

int32_t jmp(GpuSqlDispatcher &dispatcher);

int32_t done(GpuSqlDispatcher &dispatcher);

int32_t showDatabases(GpuSqlDispatcher &dispatcher);
int32_t showTables(GpuSqlDispatcher &dispatcher);
int32_t showColumns(GpuSqlDispatcher &dispatcher);
int32_t insertIntoDone(GpuSqlDispatcher & dispatcher);

//// FILTERS WITH FUNCTORS

template<typename OP, typename T, typename U>
int32_t filterColConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t filterConstCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t filterColCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t filterConstConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t logicalColConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t logicalConstCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t logicalColCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t logicalConstConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t arithmeticColConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t arithmeticConstCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t arithmeticColCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t arithmeticConstConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t aggregationColCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t aggregationColConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t aggregationConstCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t aggregationConstConst(GpuSqlDispatcher &dispatcher);

////

//contains

template<typename T, typename U>
int32_t containsColConst(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t containsConstCol(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t containsColCol(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t containsConstConst(GpuSqlDispatcher &dispatcher);

template <typename OP, typename T, typename U>
int32_t polygonOperationColConst(GpuSqlDispatcher& dispatcher);

template <typename OP, typename T, typename U>
int32_t polygonOperationConstCol(GpuSqlDispatcher& dispatcher);

template <typename OP, typename T, typename U>
int32_t polygonOperationColCol(GpuSqlDispatcher& dispatcher);

template <typename OP, typename T, typename U>
int32_t polygonOperationConstConst(GpuSqlDispatcher& dispatcher);

template<typename T>
int32_t logicalNotCol(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t logicalNotConst(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t minusCol(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t minusConst(GpuSqlDispatcher &dispatcher);

template<typename OP>
int32_t dateExtractCol(GpuSqlDispatcher &dispatcher);

template<typename OP>
int32_t dateExtractConst(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t groupByConst(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t groupByCol(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t insertInto(GpuSqlDispatcher &dispatcher);

//// FUNCTOR ERROR HANDLERS

template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T>
int32_t invalidOperandTypesErrorHandlerCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T>
int32_t invalidOperandTypesErrorHandlerConst(GpuSqlDispatcher &dispatcher);


////

template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColConst(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstCol(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColCol(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstConst(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t invalidOperandTypesErrorHandlerCol(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t invalidOperandTypesErrorHandlerConst(GpuSqlDispatcher &dispatcher);

template<typename T>
void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<T[]> &data, int32_t dataSize);

template<>
void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<int32_t[]> &data, int32_t dataSize);

template<>
void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<int64_t[]> &data, int32_t dataSize);

template<>
void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<float[]> &data, int32_t dataSize);

template<>
void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<double[]> &data, int32_t dataSize);

template<>
void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<std::string[]> &data, int32_t dataSize);

class GpuSqlDispatcher
{

private:
    std::vector<std::function<int32_t(GpuSqlDispatcher &)>> dispatcherFunctions;
    MemoryStream arguments;
	int32_t blockIndex;
	int64_t usedRegisterMemory;
	const int64_t maxRegisterMemory;
	int32_t dispatcherThreadId;
	int32_t instructionPointer;
	int32_t constPointCounter;
	int32_t constPolygonCounter;
    const std::shared_ptr<Database> &database;
	std::unordered_map<std::string, std::tuple<std::uintptr_t, int32_t, bool>> allocatedPointers;
	ColmnarDB::NetworkClient::Message::QueryResponseMessage responseMessage;
	std::uintptr_t filter_;
	bool usingGroupBy;
	bool isLastBlockOfDevice;
	bool isOverallLastBlock;
	bool noLoad;
	std::unordered_set<std::string> groupByColumns;
	bool isRegisterAllocated(std::string& reg);
	std::vector<std::unique_ptr<IGroupBy>>& groupByTables;

    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterEqualFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessEqualFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> equalFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> notEqualFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalAndFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalOrFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> mulFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> divFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> addFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> subFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> modFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> containsFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher&)>, 
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> intersectFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher&)>, 
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> unionFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> logicalNotFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> minusFunctions;
	static std::array<std::function<int32_t(GpuSqlDispatcher &)>, 
		DataType::DATA_TYPE_SIZE> yearFunctions;
	static std::array<std::function<int32_t(GpuSqlDispatcher &)>, 
		DataType::DATA_TYPE_SIZE> monthFunctions;
	static std::array<std::function<int32_t(GpuSqlDispatcher &)>, 
		DataType::DATA_TYPE_SIZE> dayFunctions;
	static std::array<std::function<int32_t(GpuSqlDispatcher &)>, 
		DataType::DATA_TYPE_SIZE> hourFunctions;
	static std::array<std::function<int32_t(GpuSqlDispatcher &)>, 
		DataType::DATA_TYPE_SIZE> minuteFunctions;
	static std::array<std::function<int32_t(GpuSqlDispatcher &)>, 
		DataType::DATA_TYPE_SIZE> secondFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> minFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> maxFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> sumFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> countFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> avgFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> retFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> groupByFunctions;
    static std::function<int32_t(GpuSqlDispatcher &)> filFunction;
	static std::function<int32_t(GpuSqlDispatcher &)> jmpFunction;
    static std::function<int32_t(GpuSqlDispatcher &)> doneFunction;
	static std::function<int32_t(GpuSqlDispatcher &)> showDatabasesFunction;
	static std::function<int32_t(GpuSqlDispatcher &)> showTablesFunction;
	static std::function<int32_t(GpuSqlDispatcher &)> showColumnsFunction;
	static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
		DataType::DATA_TYPE_SIZE> insertIntoFunctions;
	static std::function<int32_t(GpuSqlDispatcher &)> insertIntoDoneFunction;

	static int32_t groupByDoneCounter_;
	static int32_t groupByDoneLimit_;

public:
	static std::mutex groupByMutex_;
	static std::condition_variable groupByCV_;

	static void IncGroupByDoneCounter()
	{
		groupByDoneCounter_++;
	}

	static bool IsGroupByDone()
	{
		return (groupByDoneCounter_ == groupByDoneLimit_);
	}

	static void ResetGroupByCounters()
	{
		groupByDoneCounter_ = 0;
		groupByDoneLimit_ = 0;
	}

    GpuSqlDispatcher(const std::shared_ptr<Database> &database, std::vector<std::unique_ptr<IGroupBy>>& groupByTables, int dispatcherThreadId);

	~GpuSqlDispatcher();

	GpuSqlDispatcher(const GpuSqlDispatcher& dispatcher2) = delete;

	GpuSqlDispatcher& operator=(const GpuSqlDispatcher&) = delete;

	void copyExecutionDataTo(GpuSqlDispatcher& other);

	void execute(std::unique_ptr<google::protobuf::Message>& result);

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

    void addContainsFunction(DataType left, DataType right);

	void addIntersectFunction(DataType left, DataType right);

	void addUnionFunction(DataType left, DataType right);

    void addLogicalNotFunction(DataType type);

    void addMinusFunction(DataType type);

	void addYearFunction(DataType type);

	void addMonthFunction(DataType type);

	void addDayFunction(DataType type);

	void addHourFunction(DataType type);

	void addMinuteFunction(DataType type);

	void addSecondFunction(DataType type);

    void addMinFunction(DataType key, DataType value);

    void addMaxFunction(DataType key, DataType value);

    void addSumFunction(DataType key, DataType value);

    void addCountFunction(DataType key, DataType value);

    void addAvgFunction(DataType key, DataType value);

    void addRetFunction(DataType type);

    void addFilFunction();

	void addJmpInstruction();

    void addDoneFunction();

	void addShowDatabasesFunction();

	void addShowTablesFunction();

	void addShowColumnsFunction();

	void addInsertIntoFunction(DataType type);

	void addInsertIntoDoneFunction();

    void addGroupByFunction(DataType type);

    void addBetweenFunction(DataType op1, DataType op2, DataType op3);

	template<typename T>
	T* allocateRegister(const std::string& reg, int32_t size);

	template<typename T>
	void addCachedRegister(const std::string& reg, T* ptr, int32_t size);
	
	template<typename T>
	int32_t loadCol(std::string& colName)
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
			auto block = dynamic_cast<BlockBase<T>*>(col->GetBlocksList()[blockIndex].get());

			auto cacheEntry = Context::getInstance().getCacheForCurrentDevice().getColumn<T>(
				database->GetName(), colName, blockIndex, block->GetSize());
            if (!std::get<2>(cacheEntry))
            {
                GPUMemory::copyHostToDevice(std::get<0>(cacheEntry), block->GetData(), block->GetSize());
            }
            addCachedRegister(colName, std::get<0>(cacheEntry), block->GetSize());

			noLoad = false;
		}
		return 0;
	}
	template <typename T>
	void freeColumnIfRegister(std::string& col);

	void mergePayloadToResponse(const std::string &key, ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload);

	void insertComplexPolygon(const std::string& databaseName, const std::string& colName, const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons, int32_t size, bool useCache = false);
	std::tuple<GPUMemory::GPUPolygon, int32_t> findComplexPolygon(std::string colName);
	NativeGeoPoint* insertConstPointGpu(ColmnarDB::Types::Point& point);
	std::string insertConstPolygonGpu(ColmnarDB::Types::ComplexPolygon& polygon);

    template<typename T>
    friend int32_t retConst(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t retCol(GpuSqlDispatcher &dispatcher);

    friend int32_t fil(GpuSqlDispatcher &dispatcher);

	friend int32_t jmp(GpuSqlDispatcher &dispatcher);

    friend int32_t done(GpuSqlDispatcher &dispatcher);

	friend int32_t showDatabases(GpuSqlDispatcher &dispatcher);

	friend int32_t showTables(GpuSqlDispatcher &dispatcher);

	friend int32_t showColumns(GpuSqlDispatcher &dispatcher);

	void cleanUpGpuPointers();

	//// FILTERS WITH FUNCTORS

	template<typename OP, typename T, typename U>
	friend int32_t filterColConst(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t filterConstCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t filterColCol (GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t filterConstConst(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t logicalColConst(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t logicalConstCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t logicalColCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t logicalConstConst(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t arithmeticColConst(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t arithmeticConstCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t arithmeticColCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t arithmeticConstConst(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename R, typename T, typename U>
	friend int32_t aggregationColCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t aggregationColConst(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t aggregationConstCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t aggregationConstConst(GpuSqlDispatcher &dispatcher);

	////

	//contains

    template<typename T, typename U>
    friend int32_t containsColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t containsConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t containsColCol(GpuSqlDispatcher &dispatcher);

    template <typename T, typename U>
    friend int32_t containsConstConst(GpuSqlDispatcher& dispatcher);

	template <typename OP, typename T, typename U>
    friend int32_t polygonOperationColConst(GpuSqlDispatcher& dispatcher);

    template <typename OP, typename T, typename U>
    friend int32_t polygonOperationConstCol(GpuSqlDispatcher& dispatcher);

    template <typename OP, typename T, typename U>
    friend int32_t polygonOperationColCol(GpuSqlDispatcher& dispatcher);

    template <typename OP, typename T, typename U>
    friend int32_t polygonOperationConstConst(GpuSqlDispatcher& dispatcher);

    int32_t between();

    template<typename T>
    friend int32_t logicalNotCol(GpuSqlDispatcher &dispatcher);


    template<typename T>
    friend int32_t logicalNotConst(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t minusCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t minusConst(GpuSqlDispatcher &dispatcher);

	template<typename OP>
	friend int32_t dateExtractCol(GpuSqlDispatcher &dispatcher);

	template<typename OP>
	friend int32_t dateExtractConst(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t groupByCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t groupByConst(GpuSqlDispatcher &dispatcher);

	template<typename T>
	friend int32_t insertInto(GpuSqlDispatcher &dispatcher);

	friend int32_t insertIntoDone(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t invalidOperandTypesErrorHandlerColConst(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t invalidOperandTypesErrorHandlerConstCol(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t invalidOperandTypesErrorHandlerColCol(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t invalidOperandTypesErrorHandlerConstConst(GpuSqlDispatcher &dispatcher);


	//// FUNCTOR ERROR HANDLERS

	template<typename OP, typename T, typename U>
	friend int32_t invalidOperandTypesErrorHandlerColConst(GpuSqlDispatcher &dispatcher);


	template<typename OP, typename T, typename U>
	friend int32_t invalidOperandTypesErrorHandlerConstCol(GpuSqlDispatcher &dispatcher);


	template<typename OP, typename T, typename U>
	friend int32_t invalidOperandTypesErrorHandlerColCol(GpuSqlDispatcher &dispatcher);


	template<typename OP, typename T, typename U>
	friend int32_t invalidOperandTypesErrorHandlerConstConst(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T>
	friend int32_t invalidOperandTypesErrorHandlerCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T>
	friend int32_t invalidOperandTypesErrorHandlerConst(GpuSqlDispatcher &dispatcher);

	////

	template<typename T>
	friend int32_t invalidOperandTypesErrorHandlerCol(GpuSqlDispatcher &dispatcher);

	template<typename T>
	friend int32_t invalidOperandTypesErrorHandlerConst(GpuSqlDispatcher &dispatcher);

    template<typename T>
    void addArgument(T argument)
    {
        arguments.insert<T>(argument);
    }
};

#endif //DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H
