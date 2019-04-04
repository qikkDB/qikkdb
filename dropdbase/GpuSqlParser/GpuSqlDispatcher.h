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

class GpuSqlDispatcher
{
private:
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
	static std::array<DispatchFunction,
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> pointFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> containsFunctions;
    static std::array<DispatchFunction, 
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> intersectFunctions;
    static std::array<DispatchFunction, 
			DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> unionFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE> logicalNotFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE> minusFunctions;
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
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> minFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> maxFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> sumFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> countFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> avgFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE> retFunctions;
    static std::array<DispatchFunction,
            DataType::DATA_TYPE_SIZE> groupByFunctions;
    static DispatchFunction filFunction;
	static DispatchFunction jmpFunction;
    static DispatchFunction doneFunction;
	static DispatchFunction showDatabasesFunction;
	static DispatchFunction showTablesFunction;
	static DispatchFunction showColumnsFunction;
	static std::array<DispatchFunction,
		DataType::DATA_TYPE_SIZE> insertIntoFunctions;
	static DispatchFunction insertIntoDoneFunction;

	static int32_t groupByDoneCounter_;
	static int32_t groupByDoneLimit_;

	void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<int32_t[]> &data, int32_t dataSize);

	void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<int64_t[]> &data, int32_t dataSize);

	void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<float[]> &data, int32_t dataSize);

	void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<double[]> &data, int32_t dataSize);

	void insertIntoPayload(ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload, std::unique_ptr<std::string[]> &data, int32_t dataSize);
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

	void addPointFunction(DataType left, DataType right);

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

	static std::unordered_map<std::string, int32_t> linkTable;
	
	template<typename T>
	T* allocateRegister(const std::string& reg, int32_t size)
	{
		T * mask;
		GPUMemory::alloc<T>(&mask, size);
		allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), size, true)});
		usedRegisterMemory += size * sizeof(T);
		return mask;
	}

	template<typename T>
	void addCachedRegister(const std::string& reg, T* ptr, int32_t size)
	{
		allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(ptr), size, false) });
	}

	template<typename T>
	int32_t loadCol(std::string& colName);

	template <typename T>
	void freeColumnIfRegister(const std::string& col)
	{
		if (usedRegisterMemory > maxRegisterMemory && !col.empty() && col.front() == '$')
		{
			GPUMemory::free(reinterpret_cast<void*>(std::get<0>(allocatedPointers.at(col))));
			usedRegisterMemory -= std::get<1>(allocatedPointers.at(col)) * sizeof(T);
			allocatedPointers.erase(col);
			std::cout << "Free: " << col << std::endl;
		}
	}

	void mergePayloadToResponse(const std::string &key, ColmnarDB::NetworkClient::Message::QueryResponsePayload &payload);

	void insertComplexPolygon(const std::string& databaseName, const std::string& colName, const std::vector<ColmnarDB::Types::ComplexPolygon>& polygons, int32_t size, bool useCache = false);
	std::tuple<GPUMemory::GPUPolygon, int32_t> findComplexPolygon(std::string colName);
	NativeGeoPoint* insertConstPointGpu(ColmnarDB::Types::Point& point);
	std::string insertConstPolygonGpu(ColmnarDB::Types::ComplexPolygon& polygon);

    template<typename T>
    int32_t retConst();

    template<typename T>
    int32_t retCol();

    int32_t fil();

	int32_t jmp();

    int32_t done();

	int32_t showDatabases();

	int32_t showTables();

	int32_t showColumns();

	void cleanUpGpuPointers();

	template <>
	int32_t retCol<ColmnarDB::Types::ComplexPolygon>();

	template<>
	int32_t retCol<ColmnarDB::Types::Point>();

	template <>
	int32_t retCol<std::string>();


	//// FILTERS WITH FUNCTORS

	template<typename OP, typename T, typename U>
	int32_t filterColConst();

	template<typename OP, typename T, typename U>
	int32_t filterConstCol();

	template<typename OP, typename T, typename U>
	int32_t filterColCol ();

	template<typename OP, typename T, typename U>
	int32_t filterConstConst();

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

	template<typename OP, typename R, typename T, typename U>
	int32_t aggregationColCol();

	template<typename OP, typename T, typename U>
	int32_t aggregationColConst();

	template<typename OP, typename T, typename U>
	int32_t aggregationConstCol();

	template<typename OP, typename T, typename U>
	int32_t aggregationConstConst();

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

    template<typename T>
    int32_t minusCol();

    template<typename T>
    int32_t minusConst();

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

	template<>
	int32_t insertInto<ColmnarDB::Types::ComplexPolygon>();

	template<>
	int32_t insertInto<ColmnarDB::Types::Point>();

	int32_t insertIntoDone();

    template<typename T, typename U>
    int32_t invalidOperandTypesErrorHandlerColConst()
	{
		return 0;
	}

    template<typename T, typename U>
    int32_t invalidOperandTypesErrorHandlerConstCol()
	{
		return 0;
	}

    template<typename T, typename U>
    int32_t invalidOperandTypesErrorHandlerColCol()
	{
		return 0;
	}

    template<typename T, typename U>
    int32_t invalidOperandTypesErrorHandlerConstConst()
	{
		return 0;
	}


	//// FUNCTOR ERROR HANDLERS

	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerColConst()
	{
		return 0;
	}


	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerConstCol()
	{
		return 0;
	}


	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerColCol()
	{
		return 0;
	}


	template<typename OP, typename T, typename U>
	int32_t invalidOperandTypesErrorHandlerConstConst()
	{
		return 0;
	}

	template<typename OP, typename T>
	int32_t invalidOperandTypesErrorHandlerCol()
	{
		return 0;
	}

	template<typename OP, typename T>
	int32_t invalidOperandTypesErrorHandlerConst()
	{
		return 0;
	}

	////

	template<typename T>
	int32_t invalidOperandTypesErrorHandlerCol()
	{
		return 0;
	}

	template<typename T>
	int32_t invalidOperandTypesErrorHandlerConst()
	{
		return 0;
	}

    template<typename T>
    void addArgument(T argument)
    {
        arguments.insert<T>(argument);
    }
};

#endif //DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H
