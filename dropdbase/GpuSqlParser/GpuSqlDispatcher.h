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
#include <string>
#include "../messages/QueryResponseMessage.pb.h"
#include "MemoryStream.h"
#include "../ComplexPolygonFactory.h"
#include "../PointFactory.h"
#include "../DataType.h"
#include "../Database.h"
#include "../Table.h"
#include "../ColumnBase.h"
#include "../BlockBase.h"

class GpuSqlDispatcher;

template<typename T>
int32_t loadConst(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t loadCol(GpuSqlDispatcher &dispatcher);

int32_t loadReg(GpuSqlDispatcher &dispatcher);


template<typename T>
int32_t retConst(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t retCol(GpuSqlDispatcher &dispatcher);

int32_t retReg(GpuSqlDispatcher &dispatcher);

int32_t fil(GpuSqlDispatcher &dispatcher);

int32_t done(GpuSqlDispatcher &dispatcher);

//// FILTERS WITH FUNCTORS

template<typename OP, typename T, typename U>
int32_t filterColConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t filterConstCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t filterColCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t filterConstConst(GpuSqlDispatcher &dispatcher);

template<typename OP>
int32_t filterRegReg(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t logicalColConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t logicalConstCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t logicalColCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t logicalConstConst(GpuSqlDispatcher &dispatcher);

template<typename OP>
int32_t logicalRegReg(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t arithmeticColConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t arithmeticConstCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t arithmeticColCol(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t arithmeticConstConst(GpuSqlDispatcher &dispatcher);

template<typename OP>
int32_t arithmeticRegReg(GpuSqlDispatcher &dispatcher);

////

template<typename T, typename U>
int32_t containsColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t containsConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t containsColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t containsConstConst(GpuSqlDispatcher &dispatcher);


int32_t containsRegReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t logicalNotCol(GpuSqlDispatcher &dispatcher);


template<typename T>
int32_t logicalNotConst(GpuSqlDispatcher &dispatcher);


int32_t logicalNotReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t minusCol(GpuSqlDispatcher &dispatcher);


template<typename T>
int32_t minusConst(GpuSqlDispatcher &dispatcher);


int32_t minusReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t minCol(GpuSqlDispatcher &dispatcher);


template<typename T>
int32_t minConst(GpuSqlDispatcher &dispatcher);


int32_t minReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t maxCol(GpuSqlDispatcher &dispatcher);


template<typename T>
int32_t maxConst(GpuSqlDispatcher &dispatcher);


int32_t maxReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t sumCol(GpuSqlDispatcher &dispatcher);


template<typename T>
int32_t sumConst(GpuSqlDispatcher &dispatcher);


int32_t sumReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t countCol(GpuSqlDispatcher &dispatcher);


template<typename T>
int32_t countConst(GpuSqlDispatcher &dispatcher);


int32_t countReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t avgCol(GpuSqlDispatcher &dispatcher);


template<typename T>
int32_t avgConst(GpuSqlDispatcher &dispatcher);


int32_t avgReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t groupByConst(GpuSqlDispatcher &dispatcher);


template<typename T>
int32_t groupByCol(GpuSqlDispatcher &dispatcher);


int32_t groupByReg(GpuSqlDispatcher &dispatcher);

//// FUNCTOR ERROR HANDLERS

template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColConst(GpuSqlDispatcher &dispatcher);


template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstCol(GpuSqlDispatcher &dispatcher);


template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColCol(GpuSqlDispatcher &dispatcher);


template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstConst(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerRegCol(GpuSqlDispatcher &dispatcher);


template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerRegConst(GpuSqlDispatcher &dispatcher);


template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColReg(GpuSqlDispatcher &dispatcher);


template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstReg(GpuSqlDispatcher &dispatcher);

////

template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerRegCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerRegConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColReg(GpuSqlDispatcher &dispatcher);

int32_t invalidOperandTypesErrorHandlerRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t invalidOperandTypesErrorHandlerCol(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t invalidOperandTypesErrorHandlerConst(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t invalidOperandTypesErrorHandlerReg(GpuSqlDispatcher &dispatcher);

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

class GpuSqlDispatcher
{

private:
    std::vector<std::function<int32_t(GpuSqlDispatcher &)>> dispatcherFunctions;
    MemoryStream arguments;
	int32_t blockIndex;
	int32_t constPointCounter;
	int32_t constPolygonCounter;
    const std::shared_ptr<Database> &database;
	std::unordered_map<std::string, std::tuple<std::uintptr_t, int32_t>> allocatedPointers;
	ColmnarDB::NetworkClient::Message::QueryResponseMessage responseMessage;
	std::uintptr_t filter_;

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
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> logicalNotFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> minusFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> minFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> maxFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> sumFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> countFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> avgFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> loadFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> retFunctions;
    static std::array<std::function<int32_t(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> groupByFunctions;
    static std::function<int32_t(GpuSqlDispatcher &)> filFunction;
    static std::function<int32_t(GpuSqlDispatcher &)> doneFunction;

public:
    explicit GpuSqlDispatcher(const std::shared_ptr<Database> &database);

	std::unique_ptr<google::protobuf::Message> execute();

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

    void addLogicalNotFunction(DataType type);

    void addMinusFunction(DataType type);

    void addMinFunction(DataType type);

    void addMaxFunction(DataType type);

    void addSumFunction(DataType type);

    void addCountFunction(DataType type);

    void addAvgFunction(DataType type);

    void addLoadFunction(DataType type);

    void addRetFunction(DataType type);

    void addFilFunction();

    void addDoneFunction();

    void addGroupByFunction(DataType type);

    void addBetweenFunction(DataType op1, DataType op2, DataType op3);

	template<typename T>
	T* allocateRegister(std::string reg, int32_t size);

	void insertComplexPolygon(std::string colName, GPUMemory::GPUPolygon polygon, int32_t size);
	std::tuple<GPUMemory::GPUPolygon, int32_t> findComplexPolygon(std::string colName);
	NativeGeoPoint* insertConstPointGpu(ColmnarDB::Types::Point& point);

    template<typename T>
    friend int32_t loadConst(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t loadCol(GpuSqlDispatcher &dispatcher);

    friend int32_t loadReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t retConst(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t retCol(GpuSqlDispatcher &dispatcher);

    friend int32_t retReg(GpuSqlDispatcher &dispatcher);

    friend int32_t fil(GpuSqlDispatcher &dispatcher);

    friend int32_t done(GpuSqlDispatcher &dispatcher);

	//// FILTERS WITH FUNCTORS

	template<typename OP, typename T, typename U>
	friend int32_t filterColConst(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t filterConstCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t filterColCol (GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t filterConstConst(GpuSqlDispatcher &dispatcher);

	template<typename OP>
	friend int32_t filterRegReg(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t logicalColConst(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t logicalConstCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t logicalColCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t logicalConstConst(GpuSqlDispatcher &dispatcher);

	template<typename OP>
	friend int32_t logicalRegReg(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t arithmeticColConst(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t arithmeticConstCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t arithmeticColCol(GpuSqlDispatcher &dispatcher);

	template<typename OP, typename T, typename U>
	friend int32_t arithmeticConstConst(GpuSqlDispatcher &dispatcher);

	template<typename OP>
	friend int32_t arithmeticRegReg(GpuSqlDispatcher &dispatcher);

	////

    template<typename T, typename U>
    friend int32_t containsColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t containsConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t containsColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t containsConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t containsRegReg(GpuSqlDispatcher &dispatcher);

    int32_t between();

    template<typename T>
    friend int32_t logicalNotCol(GpuSqlDispatcher &dispatcher);


    template<typename T>
    friend int32_t logicalNotConst(GpuSqlDispatcher &dispatcher);

    friend int32_t logicalNotReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t minusCol(GpuSqlDispatcher &dispatcher);


    template<typename T>
    friend int32_t minusConst(GpuSqlDispatcher &dispatcher);

    friend int32_t minusReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t minCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t minConst(GpuSqlDispatcher &dispatcher);

    friend int32_t minReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t maxCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t maxConst(GpuSqlDispatcher &dispatcher);

    friend int32_t maxReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t sumCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t sumConst(GpuSqlDispatcher &dispatcher);

    friend int32_t sumReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t countCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t countConst(GpuSqlDispatcher &dispatcher);

    friend int32_t countReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t avgCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t avgConst(GpuSqlDispatcher &dispatcher);

    friend int32_t avgReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t groupByCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend int32_t groupByConst(GpuSqlDispatcher &dispatcher);

    friend int32_t groupByReg(GpuSqlDispatcher &dispatcher);

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

	template<typename OP, typename T, typename U>
	friend int32_t invalidOperandTypesErrorHandlerRegCol(GpuSqlDispatcher &dispatcher);


	template<typename OP, typename T, typename U>
	friend int32_t invalidOperandTypesErrorHandlerRegConst(GpuSqlDispatcher &dispatcher);


	template<typename OP, typename T, typename U>
	friend int32_t invalidOperandTypesErrorHandlerColReg(GpuSqlDispatcher &dispatcher);


	template<typename OP, typename T, typename U>
	friend int32_t invalidOperandTypesErrorHandlerConstReg(GpuSqlDispatcher &dispatcher);

	////


    template<typename T, typename U>
    friend int32_t invalidOperandTypesErrorHandlerColReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t invalidOperandTypesErrorHandlerConstReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t invalidOperandTypesErrorHandlerRegCol(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t invalidOperandTypesErrorHandlerRegConst(GpuSqlDispatcher &dispatcher);

	friend int32_t invalidOperandTypesErrorHandlerRegReg(GpuSqlDispatcher &dispatcher);

	template<typename T>
	friend int32_t invalidOperandTypesErrorHandlerCol(GpuSqlDispatcher &dispatcher);

	template<typename T>
	friend int32_t invalidOperandTypesErrorHandlerConst(GpuSqlDispatcher &dispatcher);

	template<typename T>
	friend int32_t invalidOperandTypesErrorHandlerReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    void addArgument(T argument)
    {
        arguments.insert<T>(argument);
    }
};

#endif //DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H
