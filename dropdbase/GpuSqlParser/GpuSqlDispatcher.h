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
#include "../DataType.h"
#include "../Database.h"
#include "../Table.h"
#include "../ColumnBase.h"
#include "../BlockBase.h"
#include "../QueryEngine/GPUCore/GPUFilter.cuh"
#include "../QueryEngine/GPUCore/GPUFiltersAll.cuh"
#include "../QueryEngine/GPUCore/GPUFilterConst.cuh"
#include "../QueryEngine/GPUCore/GPUArithmetic.cuh"
#include "../QueryEngine/GPUCore/GPUArithmeticConst.cuh"
#include "../QueryEngine/GPUCore/GPULogic.cuh"
#include "../QueryEngine/GPUCore/GPULogicConst.cuh"
#include "../QueryEngine/GPUCore/GPUAggregation.cuh"
#include "../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../QueryEngine/GPUCore/GPUReconstruct.cuh"

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

////

template<typename T, typename U>
int32_t greaterColConst(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t greaterConstCol(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t greaterColCol(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t greaterConstConst(GpuSqlDispatcher &dispatcher);

int32_t greaterRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t lessColConst(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t lessConstCol(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t lessColCol(GpuSqlDispatcher &dispatcher);

int32_t lessRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t lessConstConst(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t greaterEqualColConst(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t greaterEqualConstCol(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t greaterEqualColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t greaterEqualConstConst(GpuSqlDispatcher &dispatcher);

int32_t greaterEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t lessEqualColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t lessEqualConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t lessEqualColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t lessEqualConstConst(GpuSqlDispatcher &dispatcher);


int32_t lessEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t equalColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t equalConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t equalColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t equalConstConst(GpuSqlDispatcher &dispatcher);


int32_t equalRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t notEqualColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t notEqualConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t notEqualColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t notEqualConstConst(GpuSqlDispatcher &dispatcher);


int32_t notEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t logicalAndColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t logicalAndConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t logicalAndColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t logicalAndConstConst(GpuSqlDispatcher &dispatcher);


int32_t logicalAndRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t logicalOrColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t logicalOrConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t logicalOrColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t logicalOrConstConst(GpuSqlDispatcher &dispatcher);


int32_t logicalOrRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t mulColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t mulConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t mulColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t mulConstConst(GpuSqlDispatcher &dispatcher);


int32_t mulRegReg(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t divColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t divConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t divColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t divConstConst(GpuSqlDispatcher &dispatcher);


int32_t divRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t addColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t addConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t addColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t addConstConst(GpuSqlDispatcher &dispatcher);


int32_t addRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t subColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t subConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t subColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t subConstConst(GpuSqlDispatcher &dispatcher);


int32_t subRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t modColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t modConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t modColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t modConstConst(GpuSqlDispatcher &dispatcher);


int32_t modRegReg(GpuSqlDispatcher &dispatcher);

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

template<typename OP>
int32_t filterRegReg(GpuSqlDispatcher &dispatcher);

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
	T* allocateRegister(std::string reg, int32_t size)
	{
		T * mask;
		GPUMemory::alloc<T>(&mask, size);
		allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), size) });
		return mask;
	}

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

	////

    template<typename T, typename U>
    friend int32_t greaterColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t greaterConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t greaterColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t greaterConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t greaterRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t lessColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t lessConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t lessColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t lessConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t lessRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t greaterEqualColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t greaterEqualConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t greaterEqualColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t greaterEqualConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t greaterEqualRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t lessEqualColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t lessEqualConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t lessEqualColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t lessEqualConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t lessEqualRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t equalColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t equalConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t equalColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t equalConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t equalRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t notEqualColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t notEqualConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t notEqualColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t notEqualConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t notEqualRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t logicalAndColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t logicalAndConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t logicalAndColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t logicalAndConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t logicalAndRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t logicalOrColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t logicalOrConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t logicalOrColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t logicalOrConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t logicalOrRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t mulColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t mulConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t mulColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t mulConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t mulRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t divColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t divConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t divColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t divConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t divRegReg(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t addColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t addConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t addColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t addConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t addRegReg(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t subColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t subConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t subColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t subConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t subRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend int32_t modColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t modConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t modColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend int32_t modConstConst(GpuSqlDispatcher &dispatcher);

    friend int32_t modRegReg(GpuSqlDispatcher &dispatcher);

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


class GpuSqlDispatcher;

template<typename T>
int32_t loadConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T>
int32_t loadCol(GpuSqlDispatcher &dispatcher)
{
    auto colName = dispatcher.arguments.read<std::string>();
    std::cout << "Load: " << colName << " " << typeid(T).name() << std::endl;

	// split colName to table and column name
	const size_t endOfPolyIdx = colName.find(".");
	const std::string table = colName.substr(0, endOfPolyIdx);
	const std::string column = colName.substr(endOfPolyIdx + 1);

	if (dispatcher.blockIndex >= dispatcher.database->GetTables().at(table).GetColumns().at(column).get()->GetBlockCount())
	{
		return 1;
	}

	auto col = dynamic_cast<const ColumnBase<T>*>(dispatcher.database->GetTables().at(table).GetColumns().at(column).get());
	auto block = dynamic_cast<BlockBase<T>*>(col->GetBlocksList()[0].get());

	T * gpuPointer;
	GPUMemory::alloc<T>(&gpuPointer, block->GetData().size());
	dispatcher.allocatedPointers.insert({colName, std::make_tuple(reinterpret_cast<std::uintptr_t>(gpuPointer), block->GetData().size())});

	GPUMemory::copyHostToDevice(gpuPointer, reinterpret_cast<T*>(block->GetData().data()),
		block->GetData().size());
	return 0;
}

int32_t loadReg(GpuSqlDispatcher &dispatcher);


template<typename T>
int32_t retConst(GpuSqlDispatcher &dispatcher)
{
    T cnst = dispatcher.arguments.read<T>();
    std::cout << "RET: cnst" << typeid(T).name() << std::endl;
	return 0;
}

template<typename T>
int32_t retCol(GpuSqlDispatcher &dispatcher)
{
    auto col = dispatcher.arguments.read<std::string>();
    std::cout << "RetCol: " << col << std::endl;
	std::unique_ptr<T[]> outData (new T[dispatcher.database->GetBlockSize()]);
	//ToDo: Podmienene zapnut podla velkost buffera
	//GPUMemory::hostPin(outData.get(), dispatcher.database->GetBlockSize());
	int32_t outSize;
	std::tuple<uintptr_t, int32_t> ACol = dispatcher.allocatedPointers.at(col);

	GPUReconstruct::reconstructCol(outData.get(), &outSize, reinterpret_cast<T*>(std::get<0>(ACol)), reinterpret_cast<int8_t*>(dispatcher.filter_), std::get<1>(ACol));
	//GPUMemory::hostUnregister(outData.get());
	std::cout << "dataSize: " << outSize << std::endl;
	ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
	insertIntoPayload<T>(payload, outData, outSize);
	ColmnarDB::NetworkClient::Message::QueryResponseMessage partialMessage;
	partialMessage.mutable_payloads()->insert({ col, payload });
	dispatcher.responseMessage.MergeFrom(partialMessage);
	return 0;
}

int32_t retReg(GpuSqlDispatcher &dispatcher);

int32_t fil(GpuSqlDispatcher &dispatcher);

int32_t done(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t filterColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFiltersAll::colConst<OP,T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t filterConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFiltersAll::constCol<OP, U, T>(mask, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t filterColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);

	GPUFiltersAll::colCol<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}


template<typename OP, typename T, typename U>
int32_t filterConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, dispatcher.database->GetBlockSize());
	GPUFiltersAll::constConst<OP, T, U>(mask, constLeft, constRight, dispatcher.database->GetBlockSize());
	return 0;
}

template<typename OP>
int32_t filterRegReg(GpuSqlDispatcher &dispatcher) 
{

}


template<typename T, typename U>
int32_t greaterColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "GtColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilterConst::gt<T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t greaterConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "GtConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilterConst::lt<U, T>(mask, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t greaterColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "GtColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilter::gt<T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename T, typename U>
int32_t greaterConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "GtConstConst: " << constLeft << " " << constRight << " " << reg << std::endl;
	int8_t * mask;
	if (constLeft > constRight)
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(1), dispatcher.database->GetBlockSize());
	}
	else
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(0), dispatcher.database->GetBlockSize());
	}
	dispatcher.allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), dispatcher.database->GetBlockSize())});
	return 0;
}

int32_t greaterRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t lessColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "LtColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilterConst::lt<T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t lessConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "LtConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilterConst::gt<U, T>(mask, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t lessColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "LtColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilter::lt<T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

int32_t lessRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t lessConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "LtConstConst: " << constLeft << " " << constRight << " " << reg << std::endl;
	int8_t * mask;
	if (constLeft < constRight)
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(1), dispatcher.database->GetBlockSize());
	}
	else
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(0), dispatcher.database->GetBlockSize());
	}
	dispatcher.allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), dispatcher.database->GetBlockSize())});
	return 0;
}

template<typename T, typename U>
int32_t greaterEqualColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "GtEqColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilterConst::gtEq<T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t greaterEqualConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "GtEqConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilterConst::ltEq<U, T>(mask, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t greaterEqualColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "GtEqColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilter::gtEq<T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}


template<typename T, typename U>
int32_t greaterEqualConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "GtEqConstConst: " << constLeft << " " << constRight << " " << reg << std::endl;
	int8_t * mask;
	if (constLeft >= constRight)
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(1), dispatcher.database->GetBlockSize());
	}
	else
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(0), dispatcher.database->GetBlockSize());
	}
	dispatcher.allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), dispatcher.database->GetBlockSize())});
	return 0;
}

int32_t greaterEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t lessEqualColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "LtEqColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilterConst::ltEq<T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t lessEqualConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "LtEqConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilterConst::gtEq<U, T>(mask, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t lessEqualColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "LtEqColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilter::ltEq<T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename T, typename U>
int32_t lessEqualConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "LtEqConstConst: " << constLeft << " " << constRight << " " << reg << std::endl;
	int8_t * mask;
	if (constLeft <= constRight)
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(1), dispatcher.database->GetBlockSize());
	}
	else
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(0), dispatcher.database->GetBlockSize());
	}
	dispatcher.allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), dispatcher.database->GetBlockSize())});
	return 0;
}

int32_t lessEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t equalColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "EqColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilterConst::eq<T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t equalConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "EqConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilterConst::eq<U, T>(mask, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t equalColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "EqColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilter::eq<T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename T, typename U>
int32_t equalConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "EqConstConst: " << constLeft << " " << constRight << " " << reg << std::endl;
	int8_t * mask;
	if (constLeft == constRight)
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(1), dispatcher.database->GetBlockSize());
	}
	else
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(0), dispatcher.database->GetBlockSize());
	}
	dispatcher.allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), dispatcher.database->GetBlockSize())});
	return 0;
}

int32_t equalRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t notEqualColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "NonEqColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilterConst::nonEq<T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t notEqualConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "NonEqConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilterConst::nonEq<U, T>(mask, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t notEqualColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "NonEqColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilter::nonEq<T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename T, typename U>
int32_t notEqualConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "NonEqConstConst: " << constLeft << " " << constRight << " " << reg << std::endl;
	int8_t * mask;
	if (constLeft != constRight)
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(1), dispatcher.database->GetBlockSize());
	}
	else
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(0), dispatcher.database->GetBlockSize());
	}
	dispatcher.allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), dispatcher.database->GetBlockSize())});
	return 0;
}

int32_t notEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t logicalAndColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "AndColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogicConst::and<int8_t, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t logicalAndConstCol(GpuSqlDispatcher &dispatcher)
{
	T cnst = dispatcher.arguments.read<T>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "AndConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogicConst::and<int8_t, U, T>(result, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t logicalAndColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "AndColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogic::and<int8_t, T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename T, typename U>
int32_t logicalAndConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "AndConstConst: " << constLeft << " " << constRight << " " << reg << std::endl;
	int8_t * mask;
	if (constLeft && constRight)
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(1), dispatcher.database->GetBlockSize());
	}
	else
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(0), dispatcher.database->GetBlockSize());
	}
	dispatcher.allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), dispatcher.database->GetBlockSize())});
	return 0;
}

int32_t logicalAndRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t logicalOrColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "OrColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogicConst::or<int8_t, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t logicalOrConstCol(GpuSqlDispatcher &dispatcher)
{
	T cnst = dispatcher.arguments.read<T>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "OrConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogicConst::or<int8_t, U, T>(result, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t logicalOrColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "OrColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogic::or<int8_t, T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename T, typename U>
int32_t logicalOrConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "OrConstConst: " << constLeft << " " << constRight << " " << reg << std::endl;
	int8_t * mask;
	if (constLeft || constRight)
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(1), dispatcher.database->GetBlockSize());
	}
	else
	{
		GPUMemory::allocAndSet<int8_t>(&mask, static_cast<int8_t>(0), dispatcher.database->GetBlockSize());
	}
	dispatcher.allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), dispatcher.database->GetBlockSize())});
	return 0;
}

int32_t logicalOrRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t mulColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "MulColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmeticConst::multiplication<T, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t mulConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "MulConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	U * result = dispatcher.allocateRegister<U>(reg, retSize);
	GPUArithmeticConst::multiplication<U, U, T>(result,reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t mulColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "MulColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmetic::multiplication<T, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename T, typename U>
int32_t mulConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t mulRegReg(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
int32_t divColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "DivColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmeticConst::division<T, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t divConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "DivConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	U * result = dispatcher.allocateRegister<U>(reg, retSize);
	GPUArithmeticConst::division<U, U, T>(result, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t divColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "DivColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmetic::division<T, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename T, typename U>
int32_t divConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t divRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t addColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "AddColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmeticConst::plus<T, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t addConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "AddConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	U * result = dispatcher.allocateRegister<U>(reg, retSize);
	GPUArithmeticConst::plus<U, U, T>(result, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t addColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "AddColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmetic::plus<T, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename T, typename U>
int32_t addConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t addRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t subColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "SubColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmeticConst::minus<T, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t subConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "SubConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	U * result = dispatcher.allocateRegister<U>(reg, retSize);
	GPUArithmeticConst::multiplication<U, U, T>(result, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t subColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "SubColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmetic::minus<T, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename T, typename U>
int32_t subConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t subRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t modColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "ModColConst: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmeticConst::modulo<T, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t modConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "ModConstCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	U * result = dispatcher.allocateRegister<U>(reg, retSize);
	GPUArithmeticConst::modulo<U, U, T>(result, reinterpret_cast<U*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename T, typename U>
int32_t modColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "ModColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmetic::modulo<T, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename T, typename U>
int32_t modConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t modRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t containsColConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t containsConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t containsColCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t containsConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t containsRegReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t logicalNotCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "NotCol: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogic::not<int8_t, T>(mask, reinterpret_cast<T*>(std::get<0>(column)), retSize);
	return 0;
}

template<typename T>
int32_t logicalNotConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t logicalNotReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t minusCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "MinusCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmeticConst::minus<T, T, T>(result, static_cast<T>(0), reinterpret_cast<T*>(std::get<0>(column)), retSize);
	return 0;
}

template<typename T>
int32_t minusConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t minusReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t minCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "MinCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);

	T * result = dispatcher.allocateRegister<T>(reg, 1);
	GPUAggregation::min<T>(result, reinterpret_cast<T*>(std::get<0>(column)), std::get<1>(column));
	return 0;
}

template<typename T>
int32_t minConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t minReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t maxCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "MaxCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);

	T * result = dispatcher.allocateRegister<T>(reg, 1);
	GPUAggregation::max<T>(result, reinterpret_cast<T*>(std::get<0>(column)), std::get<1>(column));
	return 0;
}

template<typename T>
int32_t maxConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}


int32_t maxReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t sumCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "SumCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);

	T * result = dispatcher.allocateRegister<T>(reg, 1);
	GPUAggregation::sum<T>(result, reinterpret_cast<T*>(std::get<0>(column)), std::get<1>(column));
	return 0;
}

template<typename T>
int32_t sumConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t sumReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t countCol(GpuSqlDispatcher &dispatcher)
{
	//TODO: CPU count
	return 0;
}

template<typename T>
int32_t countConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t countReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t avgCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "AvgCol: " << colName << " const " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);

	T * result = dispatcher.allocateRegister<T>(reg, 1);
	GPUAggregation::avg<T>(result, reinterpret_cast<T*>(std::get<0>(column)), std::get<1>(column));
	return 0;
}

template<typename T>
int32_t avgConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t avgReg(GpuSqlDispatcher &dispatcher);

template<typename T>
int32_t groupByConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T>
int32_t groupByCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t groupByReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}


template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}


//// FUNCTOR ERROR HANDLERS

template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}


template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}


template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}


template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerRegCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}


template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerRegConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}


template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColReg(GpuSqlDispatcher &dispatcher)
{
	return 0;
}


template<typename OP, typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstReg(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

////


template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerRegCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerRegConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerColReg(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t invalidOperandTypesErrorHandlerConstReg(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T>
int32_t invalidOperandTypesErrorHandlerCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T>
int32_t invalidOperandTypesErrorHandlerConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T>
int32_t invalidOperandTypesErrorHandlerReg(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

#endif //DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H
