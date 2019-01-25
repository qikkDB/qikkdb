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
#include "../QueryEngine/GPUCore/GPUFilterConst.cuh"
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
	std::unordered_map<std::string, std::uintptr_t> columnPointers;
	std::unordered_map<std::string, std::uintptr_t> registerPointers;
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

    ColmnarDB::NetworkClient::Message::QueryResponseMessage execute();

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

	T * gpuPointer;
	GPUMemory::alloc<T>(&gpuPointer, dispatcher.database->GetBlockSize());
	dispatcher.columnPointers.insert({colName, reinterpret_cast<std::uintptr_t>(gpuPointer)});
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
	GPUMemory::copyHostToDevice(gpuPointer, reinterpret_cast<T*>(block->GetData().data()),
		dispatcher.database->GetBlockSize());
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
    std::cout << "RET: " << col << std::endl;
	std::unique_ptr<T[]> outData (new T[dispatcher.database->GetBlockSize()]);
	//ToDo: Podmienene zapnut podla velkost buffera
	//GPUMemory::hostPin(outData.get(), dispatcher.database->GetBlockSize());
	int32_t outSize;
	T * ACol = reinterpret_cast<T*>(dispatcher.columnPointers.at(col));
	GPUReconstruct::reconstructCol(outData.get(), &outSize, ACol, reinterpret_cast<int8_t*>(dispatcher.filter_), dispatcher.database->GetBlockSize());
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

template<typename T, typename U>
int32_t greaterColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "GtColConst: " << colName << " const " << reg << std::endl;
	int8_t * mask;
	GPUMemory::alloc<int8_t>(&mask, dispatcher.database->GetBlockSize());
	dispatcher.registerPointers.insert({ reg, reinterpret_cast<std::uintptr_t>(mask) });
	GPUFilterConst::gt<T, U>(mask, reinterpret_cast<T*>(dispatcher.columnPointers.at(colName)), cnst, dispatcher.database->GetBlockSize());
	return 0;
}

template<typename T, typename U>
int32_t greaterConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t greaterColCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t greaterConstConst(GpuSqlDispatcher &dispatcher)
{
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
	int8_t * mask;
	GPUMemory::alloc<int8_t>(&mask, dispatcher.database->GetBlockSize());
	dispatcher.registerPointers.insert({ reg, reinterpret_cast<std::uintptr_t>(mask) });
	GPUFilterConst::lt<T, U>(mask, reinterpret_cast<T*>(dispatcher.columnPointers.at(colName)), cnst, dispatcher.database->GetBlockSize());
	return 0;
}

template<typename T, typename U>
int32_t lessConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	U cnst = dispatcher.arguments.read<U>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "LtConstCol: " << colName << " const " << reg << std::endl;
	int8_t * mask;
	GPUMemory::alloc<int8_t>(&mask, dispatcher.database->GetBlockSize());
	dispatcher.registerPointers.insert({ reg, reinterpret_cast<std::uintptr_t>(mask) });
	GPUFilterConst::lt<T, U>(mask, reinterpret_cast<T*>(dispatcher.columnPointers.at(colName)), cnst, dispatcher.database->GetBlockSize());
	return 0;
}

template<typename T, typename U>
int32_t lessColCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t lessRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t lessConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t greaterEqualColConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t greaterEqualConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t greaterEqualColCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}


template<typename T, typename U>
int32_t greaterEqualConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t greaterEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t lessEqualColConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t lessEqualConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t lessEqualColCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t lessEqualConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t lessEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t equalColConst(GpuSqlDispatcher &dispatcher)
{
    U cnst = dispatcher.arguments.read<U>();
    auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
    std::cout << "EqualColConst: " << colName << " const " << reg <<  std::endl;
	int8_t * mask;
	GPUMemory::alloc<int8_t>(&mask, dispatcher.database->GetBlockSize());
	dispatcher.registerPointers.insert({reg, reinterpret_cast<std::uintptr_t>(mask)});
	GPUFilterConst::eq<T, U>(mask, reinterpret_cast<T*>(dispatcher.columnPointers.at(colName)), cnst, dispatcher.database->GetBlockSize());
	return 0;
}

template<typename T, typename U>
int32_t equalConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	U cnst = dispatcher.arguments.read<U>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "EqualConstCol: " << colName << " const " << reg << std::endl;
	int8_t * mask;
	GPUMemory::alloc<int8_t>(&mask, dispatcher.database->GetBlockSize());
	dispatcher.registerPointers.insert({ reg, reinterpret_cast<std::uintptr_t>(mask) });
	GPUFilterConst::eq<T, U>(mask, reinterpret_cast<T*>(dispatcher.columnPointers.at(colName)), cnst, dispatcher.database->GetBlockSize());
	return 0;
}

template<typename T, typename U>
int32_t equalColCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t equalConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t equalRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t notEqualColConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t notEqualConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t notEqualColCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t notEqualConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t notEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t logicalAndColConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t logicalAndConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t logicalAndColCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t logicalAndConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t logicalAndRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t logicalOrColConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t logicalOrConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t logicalOrColCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t logicalOrConstConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

int32_t logicalOrRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
int32_t mulColConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t mulConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t mulColCol(GpuSqlDispatcher &dispatcher)
{
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
	return 0;
}

template<typename T, typename U>
int32_t divConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t divColCol(GpuSqlDispatcher &dispatcher)
{
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
	return 0;
}

template<typename T, typename U>
int32_t addConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t addColCol(GpuSqlDispatcher &dispatcher)
{
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
	return 0;
}

template<typename T, typename U>
int32_t subConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t subColCol(GpuSqlDispatcher &dispatcher)
{
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
	return 0;
}

template<typename T, typename U>
int32_t modConstCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T, typename U>
int32_t modColCol(GpuSqlDispatcher &dispatcher)
{
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
