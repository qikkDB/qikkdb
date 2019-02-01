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
#include "../QueryEngine/GPUCore/GPUArithmetic.cuh"
#include "../QueryEngine/GPUCore/GPULogic.cuh"
#include "../QueryEngine/GPUCore/GPULogicConst.cuh"
#include "../QueryEngine/GPUCore/GPUAggregation.cuh"
#include "../QueryEngine/GPUCore/GPUMemory.cuh"
#ifdef __CUDACC__
#include "../QueryEngine/GPUCore/GPUReconstruct.cuh"
#endif

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
	GPUFilter::colConst<OP,T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
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
	GPUFilter::constCol<OP, T, U>(mask, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
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

	GPUFilter::colCol<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}


template<typename OP, typename T, typename U>
int32_t filterConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, dispatcher.database->GetBlockSize());
	GPUFilter::constConst<OP, T, U>(mask, constLeft, constRight, dispatcher.database->GetBlockSize());
	return 0;
}

template<typename OP>
int32_t filterRegReg(GpuSqlDispatcher &dispatcher) 
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);

	GPUFilter::colCol<OP, int8_t, int8_t>(mask, reinterpret_cast<int8_t*>(std::get<0>(columnLeft)), reinterpret_cast<int8_t*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t logicalColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogic::colConst<OP, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t logicalConstCol(GpuSqlDispatcher &dispatcher)
{
	T cnst = dispatcher.arguments.read<T>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogic::constCol<OP, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t logicalColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogic::colCol<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t logicalConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, dispatcher.database->GetBlockSize());
	GPULogic::constConst<OP, T, U>(mask, constLeft, constRight, dispatcher.database->GetBlockSize());
	return 0;
}

template<typename OP>
int32_t logicalRegReg(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogic::colCol<OP, int8_t, int8_t>(mask, reinterpret_cast<int8_t*>(std::get<0>(columnLeft)), reinterpret_cast<int8_t*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t arithmeticColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmetic::colConst<OP,T,T,U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t arithmeticConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	U * result = dispatcher.allocateRegister<U>(reg, retSize);
	GPUArithmetic::constCol<OP, U, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t arithmeticColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmetic::colCol<OP, T, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t arithmeticConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	int32_t retSize = 1;

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmetic::constConst<OP, T, T, U>(result, constLeft, constRight, retSize);
	return 0;
}

template<typename OP>
int32_t arithmeticRegReg(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUArithmetic::colCol<OP, int8_t, int8_t, int8_t>(result, reinterpret_cast<int8_t*>(std::get<0>(columnLeft)), reinterpret_cast<int8_t*>(std::get<0>(columnRight)), retSize);
	return 0;
}

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
