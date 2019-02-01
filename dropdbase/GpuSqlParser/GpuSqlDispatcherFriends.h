#pragma once
#include <cstdint>
#include "GpuSqlDispatcher.h"
#include "../QueryEngine/GPUCore/GPUFilter.cuh"
#include "../QueryEngine/GPUCore/GPUArithmetic.cuh"
#include "../QueryEngine/GPUCore/GPULogic.cuh"
#include "../QueryEngine/GPUCore/GPULogicConst.cuh"
#include "../QueryEngine/GPUCore/GPUAggregation.cuh"
#include "../QueryEngine/GPUCore/GPUPolygon.cuh"
#include "../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../QueryEngine/GPUCore/GPUReconstruct.cuh"
	

template<typename T>
T* GpuSqlDispatcher::allocateRegister(std::string reg, int32_t size)
{
	T * mask;
	GPUMemory::alloc<T>(&mask, size);
	allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), size) });
	return mask;
}

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
	auto block = dynamic_cast<BlockBase<T>*>(col->GetBlocksList()[dispatcher.blockIndex].get());

	T * gpuPointer;

	gpuPointer = dispatcher.allocateRegister<T>(colName, block->GetData().size());

	GPUMemory::copyHostToDevice(gpuPointer, reinterpret_cast<T*>(block->GetData().data()), block->GetData().size());
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
	std::unique_ptr<T[]> outData(new T[dispatcher.database->GetBlockSize()]);
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
	GPUFilter::colConst<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
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
	GPUArithmetic::colConst<OP, T, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
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
	auto constWkt = dispatcher.arguments.read<std::string>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();


	std::cout << "Contains: " + colName << " " << constWkt << " " << reg << std::endl;

	auto polygonCol = dispatcher.findComplexPolygon(colName);
	ColmnarDB::Types::Point pointConst = PointFactory::FromWkt(constWkt);

	GPUMemory::GPUPolygon polygons = std::get<0>(polygonCol);
	NativeGeoPoint* pointConstPtr = dispatcher.insertConstPointGpu(pointConst);
	int32_t retSize = std::get<1>(polygonCol);

	int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUPolygon::contains(result, pointConstPtr, reinterpret_cast<NativeGeoPoint*>(polygons.polyPoints), reinterpret_cast<int32_t*>(polygons.polyIdx), reinterpret_cast<int32_t*>(polygons.polyCount), reinterpret_cast<int32_t*>(polygons.pointIdx), reinterpret_cast<int32_t*>(polygons.pointCount), 1, retSize);
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