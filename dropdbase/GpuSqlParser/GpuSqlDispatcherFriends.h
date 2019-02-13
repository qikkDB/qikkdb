#pragma once
#include <cstdint>
#include "GpuSqlDispatcher.h"
#include "../QueryEngine/GPUCore/GPUFilter.cuh"
#include "../QueryEngine/GPUCore/GPUArithmetic.cuh"
#include "../QueryEngine/GPUCore/GPULogic.cuh"
#include "../QueryEngine/GPUCore/GPUAggregation.cuh"
#include "../QueryEngine/GPUCore/GPUPolygon.cuh"
#include "../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../QueryEngine/GPUCore/IGroupBy.h"
#include "../QueryEngine/GPUCore/GPUGroupBy.cuh"
#include "../QueryEngine/GPUCore/AggregationFunctions.cuh"
#include "../Configuration.h"


template<typename T>
T* GpuSqlDispatcher::allocateRegister(const std::string& reg, int32_t size)
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
	const int32_t blockCount = dispatcher.database->GetTables().at(table).GetColumns().at(column).get()->GetBlockCount();

	if (dispatcher.blockIndex == blockCount - 1)
	{
		dispatcher.isLastBlock = true;
	}

	auto col = dynamic_cast<const ColumnBase<T>*>(dispatcher.database->GetTables().at(table).GetColumns().at(column).get());
	auto block = dynamic_cast<BlockBase<T>*>(col->GetBlocksList()[dispatcher.blockIndex].get());

	T * gpuPointer;

	gpuPointer = dispatcher.allocateRegister<T>(colName, block->GetData().size());

	GPUMemory::copyHostToDevice(gpuPointer, reinterpret_cast<T*>(block->GetData().data()), block->GetData().size());
	return 0;
}

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
	int32_t outSize;

	const size_t endOfPolyIdx = col.find(".");
	const std::string table = col.substr(0, endOfPolyIdx);
	const std::string column = col.substr(endOfPolyIdx + 1);

	if (dispatcher.usingGroupBy)
	{
		if (dispatcher.isLastBlock)
		{
			if (dispatcher.groupByColumns.find(col) != dispatcher.groupByColumns.end())
			{
				std::tuple<uintptr_t, int32_t> keyCol = dispatcher.allocatedPointers.at(col + "_keys");
				outSize = std::get<1>(keyCol);
				std::unique_ptr<T[]> outData(new T[outSize]);
				GPUMemory::copyDeviceToHost(outData.get(), reinterpret_cast<T*>(std::get<0>(keyCol)), outSize);

				ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
				insertIntoPayload<T>(payload, outData, outSize);
				ColmnarDB::NetworkClient::Message::QueryResponseMessage partialMessage;
				dispatcher.mergePayloadToResponse(col, payload);
			}
			else
			{
				std::tuple<uintptr_t, int32_t> valueCol = dispatcher.allocatedPointers.at(col);
				outSize = std::get<1>(valueCol);
				std::unique_ptr<T[]> outData(new T[outSize]);
				GPUMemory::copyDeviceToHost(outData.get(), reinterpret_cast<T*>(std::get<0>(valueCol)), outSize);

				ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
				insertIntoPayload<T>(payload, outData, outSize);
				dispatcher.mergePayloadToResponse(col, payload);
			}
		}
	}
	else
	{
		std::unique_ptr<T[]> outData(new T[dispatcher.database->GetBlockSize()]);
		//ToDo: Podmienene zapnut podla velkost buffera
		//GPUMemory::hostPin(outData.get(), dispatcher.database->GetBlockSize());
		std::tuple<uintptr_t, int32_t> ACol = dispatcher.allocatedPointers.at(col);
		GPUReconstruct::reconstructCol(outData.get(), &outSize, reinterpret_cast<T*>(std::get<0>(ACol)), reinterpret_cast<int8_t*>(dispatcher.filter_), std::get<1>(ACol));
		//GPUMemory::hostUnregister(outData.get());
		std::cout << "dataSize: " << outSize << std::endl;
		ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
		insertIntoPayload<T>(payload, outData, outSize);
		dispatcher.mergePayloadToResponse(col, payload);
	}
	return 0;
}

int32_t fil(GpuSqlDispatcher &dispatcher);

int32_t done(GpuSqlDispatcher &dispatcher);

template<typename OP, typename T, typename U>
int32_t filterColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::cout << "Filter: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUFilter::colConst<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	dispatcher.freeColumnIfRegister(colName);
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
	dispatcher.freeColumnIfRegister(colName);
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
	dispatcher.freeColumnIfRegister(colNameRight);
	dispatcher.freeColumnIfRegister(colNameLeft);
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
	dispatcher.freeColumnIfRegister(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t logicalConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogic::constCol<OP, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
	dispatcher.freeColumnIfRegister(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t logicalColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::cout << "Filter: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogic::colCol<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	dispatcher.freeColumnIfRegister(colNameRight);
	dispatcher.freeColumnIfRegister(colNameLeft);
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

template<typename OP, typename T, typename U>
int32_t arithmeticColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::cout << "ArithmeticColConst: " << colName << " " << reg << std::endl;

	if (dispatcher.groupByColumns.find(colName) != dispatcher.groupByColumns.end())
	{
		if (dispatcher.isLastBlock)
		{
			std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName + "_keys");
			int32_t retSize = std::get<1>(column);
			T * result = dispatcher.allocateRegister<T>(reg + "_keys", retSize);
			GPUArithmetic::colConst<OP, T, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
			dispatcher.groupByColumns.insert(reg);
		}
	}
	else if (dispatcher.isLastBlock || !dispatcher.usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
		int32_t retSize = std::get<1>(column);
		T * result = dispatcher.allocateRegister<T>(reg, retSize);
		GPUArithmetic::colConst<OP, T, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	}
	dispatcher.freeColumnIfRegister(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t arithmeticConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::cout << "ArithmeticConstCol: " << colName << " " << reg << std::endl;

	if (dispatcher.groupByColumns.find(colName) != dispatcher.groupByColumns.end())
	{
		if (dispatcher.isLastBlock)
		{
			std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName + "_keys");
			int32_t retSize = std::get<1>(column);
			U * result = dispatcher.allocateRegister<U>(reg + "_keys", retSize);
			GPUArithmetic::constCol<OP, U, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
			dispatcher.groupByColumns.insert(reg);
		}
	}
	else if (dispatcher.isLastBlock || !dispatcher.usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
		int32_t retSize = std::get<1>(column);
		U * result = dispatcher.allocateRegister<U>(reg, retSize);
		GPUArithmetic::constCol<OP, U, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
	}
	dispatcher.freeColumnIfRegister(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t arithmeticColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::cout << "ArithmeticColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	if (dispatcher.groupByColumns.find(colNameRight) != dispatcher.groupByColumns.end())
	{
		if (dispatcher.isLastBlock)
		{
			std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight + "_keys");
			std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
			int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

			T * result = dispatcher.allocateRegister<T>(reg + "_keys", retSize);
			GPUArithmetic::colCol<OP, T, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
			dispatcher.groupByColumns.insert(reg);
		}
	}
	else if (dispatcher.groupByColumns.find(colNameLeft) != dispatcher.groupByColumns.end())
	{
		if (dispatcher.isLastBlock)
		{
			std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
			std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft + "_keys");
			int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

			T * result = dispatcher.allocateRegister<T>(reg + "_keys", retSize);
			GPUArithmetic::colCol<OP, T, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
			dispatcher.groupByColumns.insert(reg);
		}
	}
	else if (dispatcher.isLastBlock || !dispatcher.usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t> columnRight = dispatcher.allocatedPointers.at(colNameRight);
		std::tuple<uintptr_t, int32_t> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
		int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

		T * result = dispatcher.allocateRegister<T>(reg, retSize);
		GPUArithmetic::colCol<OP, T, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	}
	dispatcher.freeColumnIfRegister(colNameLeft);
	dispatcher.freeColumnIfRegister(colNameRight);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t arithmeticConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::cout << "ArithmeticConstConst: " << reg << std::endl;

	int32_t retSize = 1;

	T * result = dispatcher.allocateRegister<T>(reg, retSize);
	GPUArithmetic::constConst<OP, T, T, U>(result, constLeft, constRight, retSize);
	return 0;
}

template<typename T, typename U>
int32_t containsColConst(GpuSqlDispatcher &dispatcher)
{
	auto constWkt = dispatcher.arguments.read<std::string>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();


	std::cout << "ContainsColConst: " + colName << " " << constWkt << " " << reg << std::endl;

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
	auto colName = dispatcher.arguments.read<std::string>();
	auto constWkt = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();


	std::cout << "ContainsConstCol: " + constWkt << " " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> columnPoint = dispatcher.allocatedPointers.at(colName);
	ColmnarDB::Types::ComplexPolygon polygonConst = ComplexPolygonFactory::FromWkt(constWkt);
	GPUMemory::GPUPolygon gpuPolygon = dispatcher.insertConstPolygonGpu(polygonConst);

	int32_t retSize = std::get<1>(columnPoint);

	int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUPolygon::contains(result, reinterpret_cast<NativeGeoPoint*>(std::get<0>(columnPoint)), reinterpret_cast<NativeGeoPoint*>(gpuPolygon.polyPoints), reinterpret_cast<int32_t*>(gpuPolygon.polyIdx), reinterpret_cast<int32_t*>(gpuPolygon.polyCount), reinterpret_cast<int32_t*>(gpuPolygon.pointIdx), reinterpret_cast<int32_t*>(gpuPolygon.pointCount), retSize, 1);
	return 0;
}

template<typename T, typename U>
int32_t containsColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNamePoint = dispatcher.arguments.read<std::string>();
	auto colNamePolygon = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();


	std::cout << "ContainsColCol: " + colNamePolygon << " " << colNamePoint << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> pointCol = dispatcher.allocatedPointers.at(colNamePoint);
	auto polygonCol = dispatcher.findComplexPolygon(colNamePolygon);
	GPUMemory::GPUPolygon gpuPolygon = std::get<0>(polygonCol);

	int32_t retSize = std::min(std::get<1>(pointCol), std::get<1>(polygonCol));

	int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUPolygon::contains(result, reinterpret_cast<NativeGeoPoint*>(std::get<0>(pointCol)), reinterpret_cast<NativeGeoPoint*>(gpuPolygon.polyPoints), reinterpret_cast<int32_t*>(gpuPolygon.polyIdx), reinterpret_cast<int32_t*>(gpuPolygon.polyCount), reinterpret_cast<int32_t*>(gpuPolygon.pointIdx), reinterpret_cast<int32_t*>(gpuPolygon.pointCount), std::get<1>(pointCol), std::get<1>(polygonCol));
	return 0;
}

template<typename T, typename U>
int32_t containsConstConst(GpuSqlDispatcher &dispatcher)
{
	// TODO : Specialize kernel for all cases.
	auto constPointWkt = dispatcher.arguments.read<std::string>();
	auto constPolygonWkt = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	std::cout << "ContainsConstConst: " + constPolygonWkt << " " << constPointWkt << " " << reg << std::endl;

	ColmnarDB::Types::Point constPoint = PointFactory::FromWkt(constPointWkt);
	ColmnarDB::Types::ComplexPolygon constPolygon = ComplexPolygonFactory::FromWkt(constPolygonWkt);

	NativeGeoPoint *constNativeGeoPoint = dispatcher.insertConstPointGpu(constPoint);
	GPUMemory::GPUPolygon gpuPolygon = dispatcher.insertConstPolygonGpu(constPolygon);

	int32_t retSize = dispatcher.database->GetBlockSize();

	int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPUPolygon::contains(result, constNativeGeoPoint, reinterpret_cast<NativeGeoPoint*>(gpuPolygon.polyPoints), reinterpret_cast<int32_t*>(gpuPolygon.polyIdx), reinterpret_cast<int32_t*>(gpuPolygon.polyCount), reinterpret_cast<int32_t*>(gpuPolygon.pointIdx), reinterpret_cast<int32_t*>(gpuPolygon.pointCount), 1, 1);
	return 0;
}

template<typename T>
int32_t logicalNotCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "NotCol: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
	GPULogic::not_col<int8_t, T>(mask, reinterpret_cast<T*>(std::get<0>(column)), retSize);
	dispatcher.freeColumnIfRegister(colName);
	return 0;
}

template<typename T>
int32_t logicalNotConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

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

template<typename OP, typename T, typename U>
int32_t aggregationColCol(GpuSqlDispatcher &dispatcher) 
{
	auto colTableName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "AggColCol: " << colTableName << " " << reg << std::endl;
	
	std::tuple<uintptr_t, int32_t>& column = dispatcher.allocatedPointers.at(colTableName);
	int32_t reconstructOutSize;
	T* reconstructOutReg; 
	GPUMemory::alloc(&reconstructOutReg, std::get<1>(column));
	GPUReconstruct::reconstructColKeep<T>(reconstructOutReg, &reconstructOutSize, reinterpret_cast<T*>(std::get<0>(column)), reinterpret_cast<int8_t*>(dispatcher.filter_), std::get<1>(column));

	GPUMemory::free(reinterpret_cast<void*>(std::get<0>(column)));
	std::get<0>(column) = reinterpret_cast<uintptr_t>(reconstructOutReg);
	std::get<1>(column) = reconstructOutSize;

	const size_t endOfPolyIdx = colTableName.find(".");
	const std::string table = colTableName.substr(0, endOfPolyIdx);
	const std::string columnName = colTableName.substr(endOfPolyIdx + 1);
	
	if (dispatcher.usingGroupBy)
	{
		std::cout << "Using group by" << std::endl;

		//TODO void param
		if (dispatcher.groupByTable == nullptr) 
		{
			dispatcher.groupByTable = std::make_unique<GPUGroupBy<OP,T,U,T>>(Configuration::GetInstance().GetGroupByBuckets());
		}

		std::string groupByColumnName = *(dispatcher.groupByColumns.begin());
		std::tuple<uintptr_t, int32_t> groupByColumn = dispatcher.allocatedPointers.at(groupByColumnName);
		
		int32_t dataSize = std::min(std::get<1>(groupByColumn), std::get<1>(column));

		reinterpret_cast<GPUGroupBy<OP, T, U, T>*>(dispatcher.groupByTable.get())->groupBy(reinterpret_cast<U*>(std::get<0>(groupByColumn)), reinterpret_cast<T*>(std::get<0>(column)), dataSize);

		// If last block was processed, reconstruct group by table
		if (dispatcher.isLastBlock)
		{
			int32_t outSize;
			U* outKeys = dispatcher.allocateRegister<U>(groupByColumnName + "_keys", Configuration::GetInstance().GetGroupByBuckets());
			T* outValues = dispatcher.allocateRegister<T>(reg, Configuration::GetInstance().GetGroupByBuckets());
			reinterpret_cast<GPUGroupBy<OP, T, U, T>*>(dispatcher.groupByTable.get())->getResults(outKeys, outValues, &outSize);
			std::get<1>(dispatcher.allocatedPointers.at(groupByColumnName + "_keys")) = outSize;
			std::get<1>(dispatcher.allocatedPointers.at(reg)) = outSize;
		}
	}
	else
	{
		T * result = dispatcher.allocateRegister<T>(reg, 1);
		GPUAggregation::col<OP, T>(result, reinterpret_cast<T*>(std::get<0>(column)), std::get<1>(column));
	}
	dispatcher.freeColumnIfRegister(colTableName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t aggregationColConst(GpuSqlDispatcher &dispatcher)
{
	std::cout << "AggColConst" << std::endl;
	return 0;
}

template<typename OP, typename T, typename U>
int32_t aggregationConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "AggConstCol: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t>& column = dispatcher.allocatedPointers.at(colName);

	int32_t reconstructOutSize;
	T* reconstructOutReg;
	GPUMemory::alloc(&reconstructOutReg, std::get<1>(column));
	GPUReconstruct::reconstructColKeep<T>(reconstructOutReg, &reconstructOutSize, reinterpret_cast<T*>(std::get<0>(column)), reinterpret_cast<int8_t*>(dispatcher.filter_), std::get<1>(column));

	GPUMemory::free(reinterpret_cast<void*>(std::get<0>(column)));
	std::get<0>(column) = reinterpret_cast<uintptr_t>(reconstructOutReg);
	std::get<1>(column) = reconstructOutSize;

	T * result = dispatcher.allocateRegister<T>(reg, 1);
	GPUAggregation::col<OP, T>(result, reinterpret_cast<T*>(std::get<0>(column)), std::get<1>(column));
	dispatcher.freeColumnIfRegister(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t aggregationConstConst(GpuSqlDispatcher &dispatcher)
{
	std::cout << "AggConstConst" << std::endl;
	return 0;
}


template<typename T>
int32_t groupByCol(GpuSqlDispatcher &dispatcher)
{
	std::string column = dispatcher.arguments.read<std::string>();
	std::cout << "GroupBy: " << column << std::endl;
	if (dispatcher.groupByColumns.find(column) == dispatcher.groupByColumns.end())
	{
		dispatcher.groupByColumns.insert(column);
	}
	dispatcher.usingGroupBy = true;
	return 0;
}

template<typename T>
int32_t groupByConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename T>
int32_t insertInto(GpuSqlDispatcher &dispatcher)
{
	std::string table = dispatcher.arguments.read<std::string>();
	std::string column = dispatcher.arguments.read<std::string>();
	bool isReferencedColumn = dispatcher.arguments.read<bool>();

	if (isReferencedColumn)
	{
		T args = dispatcher.arguments.read<T>();

		dynamic_cast<ColumnBase<T>*>(dispatcher.database->GetTables().at(table).GetColumns().at(column).get())->InsertData({args});
	}
	else
	{
		dynamic_cast<ColumnBase<T>*>(dispatcher.database->GetTables().at(table).GetColumns().at(column).get())->InsertNullData(1);
	}
	return 0;
}

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

template<typename OP, typename T>
int32_t invalidOperandTypesErrorHandlerCol(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

template<typename OP, typename T>
int32_t invalidOperandTypesErrorHandlerConst(GpuSqlDispatcher &dispatcher)
{
	return 0;
}

////

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