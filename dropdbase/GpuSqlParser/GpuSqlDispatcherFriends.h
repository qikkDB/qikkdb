#pragma once
#include <cstdint>
#include "GpuSqlDispatcher.h"
#include "../QueryEngine/GPUCore/GPUDate.cuh"
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
	allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), size, true)});
	usedRegisterMemory += size * sizeof(T);
	return mask;
}

template<typename T>
void GpuSqlDispatcher::addCachedRegister(const std::string& reg, T* ptr, int32_t size)
{
	allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(ptr), size, false) });
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

	int32_t loadFlag = dispatcher.loadCol<T>(col);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "RetCol: " << col << ", thread: " << dispatcher.dispatcherThreadId << std::endl;

	int32_t outSize;
	const size_t endOfPolyIdx = col.find(".");
	const std::string table = col.substr(0, endOfPolyIdx);
	const std::string column = col.substr(endOfPolyIdx + 1);

	if (dispatcher.usingGroupBy)
	{
		if (dispatcher.isOverallLastBlock)
		{
			if (dispatcher.groupByColumns.find(col) != dispatcher.groupByColumns.end())
			{
				std::tuple<uintptr_t, int32_t, bool> keyCol = dispatcher.allocatedPointers.at(col + "_keys");
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
				std::tuple<uintptr_t, int32_t, bool> valueCol = dispatcher.allocatedPointers.at(col);
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
		std::tuple<uintptr_t, int32_t, bool> ACol = dispatcher.allocatedPointers.at(col);
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

	int32_t loadFlag = dispatcher.loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Filter: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!dispatcher.isRegisterAllocated(reg))
	{
		int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
		GPUFilter::colConst<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	}

	dispatcher.freeColumnIfRegister<T>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t filterConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	int32_t loadFlag = dispatcher.loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Filter: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!dispatcher.isRegisterAllocated(reg))
	{
		int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
		GPUFilter::constCol<OP, T, U>(mask, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
	}

	dispatcher.freeColumnIfRegister<U>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t filterColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	int32_t loadFlag = dispatcher.loadCol<U>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = dispatcher.loadCol<T>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Filter: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t, bool> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	if (!dispatcher.isRegisterAllocated(reg))
	{
		int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
		GPUFilter::colCol<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	}

	dispatcher.freeColumnIfRegister<U>(colNameRight);
	dispatcher.freeColumnIfRegister<T>(colNameLeft);
	return 0;
}


template<typename OP, typename T, typename U>
int32_t filterConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	if (!dispatcher.isRegisterAllocated(reg))
	{
		int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, dispatcher.database->GetBlockSize());
		GPUFilter::constConst<OP, T, U>(mask, constLeft, constRight, dispatcher.database->GetBlockSize());
	}
	return 0;
}

template<typename OP, typename T, typename U>
int32_t logicalColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	int32_t loadFlag = dispatcher.loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::tuple<uintptr_t, int32_t, bool> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!dispatcher.isRegisterAllocated(reg))
	{
		int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
		GPULogic::colConst<OP, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	}

	dispatcher.freeColumnIfRegister<T>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t logicalConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	int32_t loadFlag = dispatcher.loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::tuple<uintptr_t, int32_t, bool> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!dispatcher.isRegisterAllocated(reg))
	{
		int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
		GPULogic::constCol<OP, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
	}

	dispatcher.freeColumnIfRegister<U>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t logicalColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	int32_t loadFlag = dispatcher.loadCol<U>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = dispatcher.loadCol<T>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Logical: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> columnRight = dispatcher.allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t, bool> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);

	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	if (!dispatcher.isRegisterAllocated(reg))
	{
		int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
		GPULogic::colCol<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	}

	dispatcher.freeColumnIfRegister<U>(colNameRight);
	dispatcher.freeColumnIfRegister<T>(colNameLeft);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t logicalConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	if (!dispatcher.isRegisterAllocated(reg))
	{
		int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, dispatcher.database->GetBlockSize());
		GPULogic::constConst<OP, T, U>(mask, constLeft, constRight, dispatcher.database->GetBlockSize());
	}

	return 0;
}

template<typename OP, typename T, typename U>
int32_t arithmeticColConst(GpuSqlDispatcher &dispatcher)
{
	U cnst = dispatcher.arguments.read<U>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
		>::type ResultType;
	int32_t loadFlag = dispatcher.loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticColConst: " << colName << " " << reg << std::endl;

	if (dispatcher.groupByColumns.find(colName) != dispatcher.groupByColumns.end())
	{
		if (dispatcher.isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> column = dispatcher.allocatedPointers.at(colName + "_keys");
			int32_t retSize = std::get<1>(column);
			ResultType * result = dispatcher.allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::colConst<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
			dispatcher.groupByColumns.insert(reg);
		}
	}
	else if (dispatcher.isLastBlockOfDevice || !dispatcher.usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t, bool> column = dispatcher.allocatedPointers.at(colName);
		int32_t retSize = std::get<1>(column);
		if (!dispatcher.isRegisterAllocated(reg))
		{
			ResultType * result = dispatcher.allocateRegister<ResultType>(reg, retSize);
			GPUArithmetic::colConst<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
		}
	}
	dispatcher.freeColumnIfRegister<T>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t arithmeticConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	T cnst = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();

	
	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
	>::type ResultType;
	int32_t loadFlag = dispatcher.loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticConstCol: " << colName << " " << reg << std::endl;

	if (dispatcher.groupByColumns.find(colName) != dispatcher.groupByColumns.end())
	{
		if (dispatcher.isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> column = dispatcher.allocatedPointers.at(colName + "_keys");
			int32_t retSize = std::get<1>(column);
			ResultType * result = dispatcher.allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::constCol<OP, ResultType, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
			dispatcher.groupByColumns.insert(reg);
		}
	}
	else if (dispatcher.isLastBlockOfDevice || !dispatcher.usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t, bool> column = dispatcher.allocatedPointers.at(colName);
		int32_t retSize = std::get<1>(column);

		if (!dispatcher.isRegisterAllocated(reg))
		{
			ResultType * result = dispatcher.allocateRegister<ResultType>(reg, retSize);
			GPUArithmetic::constCol<OP, ResultType, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
		}
	}
	dispatcher.freeColumnIfRegister<U>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t arithmeticColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
	auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();
	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
	>::type ResultType;

	int32_t loadFlag = dispatcher.loadCol<U>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = dispatcher.loadCol<T>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	if (dispatcher.groupByColumns.find(colNameRight) != dispatcher.groupByColumns.end())
	{
		if (dispatcher.isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> columnRight = dispatcher.allocatedPointers.at(colNameRight + "_keys");
			std::tuple<uintptr_t, int32_t, bool> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
			int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

			ResultType * result = dispatcher.allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::colCol<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
			dispatcher.groupByColumns.insert(reg);
		}
	}
	else if (dispatcher.groupByColumns.find(colNameLeft) != dispatcher.groupByColumns.end())
	{
		if (dispatcher.isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> columnRight = dispatcher.allocatedPointers.at(colNameRight);
			std::tuple<uintptr_t, int32_t, bool> columnLeft = dispatcher.allocatedPointers.at(colNameLeft + "_keys");
			int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

			ResultType * result = dispatcher.allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::colCol<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
			dispatcher.groupByColumns.insert(reg);
		}
	}
	else if (dispatcher.isLastBlockOfDevice || !dispatcher.usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t, bool> columnRight = dispatcher.allocatedPointers.at(colNameRight);
		std::tuple<uintptr_t, int32_t, bool> columnLeft = dispatcher.allocatedPointers.at(colNameLeft);
		int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

		if (!dispatcher.isRegisterAllocated(reg))
		{
			ResultType * result = dispatcher.allocateRegister<ResultType>(reg, retSize);
			GPUArithmetic::colCol<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
		}
	}
	dispatcher.freeColumnIfRegister<T>(colNameLeft);
	dispatcher.freeColumnIfRegister<U>(colNameRight);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t arithmeticConstConst(GpuSqlDispatcher &dispatcher)
{
	U constRight = dispatcher.arguments.read<U>();
	T constLeft = dispatcher.arguments.read<T>();
	auto reg = dispatcher.arguments.read<std::string>();
	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point< U>::value, U, void>::type>::type
	>::type ResultType;
	std::cout << "ArithmeticConstConst: " << reg << std::endl;

	int32_t retSize = 1;

	if (!dispatcher.isRegisterAllocated(reg))
	{
		ResultType * result = dispatcher.allocateRegister<ResultType>(reg, retSize);
		GPUArithmetic::constConst<OP, ResultType, T, U>(result, constLeft, constRight, retSize);
	}
	return 0;
}

template<typename T, typename U>
int32_t containsColConst(GpuSqlDispatcher &dispatcher)
{
	auto constWkt = dispatcher.arguments.read<std::string>();
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	int32_t loadFlag = dispatcher.loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ContainsColConst: " + colName << " " << constWkt << " " << reg << std::endl;

	auto polygonCol = dispatcher.findComplexPolygon(colName);
	ColmnarDB::Types::Point pointConst = PointFactory::FromWkt(constWkt);

	GPUMemory::GPUPolygon polygons = std::get<0>(polygonCol);
	NativeGeoPoint* pointConstPtr = dispatcher.insertConstPointGpu(pointConst);
	int32_t retSize = std::get<1>(polygonCol);

	if (!dispatcher.isRegisterAllocated(reg))
	{
        int8_t* result = dispatcher.allocateRegister<int8_t>(reg, retSize);
        GPUPolygon::contains(result, pointConstPtr, reinterpret_cast<NativeGeoPoint*>(polygons.polyPoints),
                             reinterpret_cast<int32_t*>(polygons.polyIdx),
                             reinterpret_cast<int32_t*>(polygons.polyCount),
                             reinterpret_cast<int32_t*>(polygons.pointIdx),
                             reinterpret_cast<int32_t*>(polygons.pointCount), 1, retSize);
	}
	return 0;
}

template<typename T, typename U>
int32_t containsConstCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	auto constWkt = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	int32_t loadFlag = dispatcher.loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ContainsConstCol: " + constWkt << " " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> columnPoint = dispatcher.allocatedPointers.at(colName);
	ColmnarDB::Types::ComplexPolygon polygonConst = ComplexPolygonFactory::FromWkt(constWkt);
	std::string gpuPolygon = dispatcher.insertConstPolygonGpu(polygonConst);

	int32_t retSize = std::get<1>(columnPoint);

	if (!dispatcher.isRegisterAllocated(reg))
	{
        int8_t* result = dispatcher.allocateRegister<int8_t>(reg, retSize);
        GPUPolygon::contains(result, reinterpret_cast<NativeGeoPoint*>(std::get<0>(columnPoint)),
                             reinterpret_cast<NativeGeoPoint*>(std::get<0>(
                                 dispatcher.allocatedPointers.at(gpuPolygon + "_polyPoints"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 dispatcher.allocatedPointers.at(gpuPolygon + "_polyIdx"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 dispatcher.allocatedPointers.at(gpuPolygon + "_polyCount"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 dispatcher.allocatedPointers.at(gpuPolygon + "_pointIdx"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 dispatcher.allocatedPointers.at(gpuPolygon + "_pointCount"))),
                             retSize, 1);
	}
	return 0;
}

template<typename T, typename U>
int32_t containsColCol(GpuSqlDispatcher &dispatcher)
{
	auto colNamePoint = dispatcher.arguments.read<std::string>();
	auto colNamePolygon = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	int32_t loadFlag = dispatcher.loadCol<U>(colNamePoint);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = dispatcher.loadCol<T>(colNamePolygon);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ContainsColCol: " + colNamePolygon << " " << colNamePoint << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> pointCol = dispatcher.allocatedPointers.at(colNamePoint);
	auto polygonCol = dispatcher.findComplexPolygon(colNamePolygon);
	GPUMemory::GPUPolygon gpuPolygon = std::get<0>(polygonCol);


	int32_t retSize = std::min(std::get<1>(pointCol), std::get<1>(polygonCol));

	if (!dispatcher.isRegisterAllocated(reg))
	{
		int8_t * result = dispatcher.allocateRegister<int8_t>(reg, retSize);
		GPUPolygon::contains(result, reinterpret_cast<NativeGeoPoint*>(std::get<0>(pointCol)), reinterpret_cast<NativeGeoPoint*>(gpuPolygon.polyPoints), reinterpret_cast<int32_t*>(gpuPolygon.polyIdx), reinterpret_cast<int32_t*>(gpuPolygon.polyCount), reinterpret_cast<int32_t*>(gpuPolygon.pointIdx), reinterpret_cast<int32_t*>(gpuPolygon.pointCount), std::get<1>(pointCol), std::get<1>(polygonCol));
	}
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
	std::string gpuPolygon = dispatcher.insertConstPolygonGpu(constPolygon);

	int32_t retSize = dispatcher.database->GetBlockSize();

	if (!dispatcher.isRegisterAllocated(reg))
	{
        int8_t* result = dispatcher.allocateRegister<int8_t>(reg, retSize);
        GPUPolygon::containsConst(result, constNativeGeoPoint,
                             reinterpret_cast<NativeGeoPoint*>(std::get<0>(
                                 dispatcher.allocatedPointers.at(gpuPolygon + "_polyPoints"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 dispatcher.allocatedPointers.at(gpuPolygon + "_polyIdx"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 dispatcher.allocatedPointers.at(gpuPolygon + "_polyCount"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 dispatcher.allocatedPointers.at(gpuPolygon + "_pointIdx"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 dispatcher.allocatedPointers.at(gpuPolygon + "_pointCount"))),
			retSize);
	}
	return 0;
}

template <typename OP, typename T, typename U>
int32_t polygonOperationColConst(GpuSqlDispatcher& dispatcher)
{
    std::cout << "Polygon operation: " << std::endl;
    return 0;
}

template <typename OP, typename T, typename U>
int32_t polygonOperationConstCol(GpuSqlDispatcher& dispatcher)
{
    std::cout << "Polygon operation: " << std::endl;
    return 0;
}

template <typename OP, typename T, typename U>
int32_t polygonOperationColCol(GpuSqlDispatcher& dispatcher)
{
	auto colNameRight = dispatcher.arguments.read<std::string>();
    auto colNameLeft = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

    std::cout << "Polygon operation: " << colNameRight << " " << colNameLeft << " " << reg << std::endl;
    return 0;
}

template <typename OP, typename T, typename U>
int32_t polygonOperationConstConst(GpuSqlDispatcher& dispatcher)
{
    std::cout << "Polygon operation: " << std::endl;
    return 0;
}

template<typename T>
int32_t logicalNotCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	int32_t loadFlag = dispatcher.loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "NotCol: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!dispatcher.isRegisterAllocated(reg))
	{
		int8_t * mask = dispatcher.allocateRegister<int8_t>(reg, retSize);
		GPULogic::not_col<int8_t, T>(mask, reinterpret_cast<T*>(std::get<0>(column)), retSize);
	}

	dispatcher.freeColumnIfRegister<T>(colName);
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

template<typename OP>
int32_t dateExtractCol(GpuSqlDispatcher &dispatcher)
{
	auto colName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	int32_t loadFlag = dispatcher.loadCol<int64_t>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ExtractDatePartCol: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> column = dispatcher.allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!dispatcher.isRegisterAllocated(reg))
	{
		int32_t * result = dispatcher.allocateRegister<int32_t>(reg, retSize);
		GPUDate::extractCol<OP>(result, reinterpret_cast<int64_t*>(std::get<0>(column)), retSize);
	}

	dispatcher.freeColumnIfRegister<int64_t>(colName);
	return 0;
}

template<typename OP>
int32_t dateExtractConst(GpuSqlDispatcher &dispatcher)
{
	int64_t cnst = dispatcher.arguments.read<int64_t>();
	auto reg = dispatcher.arguments.read<std::string>();
	std::cout << "ExtractDatePartConst: " << cnst << " " << reg << std::endl;

	int32_t retSize = 1;

	if (!dispatcher.isRegisterAllocated(reg))
	{
		int32_t * result = dispatcher.allocateRegister<int32_t>(reg, retSize);
		GPUDate::extractConst<OP>(result, cnst, retSize);
	}
	return 0;
}


template<typename OP, typename R, typename T, typename U>
int32_t aggregationColCol(GpuSqlDispatcher &dispatcher) 
{
	auto colTableName = dispatcher.arguments.read<std::string>();
	auto reg = dispatcher.arguments.read<std::string>();

	int32_t loadFlag = dispatcher.loadCol<U>(colTableName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "AggColCol: " << colTableName << " " << reg << ", thread: " << dispatcher.dispatcherThreadId << std::endl;

	
	std::tuple<uintptr_t, int32_t, bool>& column = dispatcher.allocatedPointers.at(colTableName);
	int32_t reconstructOutSize;

	if (!dispatcher.usingGroupBy || colTableName != *(dispatcher.groupByColumns.begin()))
	{
		T* reconstructOutReg;
		GPUReconstruct::reconstructColKeep<T>(&reconstructOutReg, &reconstructOutSize, reinterpret_cast<T*>(std::get<0>(column)), reinterpret_cast<int8_t*>(dispatcher.filter_), std::get<1>(column));

		if (std::get<2>(column))
		{
			GPUMemory::free(reinterpret_cast<void*>(std::get<0>(column)));
		}
		else
		{
			std::get<2>(column) = true;
		}
		std::get<0>(column) = reinterpret_cast<uintptr_t>(reconstructOutReg);
		std::get<1>(column) = reconstructOutSize;
	}
	const size_t endOfPolyIdx = colTableName.find(".");
	const std::string table = colTableName.substr(0, endOfPolyIdx);
	const std::string columnName = colTableName.substr(endOfPolyIdx + 1);
	
	if (dispatcher.usingGroupBy)
	{
		//TODO void param
		if (dispatcher.groupByTables[dispatcher.dispatcherThreadId] == nullptr)
		{
			dispatcher.groupByTables[dispatcher.dispatcherThreadId] = std::make_unique<GPUGroupBy<OP,R,U,T>>(Configuration::GetInstance().GetGroupByBuckets());
		}

		std::string groupByColumnName = *(dispatcher.groupByColumns.begin());
		std::tuple<uintptr_t, int32_t, bool> groupByColumn = dispatcher.allocatedPointers.at(groupByColumnName);


		
		int32_t dataSize = std::min(std::get<1>(groupByColumn), std::get<1>(column));

		reinterpret_cast<GPUGroupBy<OP, R, U, T>*>(dispatcher.groupByTables[dispatcher.dispatcherThreadId].get())->groupBy(reinterpret_cast<U*>(std::get<0>(groupByColumn)), reinterpret_cast<T*>(std::get<0>(column)), dataSize);

		// If last block was processed, reconstruct group by table
		if (dispatcher.isLastBlockOfDevice)
		{
			if (dispatcher.isOverallLastBlock)
			{
				// Wait until all threads finished work
				std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
				GpuSqlDispatcher::groupByCV_.wait(lock, []{ return GpuSqlDispatcher::IsGroupByDone(); });

				std::cout << "Reconstructing group by in thread: " << dispatcher.dispatcherThreadId << std::endl;
				int32_t outSize;
				U* outKeys;
				R* outValues;
				reinterpret_cast<GPUGroupBy<OP, R, U, T>*>(dispatcher.groupByTables[dispatcher.dispatcherThreadId].get())->getResults(&outKeys, &outValues, &outSize, dispatcher.groupByTables);
				dispatcher.allocatedPointers.insert({ groupByColumnName + "_keys",std::make_tuple(reinterpret_cast<uintptr_t>(outKeys), outSize, true) });
				dispatcher.allocatedPointers.insert({ reg,std::make_tuple(reinterpret_cast<uintptr_t>(outValues), outSize, true) });
			}
			else
			{
				std::cout << "Group by all blocks done in thread: " << dispatcher.dispatcherThreadId << std::endl;
				// Increment counter and notify threads
				std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
				GpuSqlDispatcher::IncGroupByDoneCounter();
				GpuSqlDispatcher::groupByCV_.notify_all();
			}
		}
	}
	else
	{
		if (!dispatcher.isRegisterAllocated(reg))
		{
			T * result = dispatcher.allocateRegister<T>(reg, 1);
			GPUAggregation::col<OP, T>(result, reinterpret_cast<T*>(std::get<0>(column)), std::get<1>(column));
		}
	}
	dispatcher.freeColumnIfRegister<U>(colTableName);
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

	int32_t loadFlag = dispatcher.loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "AggConstCol: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool>& column = dispatcher.allocatedPointers.at(colName);
	int32_t reconstructOutSize;

	T* reconstructOutReg;
	GPUReconstruct::reconstructColKeep<T>(&reconstructOutReg, &reconstructOutSize, reinterpret_cast<T*>(std::get<0>(column)), reinterpret_cast<int8_t*>(dispatcher.filter_), std::get<1>(column));

	if (std::get<2>(column))
	{
		GPUMemory::free(reinterpret_cast<void*>(std::get<0>(column)));
	}
	else
	{
		std::get<2>(column) = true;
	}

	std::get<0>(column) = reinterpret_cast<uintptr_t>(reconstructOutReg);
	std::get<1>(column) = reconstructOutSize;

	if (!dispatcher.isRegisterAllocated(reg))
	{
		T * result = dispatcher.allocateRegister<T>(reg, 1);
		GPUAggregation::col<OP, T>(result, reinterpret_cast<T*>(std::get<0>(column)), std::get<1>(column));
	}
	dispatcher.freeColumnIfRegister<T>(colName);
	dispatcher.filter_ = 0;
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
	std::string columnName = dispatcher.arguments.read<std::string>();
	
	int32_t loadFlag = dispatcher.loadCol<T>(columnName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "GroupBy: " << columnName << std::endl;

	std::tuple<uintptr_t, int32_t, bool>& column = dispatcher.allocatedPointers.at(columnName);

	int32_t reconstructOutSize;
	T* reconstructOutReg;
	GPUReconstruct::reconstructColKeep<T>(&reconstructOutReg, &reconstructOutSize, reinterpret_cast<T*>(std::get<0>(column)), reinterpret_cast<int8_t*>(dispatcher.filter_), std::get<1>(column));

	if (std::get<2>(column))
	{
		GPUMemory::free(reinterpret_cast<void*>(std::get<0>(column)));
	}
	else
	{
		std::get<2>(column) = true;
	}
	std::get<0>(column) = reinterpret_cast<uintptr_t>(reconstructOutReg);
	std::get<1>(column) = reconstructOutSize;

	if (dispatcher.groupByColumns.find(columnName) == dispatcher.groupByColumns.end())
	{
		dispatcher.groupByColumns.insert(columnName);
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