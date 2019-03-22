#pragma once
#include <cstdint>
#include "GpuSqlDispatcher.h"
#include "../QueryEngine/GPUCore/GPUConversion.cuh"
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
int32_t GpuSqlDispatcher::retConst()
{
	T cnst = arguments.read<T>();
	std::cout << "RET: cnst" << typeid(T).name() << std::endl;
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::retCol()
{
	auto col = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(col);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "RetCol: " << col << ", thread: " << dispatcherThreadId << std::endl;

	int32_t outSize;

	if (usingGroupBy)
	{
		if (isOverallLastBlock)
		{
			if (groupByColumns.find(col) != groupByColumns.end())
			{
				std::tuple<uintptr_t, int32_t, bool> keyCol = allocatedPointers.at(col + "_keys");
				outSize = std::get<1>(keyCol);
				std::unique_ptr<T[]> outData(new T[outSize]);
				GPUMemory::copyDeviceToHost(outData.get(), reinterpret_cast<T*>(std::get<0>(keyCol)), outSize);

				ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
				insertIntoPayload<T>(payload, outData, outSize);
				ColmnarDB::NetworkClient::Message::QueryResponseMessage partialMessage;
				mergePayloadToResponse(col, payload);
			}
			else
			{
				std::tuple<uintptr_t, int32_t, bool> valueCol = allocatedPointers.at(col);
				outSize = std::get<1>(valueCol);
				std::unique_ptr<T[]> outData(new T[outSize]);
				GPUMemory::copyDeviceToHost(outData.get(), reinterpret_cast<T*>(std::get<0>(valueCol)), outSize);

				ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
				insertIntoPayload<T>(payload, outData, outSize);
				mergePayloadToResponse(col, payload);
			}
		}
	}
	else
	{
		std::unique_ptr<T[]> outData(new T[database->GetBlockSize()]);
		//ToDo: Podmienene zapnut podla velkost buffera
		//GPUMemory::hostPin(outData.get(), database->GetBlockSize());
		std::tuple<uintptr_t, int32_t, bool> ACol = allocatedPointers.at(col);
		GPUReconstruct::reconstructCol(outData.get(), &outSize, reinterpret_cast<T*>(std::get<0>(ACol)), reinterpret_cast<int8_t*>(filter_), std::get<1>(ACol));
		//GPUMemory::hostUnregister(outData.get());
		std::cout << "dataSize: " << outSize << std::endl;
		ColmnarDB::NetworkClient::Message::QueryResponsePayload payload;
		insertIntoPayload<T>(payload, outData, outSize);
		mergePayloadToResponse(col, payload);
	}
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::filterColConst()
{
	U cnst = arguments.read<U>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Filter: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPUFilter::colConst<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	}

	freeColumnIfRegister<T>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::filterConstCol()
{
	auto colName = arguments.read<std::string>();
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Filter: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPUFilter::constCol<OP, T, U>(mask, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
	}

	freeColumnIfRegister<U>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::filterColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = loadCol<T>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Filter: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t, bool> columnLeft = allocatedPointers.at(colNameLeft);
	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPUFilter::colCol<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	}

	freeColumnIfRegister<U>(colNameRight);
	freeColumnIfRegister<T>(colNameLeft);
	return 0;
}


template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::filterConstConst()
{
	U constRight = arguments.read<U>();
	T constLeft = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, database->GetBlockSize());
		GPUFilter::constConst<OP, T, U>(mask, constLeft, constRight, database->GetBlockSize());
	}
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::logicalColConst()
{
	U cnst = arguments.read<U>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		int8_t * result = allocateRegister<int8_t>(reg, retSize);
		GPULogic::colConst<OP, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
	}

	freeColumnIfRegister<T>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::logicalConstCol()
{
	auto colName = arguments.read<std::string>();
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		int8_t * result = allocateRegister<int8_t>(reg, retSize);
		GPULogic::constCol<OP, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
	}

	freeColumnIfRegister<U>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::logicalColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = loadCol<T>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Logical: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t, bool> columnLeft = allocatedPointers.at(colNameLeft);

	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPULogic::colCol<OP, T, U>(mask, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	}

	freeColumnIfRegister<U>(colNameRight);
	freeColumnIfRegister<T>(colNameLeft);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::logicalConstConst()
{
	U constRight = arguments.read<U>();
	T constLeft = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, database->GetBlockSize());
		GPULogic::constConst<OP, T, U>(mask, constLeft, constRight, database->GetBlockSize());
	}

	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::arithmeticColConst()
{
	U cnst = arguments.read<U>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
		>::type ResultType;
	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticColConst: " << colName << " " << reg << std::endl;

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		if (isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName + "_keys");
			int32_t retSize = std::get<1>(column);
			ResultType * result = allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::colConst<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
			groupByColumns.insert(reg);
		}
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
		int32_t retSize = std::get<1>(column);
		if (!isRegisterAllocated(reg))
		{
			ResultType * result = allocateRegister<ResultType>(reg, retSize);
			GPUArithmetic::colConst<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(column)), cnst, retSize);
		}
	}
	freeColumnIfRegister<T>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::arithmeticConstCol()
{
	auto colName = arguments.read<std::string>();
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	
	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
	>::type ResultType;
	int32_t loadFlag = loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticConstCol: " << colName << " " << reg << std::endl;

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		if (isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName + "_keys");
			int32_t retSize = std::get<1>(column);
			ResultType * result = allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::constCol<OP, ResultType, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
			groupByColumns.insert(reg);
		}
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
		int32_t retSize = std::get<1>(column);

		if (!isRegisterAllocated(reg))
		{
			ResultType * result = allocateRegister<ResultType>(reg, retSize);
			GPUArithmetic::constCol<OP, ResultType, T, U>(result, cnst, reinterpret_cast<U*>(std::get<0>(column)), retSize);
		}
	}
	freeColumnIfRegister<U>(colName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::arithmeticColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();
	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
	>::type ResultType;

	int32_t loadFlag = loadCol<U>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = loadCol<T>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	if (groupByColumns.find(colNameRight) != groupByColumns.end())
	{
		if (isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(colNameRight + "_keys");
			std::tuple<uintptr_t, int32_t, bool> columnLeft = allocatedPointers.at(colNameLeft);
			int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

			ResultType * result = allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::colCol<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
			groupByColumns.insert(reg);
		}
	}
	else if (groupByColumns.find(colNameLeft) != groupByColumns.end())
	{
		if (isLastBlockOfDevice)
		{
			std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(colNameRight);
			std::tuple<uintptr_t, int32_t, bool> columnLeft = allocatedPointers.at(colNameLeft + "_keys");
			int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

			ResultType * result = allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::colCol<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
			groupByColumns.insert(reg);
		}
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(colNameRight);
		std::tuple<uintptr_t, int32_t, bool> columnLeft = allocatedPointers.at(colNameLeft);
		int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

		if (!isRegisterAllocated(reg))
		{
			ResultType * result = allocateRegister<ResultType>(reg, retSize);
			GPUArithmetic::colCol<OP, ResultType, T, U>(result, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
		}
	}
	freeColumnIfRegister<T>(colNameLeft);
	freeColumnIfRegister<U>(colNameRight);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::arithmeticConstConst()
{
	U constRight = arguments.read<U>();
	T constLeft = arguments.read<T>();
	auto reg = arguments.read<std::string>();
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

	if (!isRegisterAllocated(reg))
	{
		ResultType * result = allocateRegister<ResultType>(reg, retSize);
		GPUArithmetic::constConst<OP, ResultType, T, U>(result, constLeft, constRight, retSize);
	}
	return 0;
}

template<typename T, typename U>
int32_t GpuSqlDispatcher::pointColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	std::cout << "PointColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	int32_t loadFlag = loadCol<U>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = loadCol<T>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(colNameRight);
	std::tuple<uintptr_t, int32_t, bool> columnLeft = allocatedPointers.at(colNameLeft);

	int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

	if (!isRegisterAllocated(reg))
	{
		NativeGeoPoint * pointCol = allocateRegister<NativeGeoPoint>(reg, retSize);
		GPUConversion::ConvertColCol(pointCol, reinterpret_cast<T*>(std::get<0>(columnLeft)), reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	}

	freeColumnIfRegister<U>(colNameRight);
	freeColumnIfRegister<T>(colNameLeft);
	return 0;
}

template<typename T, typename U>
int32_t GpuSqlDispatcher::pointColConst()
{
	U cnst = arguments.read<U>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	std::cout << "PointColConst: " << colNameLeft << " " << reg << std::endl;

	int32_t loadFlag = loadCol<T>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::tuple<uintptr_t, int32_t, bool> columnLeft = allocatedPointers.at(colNameLeft);

	int32_t retSize = std::get<1>(columnLeft);

	if (!isRegisterAllocated(reg))
	{
		NativeGeoPoint * pointCol = allocateRegister<NativeGeoPoint>(reg, retSize);
		GPUConversion::ConvertColConst(pointCol, reinterpret_cast<T*>(std::get<0>(columnLeft)), cnst, retSize);
	}

	freeColumnIfRegister<T>(colNameLeft);
	return 0;
}

template<typename T, typename U>
int32_t GpuSqlDispatcher::pointConstCol()
{
	auto colNameRight = arguments.read<std::string>();
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	std::cout << "PointConstCol: " << colNameRight << " " << reg << std::endl;

	int32_t loadFlag = loadCol<U>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(colNameRight);

	int32_t retSize = std::get<1>(columnRight);

	if (!isRegisterAllocated(reg))
	{
		NativeGeoPoint * pointCol = allocateRegister<NativeGeoPoint>(reg, retSize);
		GPUConversion::ConvertConstCol(pointCol, cnst, reinterpret_cast<U*>(std::get<0>(columnRight)), retSize);
	}

	freeColumnIfRegister<U>(colNameRight);
	return 0;
}

template<typename T, typename U>
int32_t GpuSqlDispatcher::containsColConst()
{
	auto constWkt = arguments.read<std::string>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ContainsColConst: " + colName << " " << constWkt << " " << reg << std::endl;

	auto polygonCol = findComplexPolygon(colName);
	ColmnarDB::Types::Point pointConst = PointFactory::FromWkt(constWkt);

	GPUMemory::GPUPolygon polygons = std::get<0>(polygonCol);
	NativeGeoPoint* pointConstPtr = insertConstPointGpu(pointConst);
	int32_t retSize = std::get<1>(polygonCol);

	if (!isRegisterAllocated(reg))
	{
        int8_t* result = allocateRegister<int8_t>(reg, retSize);
        GPUPolygon::contains(result, pointConstPtr, reinterpret_cast<NativeGeoPoint*>(polygons.polyPoints),
                             reinterpret_cast<int32_t*>(polygons.polyIdx),
                             reinterpret_cast<int32_t*>(polygons.polyCount),
                             reinterpret_cast<int32_t*>(polygons.pointIdx),
                             reinterpret_cast<int32_t*>(polygons.pointCount), 1, retSize);
	}
	return 0;
}

template<typename T, typename U>
int32_t GpuSqlDispatcher::containsConstCol()
{
	auto colName = arguments.read<std::string>();
	auto constWkt = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ContainsConstCol: " + constWkt << " " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> columnPoint = allocatedPointers.at(colName);
	ColmnarDB::Types::ComplexPolygon polygonConst = ComplexPolygonFactory::FromWkt(constWkt);
	std::string gpuPolygon = insertConstPolygonGpu(polygonConst);

	int32_t retSize = std::get<1>(columnPoint);

	if (!isRegisterAllocated(reg))
	{
        int8_t* result = allocateRegister<int8_t>(reg, retSize);
        GPUPolygon::contains(result, reinterpret_cast<NativeGeoPoint*>(std::get<0>(columnPoint)),
                             reinterpret_cast<NativeGeoPoint*>(std::get<0>(
                                 allocatedPointers.at(gpuPolygon + "_polyPoints"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 allocatedPointers.at(gpuPolygon + "_polyIdx"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 allocatedPointers.at(gpuPolygon + "_polyCount"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 allocatedPointers.at(gpuPolygon + "_pointIdx"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 allocatedPointers.at(gpuPolygon + "_pointCount"))),
                             retSize, 1);
	}
	return 0;
}

template<typename T, typename U>
int32_t GpuSqlDispatcher::containsColCol()
{
	auto colNamePoint = arguments.read<std::string>();
	auto colNamePolygon = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colNamePoint);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = loadCol<T>(colNamePolygon);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ContainsColCol: " + colNamePolygon << " " << colNamePoint << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> pointCol = allocatedPointers.at(colNamePoint);
	auto polygonCol = findComplexPolygon(colNamePolygon);
	GPUMemory::GPUPolygon gpuPolygon = std::get<0>(polygonCol);


	int32_t retSize = std::min(std::get<1>(pointCol), std::get<1>(polygonCol));

	if (!isRegisterAllocated(reg))
	{
		int8_t * result = allocateRegister<int8_t>(reg, retSize);
		GPUPolygon::contains(result, reinterpret_cast<NativeGeoPoint*>(std::get<0>(pointCol)), reinterpret_cast<NativeGeoPoint*>(gpuPolygon.polyPoints), reinterpret_cast<int32_t*>(gpuPolygon.polyIdx), reinterpret_cast<int32_t*>(gpuPolygon.polyCount), reinterpret_cast<int32_t*>(gpuPolygon.pointIdx), reinterpret_cast<int32_t*>(gpuPolygon.pointCount), std::get<1>(pointCol), std::get<1>(polygonCol));
	}
	return 0;
}

template<typename T, typename U>
int32_t GpuSqlDispatcher::containsConstConst()
{
	// TODO : Specialize kernel for all cases.
	auto constPointWkt = arguments.read<std::string>();
	auto constPolygonWkt = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	std::cout << "ContainsConstConst: " + constPolygonWkt << " " << constPointWkt << " " << reg << std::endl;

	ColmnarDB::Types::Point constPoint = PointFactory::FromWkt(constPointWkt);
	ColmnarDB::Types::ComplexPolygon constPolygon = ComplexPolygonFactory::FromWkt(constPolygonWkt);

	NativeGeoPoint *constNativeGeoPoint = insertConstPointGpu(constPoint);
	std::string gpuPolygon = insertConstPolygonGpu(constPolygon);

	int32_t retSize = database->GetBlockSize();

	if (!isRegisterAllocated(reg))
	{
        int8_t* result = allocateRegister<int8_t>(reg, retSize);
        GPUPolygon::containsConst(result, constNativeGeoPoint,
                             reinterpret_cast<NativeGeoPoint*>(std::get<0>(
                                 allocatedPointers.at(gpuPolygon + "_polyPoints"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 allocatedPointers.at(gpuPolygon + "_polyIdx"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 allocatedPointers.at(gpuPolygon + "_polyCount"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 allocatedPointers.at(gpuPolygon + "_pointIdx"))),
                             reinterpret_cast<int32_t*>(std::get<0>(
                                 allocatedPointers.at(gpuPolygon + "_pointCount"))),
			retSize);
	}
	return 0;
}

template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::polygonOperationColConst(GpuSqlDispatcher& dispatcher)
{
    std::cout << "Polygon operation: " << std::endl;
    return 0;
}

template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::polygonOperationConstCol(GpuSqlDispatcher& dispatcher)
{
    std::cout << "Polygon operation: " << std::endl;
    return 0;
}

template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::polygonOperationColCol(GpuSqlDispatcher& dispatcher)
{
	auto colNameRight = arguments.read<std::string>();
    auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

    std::cout << "Polygon operation: " << colNameRight << " " << colNameLeft << " " << reg << std::endl;
    return 0;
}

template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::polygonOperationConstConst(GpuSqlDispatcher& dispatcher)
{
    std::cout << "Polygon operation: " << std::endl;
    return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::logicalNotCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "NotCol: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, retSize);
		GPULogic::not_col<int8_t, T>(mask, reinterpret_cast<T*>(std::get<0>(column)), retSize);
	}

	freeColumnIfRegister<T>(colName);
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::logicalNotConst()
{
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::minusCol()
{
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::minusConst()
{
	return 0;
}

template<typename OP>
int32_t GpuSqlDispatcher::dateExtractCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<int64_t>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ExtractDatePartCol: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(colName);
	int32_t retSize = std::get<1>(column);

	if (!isRegisterAllocated(reg))
	{
		int32_t * result = allocateRegister<int32_t>(reg, retSize);
		GPUDate::extractCol<OP>(result, reinterpret_cast<int64_t*>(std::get<0>(column)), retSize);
	}

	freeColumnIfRegister<int64_t>(colName);
	return 0;
}

template<typename OP>
int32_t GpuSqlDispatcher::dateExtractConst()
{
	int64_t cnst = arguments.read<int64_t>();
	auto reg = arguments.read<std::string>();
	std::cout << "ExtractDatePartConst: " << cnst << " " << reg << std::endl;

	int32_t retSize = 1;

	if (!isRegisterAllocated(reg))
	{
		int32_t * result = allocateRegister<int32_t>(reg, retSize);
		GPUDate::extractConst<OP>(result, cnst, retSize);
	}
	return 0;
}


template<typename OP, typename R, typename T, typename U>
int32_t GpuSqlDispatcher::aggregationColCol() 
{
	auto colTableName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colTableName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "AggColCol: " << colTableName << " " << reg << ", thread: " << dispatcherThreadId << std::endl;

	
	std::tuple<uintptr_t, int32_t, bool>& column = allocatedPointers.at(colTableName);
	int32_t reconstructOutSize;

	if (!usingGroupBy || colTableName != *(groupByColumns.begin()))
	{
		T* reconstructOutReg;
		GPUReconstruct::reconstructColKeep<T>(&reconstructOutReg, &reconstructOutSize, reinterpret_cast<T*>(std::get<0>(column)), reinterpret_cast<int8_t*>(filter_), std::get<1>(column));

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
	
	if (usingGroupBy)
	{
		//TODO void param
		if (groupByTables[dispatcherThreadId] == nullptr)
		{
			groupByTables[dispatcherThreadId] = std::make_unique<GPUGroupBy<OP,R,U,T>>(Configuration::GetInstance().GetGroupByBuckets());
		}

		std::string groupByColumnName = *(groupByColumns.begin());
		std::tuple<uintptr_t, int32_t, bool> groupByColumn = allocatedPointers.at(groupByColumnName);


		
		int32_t dataSize = std::min(std::get<1>(groupByColumn), std::get<1>(column));

		reinterpret_cast<GPUGroupBy<OP, R, U, T>*>(groupByTables[dispatcherThreadId].get())->groupBy(reinterpret_cast<U*>(std::get<0>(groupByColumn)), reinterpret_cast<T*>(std::get<0>(column)), dataSize);

		// If last block was processed, reconstruct group by table
		if (isLastBlockOfDevice)
		{
			if (isOverallLastBlock)
			{
				// Wait until all threads finished work
				std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
				GpuSqlDispatcher::groupByCV_.wait(lock, []{ return GpuSqlDispatcher::IsGroupByDone(); });

				std::cout << "Reconstructing group by in thread: " << dispatcherThreadId << std::endl;
				int32_t outSize;
				U* outKeys;
				R* outValues;
				reinterpret_cast<GPUGroupBy<OP, R, U, T>*>(groupByTables[dispatcherThreadId].get())->getResults(&outKeys, &outValues, &outSize, groupByTables);
				allocatedPointers.insert({ groupByColumnName + "_keys",std::make_tuple(reinterpret_cast<uintptr_t>(outKeys), outSize, true) });
				allocatedPointers.insert({ reg,std::make_tuple(reinterpret_cast<uintptr_t>(outValues), outSize, true) });
			}
			else
			{
				std::cout << "Group by all blocks done in thread: " << dispatcherThreadId << std::endl;
				// Increment counter and notify threads
				std::unique_lock<std::mutex> lock(GpuSqlDispatcher::groupByMutex_);
				GpuSqlDispatcher::IncGroupByDoneCounter();
				GpuSqlDispatcher::groupByCV_.notify_all();
			}
		}
	}
	else
	{
		if (!isRegisterAllocated(reg))
		{
			T * result = allocateRegister<T>(reg, 1);
			GPUAggregation::col<OP, T>(result, reinterpret_cast<T*>(std::get<0>(column)), std::get<1>(column));
		}
	}
	freeColumnIfRegister<U>(colTableName);
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::aggregationColConst()
{
	std::cout << "AggColConst" << std::endl;
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::aggregationConstCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "AggConstCol: " << colName << " " << reg << std::endl;

	std::tuple<uintptr_t, int32_t, bool>& column = allocatedPointers.at(colName);
	int32_t reconstructOutSize;

	T* reconstructOutReg;
	GPUReconstruct::reconstructColKeep<T>(&reconstructOutReg, &reconstructOutSize, reinterpret_cast<T*>(std::get<0>(column)), reinterpret_cast<int8_t*>(filter_), std::get<1>(column));

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

	if (!isRegisterAllocated(reg))
	{
		T * result = allocateRegister<T>(reg, 1);
		GPUAggregation::col<OP, T>(result, reinterpret_cast<T*>(std::get<0>(column)), std::get<1>(column));
	}
	freeColumnIfRegister<T>(colName);
	filter_ = 0;
	return 0;
}

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::aggregationConstConst()
{
	std::cout << "AggConstConst" << std::endl;
	return 0;
}


template<typename T>
int32_t GpuSqlDispatcher::groupByCol()
{
	std::string columnName = arguments.read<std::string>();
	
	int32_t loadFlag = loadCol<T>(columnName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "GroupBy: " << columnName << std::endl;

	std::tuple<uintptr_t, int32_t, bool>& column = allocatedPointers.at(columnName);

	int32_t reconstructOutSize;
	T* reconstructOutReg;
	GPUReconstruct::reconstructColKeep<T>(&reconstructOutReg, &reconstructOutSize, reinterpret_cast<T*>(std::get<0>(column)), reinterpret_cast<int8_t*>(filter_), std::get<1>(column));

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

	if (groupByColumns.find(columnName) == groupByColumns.end())
	{
		groupByColumns.insert(columnName);
	}
	usingGroupBy = true;
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::groupByConst()
{
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::insertInto()
{
	std::string table = arguments.read<std::string>();
	std::string column = arguments.read<std::string>();
	bool isReferencedColumn = arguments.read<bool>();

	if (isReferencedColumn)
	{
		T args = arguments.read<T>();

		dynamic_cast<ColumnBase<T>*>(database->GetTables().at(table).GetColumns().at(column).get())->InsertData({args});
	}
	else
	{
		dynamic_cast<ColumnBase<T>*>(database->GetTables().at(table).GetColumns().at(column).get())->InsertNullData(1);
	}
	return 0;
}

template<typename T, typename U>
int32_t GpuSqlDispatcher::invalidOperandTypesErrorHandlerColConst()
{
	return 0;
}

template<typename T, typename U>
int32_t GpuSqlDispatcher::invalidOperandTypesErrorHandlerConstCol()
{
	return 0;
}

template<typename T, typename U>
int32_t GpuSqlDispatcher::invalidOperandTypesErrorHandlerColCol()
{
	return 0;
}


template<typename T, typename U>
int32_t GpuSqlDispatcher::invalidOperandTypesErrorHandlerConstConst()
{
	return 0;
}


//// FUNCTOR ERROR HANDLERS

template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::invalidOperandTypesErrorHandlerColConst()
{
	return 0;
}


template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::invalidOperandTypesErrorHandlerConstCol()
{
	return 0;
}


template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::invalidOperandTypesErrorHandlerColCol()
{
	return 0;
}


template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::invalidOperandTypesErrorHandlerConstConst()
{
	return 0;
}

template<typename OP, typename T>
int32_t GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol()
{
	return 0;
}

template<typename OP, typename T>
int32_t GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst()
{
	return 0;
}

////

template<typename T>
int32_t GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol()
{
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst()
{
	return 0;
}