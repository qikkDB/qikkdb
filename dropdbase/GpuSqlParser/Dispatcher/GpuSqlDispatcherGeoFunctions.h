#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUPolygonClipping.cuh"
#include "../../QueryEngine/GPUCore/GPUPolygonContains.cuh"
#include "../../QueryEngine/GPUCore/GPUConversion.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"


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
		GPUPolygonContains::contains(result, polygons, retSize, pointConstPtr, 1);
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
	std::string gpuPolygon = insertConstPolygonGpu(polygonConst); // TODO change to return GPUMemory::GPUPolygon struct

	int32_t retSize = std::get<1>(columnPoint);

	if (!isRegisterAllocated(reg))
	{
		int8_t* result = allocateRegister<int8_t>(reg, retSize);
		// TODO change 
		GPUPolygonContains::contains(result,
			{ reinterpret_cast<NativeGeoPoint*>(std::get<0>(
				allocatedPointers.at(gpuPolygon + "_polyPoints"))),
			reinterpret_cast<int32_t*>(std::get<0>(
				allocatedPointers.at(gpuPolygon + "_pointIdx"))),
			reinterpret_cast<int32_t*>(std::get<0>(
				allocatedPointers.at(gpuPolygon + "_pointCount"))),
			reinterpret_cast<int32_t*>(std::get<0>(
				allocatedPointers.at(gpuPolygon + "_polyIdx"))),
			reinterpret_cast<int32_t*>(std::get<0>(
				allocatedPointers.at(gpuPolygon + "_polyCount"))) }, 1,
			reinterpret_cast<NativeGeoPoint*>(std::get<0>(columnPoint)), retSize);
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


	int32_t retSize = std::min(std::get<1>(pointCol), std::get<1>(polygonCol));

	if (!isRegisterAllocated(reg))
	{
		int8_t * result = allocateRegister<int8_t>(reg, retSize);
		GPUPolygonContains::contains(result, std::get<0>(polygonCol), std::get<1>(polygonCol),
			reinterpret_cast<NativeGeoPoint*>(std::get<0>(pointCol)), std::get<1>(pointCol));
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
	std::string gpuPolygon = insertConstPolygonGpu(constPolygon); // TODO change

	int32_t retSize = database->GetBlockSize();

	if (!isRegisterAllocated(reg))
	{
		int8_t* result = allocateRegister<int8_t>(reg, retSize);
		// TODO change
		GPUPolygonContains::containsConst(result,
			{ reinterpret_cast<NativeGeoPoint*>(std::get<0>(
				allocatedPointers.at(gpuPolygon + "_polyPoints"))),
			reinterpret_cast<int32_t*>(std::get<0>(
				allocatedPointers.at(gpuPolygon + "_pointIdx"))),
			reinterpret_cast<int32_t*>(std::get<0>(
				allocatedPointers.at(gpuPolygon + "_pointCount"))),
			reinterpret_cast<int32_t*>(std::get<0>(
				allocatedPointers.at(gpuPolygon + "_polyIdx"))),
			reinterpret_cast<int32_t*>(std::get<0>(
				allocatedPointers.at(gpuPolygon + "_polyCount"))) },
			constNativeGeoPoint,
			retSize);
	}
	return 0;
}

template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::polygonOperationColConst()
{
	auto colName = arguments.read<std::string>();
	auto constWkt = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "PolygonOPConstCol: " + constWkt << " " << colName << " " << reg << std::endl;

	auto polygonLeft = findComplexPolygon(colName);
	ColmnarDB::Types::ComplexPolygon polygonConst = ComplexPolygonFactory::FromWkt(constWkt);
	std::string gpuPolygon = insertConstPolygonGpu(polygonConst);

	int32_t retSize = std::get<1>(polygonLeft);

	if (!isRegisterAllocated(reg))
	{
		//TODO
	}
	return 0;
}

template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::polygonOperationConstCol()
{
	std::cout << "Polygon operation: " << std::endl;
	return 0;
}

template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::polygonOperationColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	std::cout << "Polygon operation: " << colNameRight << " " << colNameLeft << " " << reg << std::endl;

	int32_t loadFlag = loadCol<U>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = loadCol<T>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}

	auto polygonLeft = findComplexPolygon(colNameLeft);
	auto polygonRight = findComplexPolygon(colNameRight);

	int32_t dataSize = std::min(std::get<1>(polygonLeft), std::get<1>(polygonRight));
	if (!isRegisterAllocated(reg))
	{
		GPUMemory::GPUPolygon outPolygon;
		GPUPolygonClipping::ColCol<OP>(outPolygon, std::get<0>(polygonLeft), std::get<0>(polygonRight), dataSize);
		fillPolygonRegister(outPolygon, reg, dataSize);
	}
}

template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::polygonOperationConstConst()
{
	std::cout << "Polygon operation: " << std::endl;
	return 0;
}