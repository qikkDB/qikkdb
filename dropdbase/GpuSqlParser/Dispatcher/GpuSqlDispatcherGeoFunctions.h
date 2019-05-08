#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUPolygon.cuh"
#include "../../QueryEngine/GPUCore/GPUConversion.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../../ComplexPolygonFactory.h"
#include "../../PointFactory.h"

/// Implementation of POINT(a, b) operation dispatching - concatenation of two numeric attributes to single point column
/// Implementation for column column case
/// Pops data from argument memory stream and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of POINT(a, b) operation dispatching - concatenation of two numeric attributes to single point column
/// Implementation for column constant case
/// Pops data from argument memory stream and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of POINT(a, b) operation dispatching - concatenation of two numeric attributes to single point column
/// Implementation for onstant column case
/// Pops data from argument memory stream and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for column constant case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for constant column case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for column column case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for constant constant case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
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

/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does not meet this requrement) based on functor OP
/// eg. INTRSECT(a,b), UNION(a,b)
/// Implementation for column constant case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::polygonOperationColConst()
{
	std::cout << "Polygon operation: " << std::endl;
	return 0;
}

/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does not meet this requrement) based on functor OP
/// eg. INTRSECT(a,b), UNION(a,b)
/// Implementation for constant column case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::polygonOperationConstCol()
{
	std::cout << "Polygon operation: " << std::endl;
	return 0;
}
/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does not meet this requrement) based on functor OP
/// eg. INTRSECT(a,b), UNION(a,b)
/// Implementation for column column case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::polygonOperationColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	std::cout << "Polygon operation: " << colNameRight << " " << colNameLeft << " " << reg << std::endl;
	return 0;
}

/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does not meet this requrement) based on functor OP
/// eg. INTRSECT(a,b), UNION(a,b)
/// Implementation for constant constant case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::polygonOperationConstConst()
{
	std::cout << "Polygon operation: " << std::endl;
	return 0;
}