#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUPolygonClipping.cuh"
#include "../../QueryEngine/GPUCore/GPUPolygonContains.cuh"
#include "../../QueryEngine/GPUCore/GPUConversion.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../../ComplexPolygonFactory.h"
#include "../../PointFactory.h"
#include "../../Database.h"

/// Implementation of POINT(a, b) operation dispatching - concatenation of two numeric attributes to single point column
/// Implementation for column column case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t GpuSqlDispatcher::PointColCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    std::cout << "PointColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

    int32_t loadFlag = LoadCol<U>(colNameRight);
    if (loadFlag)
    {
        return loadFlag;
    }
    loadFlag = LoadCol<T>(colNameLeft);
    if (loadFlag)
    {
        return loadFlag;
    }

    PointerAllocation columnRight = allocatedPointers_.at(colNameRight);
    PointerAllocation columnLeft = allocatedPointers_.at(colNameLeft);

    int32_t retSize = std::min(columnLeft.ElementCount, columnRight.ElementCount);

    if (!IsRegisterAllocated(reg))
    {
        NativeGeoPoint* pointCol;
        if (columnLeft.GpuNullMaskPtr || columnRight.GpuNullMaskPtr)
        {
            int8_t* combinedMask;
            pointCol = AllocateRegister<NativeGeoPoint>(reg, retSize, &combinedMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
            {
                GPUArithmetic::colCol<ArithmeticOperations::bitwiseOr>(
                    combinedMask, reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                    reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
            }
            else if (columnLeft.GpuNullMaskPtr)
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr), bitMaskSize);
            }
            else if (columnRight.GpuNullMaskPtr)
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
            }
        }
        else
        {
            pointCol = AllocateRegister<NativeGeoPoint>(reg, retSize);
        }
        GPUConversion::ConvertColCol(pointCol, reinterpret_cast<T*>(columnLeft.GpuPtr),
                                     reinterpret_cast<U*>(columnRight.GpuPtr), retSize);
    }

    FreeColumnIfRegister<U>(colNameRight);
    FreeColumnIfRegister<T>(colNameLeft);
    return 0;
}

/// Implementation of POINT(a, b) operation dispatching - concatenation of two numeric attributes to single point column
/// Implementation for column constant case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t GpuSqlDispatcher::PointColConst()
{
    U cnst = arguments_.Read<U>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    std::cout << "PointColConst: " << colNameLeft << " " << reg << std::endl;

    int32_t loadFlag = LoadCol<T>(colNameLeft);
    if (loadFlag)
    {
        return loadFlag;
    }

    PointerAllocation columnLeft = allocatedPointers_.at(colNameLeft);

    int32_t retSize = columnLeft.ElementCount;

    if (!IsRegisterAllocated(reg))
    {
        NativeGeoPoint* pointCol;
        if (columnLeft.GpuNullMaskPtr)
        {
            int8_t* nullMask;
            pointCol = AllocateRegister<NativeGeoPoint>(reg, retSize, &nullMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr), bitMaskSize);
        }
        else
        {
            pointCol = AllocateRegister<NativeGeoPoint>(reg, retSize);
        }
        GPUConversion::ConvertColConst(pointCol, reinterpret_cast<T*>(columnLeft.GpuPtr), cnst, retSize);
    }

    FreeColumnIfRegister<T>(colNameLeft);
    return 0;
}

/// Implementation of POINT(a, b) operation dispatching - concatenation of two numeric attributes to single point column
/// Implementation for onstant column case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t GpuSqlDispatcher::PointConstCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    T cnst = arguments_.Read<T>();
    auto reg = arguments_.Read<std::string>();

    std::cout << "PointConstCol: " << colNameRight << " " << reg << std::endl;

    int32_t loadFlag = LoadCol<U>(colNameRight);
    if (loadFlag)
    {
        return loadFlag;
    }

    PointerAllocation columnRight = allocatedPointers_.at(colNameRight);

    int32_t retSize = columnRight.ElementCount;

    if (!IsRegisterAllocated(reg))
    {
        NativeGeoPoint* pointCol;
        if (columnRight.GpuNullMaskPtr)
        {
            int8_t* nullMask;
            pointCol = AllocateRegister<NativeGeoPoint>(reg, retSize, &nullMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr),
                                          bitMaskSize);
        }
        else
        {
            pointCol = AllocateRegister<NativeGeoPoint>(reg, retSize);
        }
        GPUConversion::ConvertConstCol(pointCol, cnst, reinterpret_cast<U*>(columnRight.GpuPtr), retSize);
    }

    FreeColumnIfRegister<U>(colNameRight);
    return 0;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for column constant case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t GpuSqlDispatcher::ContainsColConst()
{
    auto constWkt = arguments_.Read<std::string>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<T>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "ContainsColConst: " + colName << " " << constWkt << " " << reg << std::endl;

    auto polygonCol = FindComplexPolygon(colName);
    ColmnarDB::Types::Point pointConst = PointFactory::FromWkt(constWkt);

    GPUMemory::GPUPolygon polygons = std::get<0>(polygonCol);
    NativeGeoPoint* pointConstPtr = InsertConstPointGpu(pointConst);
    int32_t retSize = std::get<1>(polygonCol);

    if (!IsRegisterAllocated(reg))
    {
        int8_t* result;
        if (std::get<2>(polygonCol))
        {
            int8_t* nullMask;
            result = AllocateRegister<int8_t>(reg, retSize, &nullMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(std::get<2>(polygonCol)), bitMaskSize);
        }
        else
        {
            result = AllocateRegister<int8_t>(reg, retSize);
        }
        GPUPolygonContains::contains(result, polygons, retSize, pointConstPtr, 1);
    }
    return 0;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for constant column case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t GpuSqlDispatcher::ContainsConstCol()
{
    auto colName = arguments_.Read<std::string>();
    auto constWkt = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<U>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "ContainsConstCol: " + constWkt << " " << colName << " " << reg << std::endl;

    PointerAllocation columnPoint = allocatedPointers_.at(colName);
    ColmnarDB::Types::ComplexPolygon polygonConst = ComplexPolygonFactory::FromWkt(constWkt);
    GPUMemory::GPUPolygon gpuPolygon = InsertConstPolygonGpu(polygonConst);

    int32_t retSize = columnPoint.ElementCount;

    if (!IsRegisterAllocated(reg))
    {
        int8_t* result;
        if (columnPoint.GpuNullMaskPtr)
        {
            int8_t* nullMask;
            result = AllocateRegister<int8_t>(reg, retSize, &nullMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(columnPoint.GpuNullMaskPtr),
                                          bitMaskSize);
        }
        else
        {
            result = AllocateRegister<int8_t>(reg, retSize);
        }
        GPUPolygonContains::contains(result, gpuPolygon, 1,
                                     reinterpret_cast<NativeGeoPoint*>(columnPoint.GpuPtr), retSize);
    }
    return 0;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for column column case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t GpuSqlDispatcher::ContainsColCol()
{
    auto colNamePoint = arguments_.Read<std::string>();
    auto colNamePolygon = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<U>(colNamePoint);
    if (loadFlag)
    {
        return loadFlag;
    }
    loadFlag = LoadCol<T>(colNamePolygon);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "ContainsColCol: " + colNamePolygon << " " << colNamePoint << " " << reg << std::endl;

    PointerAllocation pointCol = allocatedPointers_.at(colNamePoint);
    auto polygonCol = FindComplexPolygon(colNamePolygon);


    int32_t retSize = std::min(pointCol.ElementCount, std::get<1>(polygonCol));

    if (!IsRegisterAllocated(reg))
    {
        int8_t* result;
        if (pointCol.GpuNullMaskPtr || std::get<2>(polygonCol))
        {
            int8_t* combinedMask;
            result = AllocateRegister<int8_t>(reg, retSize, &combinedMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            if (pointCol.GpuNullMaskPtr && std::get<2>(polygonCol))
            {
                GPUArithmetic::colCol<ArithmeticOperations::bitwiseOr>(
                    combinedMask, reinterpret_cast<int8_t*>(pointCol.GpuNullMaskPtr),
                    reinterpret_cast<int8_t*>(std::get<2>(polygonCol)), bitMaskSize);
            }
            else if (pointCol.GpuNullMaskPtr)
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(pointCol.GpuNullMaskPtr), bitMaskSize);
            }
            else if (std::get<2>(polygonCol))
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(std::get<2>(polygonCol)), bitMaskSize);
            }
        }
        else
        {
            result = AllocateRegister<int8_t>(reg, retSize);
        }
        GPUPolygonContains::contains(result, std::get<0>(polygonCol), std::get<1>(polygonCol),
                                     reinterpret_cast<NativeGeoPoint*>(pointCol.GpuPtr), pointCol.ElementCount);
    }
    return 0;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for constant constant case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
int32_t GpuSqlDispatcher::ContainsConstConst()
{
    // TODO : Specialize kernel for all cases.
    auto constPointWkt = arguments_.Read<std::string>();
    auto constPolygonWkt = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    std::cout << "ContainsConstConst: " + constPolygonWkt << " " << constPointWkt << " " << reg << std::endl;

    ColmnarDB::Types::Point constPoint = PointFactory::FromWkt(constPointWkt);
    ColmnarDB::Types::ComplexPolygon constPolygon = ComplexPolygonFactory::FromWkt(constPolygonWkt);

    NativeGeoPoint* constNativeGeoPoint = InsertConstPointGpu(constPoint);
    GPUMemory::GPUPolygon gpuPolygon = InsertConstPolygonGpu(constPolygon);

    int32_t retSize = database_->GetBlockSize();

    if (!IsRegisterAllocated(reg))
    {
        int8_t* result = AllocateRegister<int8_t>(reg, retSize);
        GPUPolygonContains::containsConst(result, gpuPolygon, constNativeGeoPoint, retSize);
    }
    return 0;
}

/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does
/// not meet this requrement) based on functor OP eg. INTRSECT(a,b), UNION(a,b) Implementation for
/// column constant case Pops data from argument memory stream, converts geo literals to their gpu
/// representation and loads data to GPU on demand <returns name="statusCode">Finish status code of
/// the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::PolygonOperationColConst()
{
    auto colName = arguments_.Read<std::string>();
    auto constWkt = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<U>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "PolygonOPConstCol: " + constWkt << " " << colName << " " << reg << std::endl;

    auto polygonLeft = FindComplexPolygon(colName);
    ColmnarDB::Types::ComplexPolygon polygonConst = ComplexPolygonFactory::FromWkt(constWkt);
    GPUMemory::GPUPolygon gpuPolygon = InsertConstPolygonGpu(polygonConst);

    int32_t retSize = std::get<1>(polygonLeft);

    if (!IsRegisterAllocated(reg))
    {
        // TODO
    }
    return 0;
}

/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does
/// not meet this requrement) based on functor OP eg. INTRSECT(a,b), UNION(a,b) Implementation for
/// constant column case Pops data from argument memory stream, converts geo literals to their gpu
/// representation and loads data to GPU on demand <returns name="statusCode">Finish status code of
/// the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::PolygonOperationConstCol()
{
    std::cout << "Polygon operation: " << std::endl;
    return 0;
}
/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does
/// not meet this requrement) based on functor OP eg. INTRSECT(a,b), UNION(a,b) Implementation for
/// column column case Pops data from argument memory stream, converts geo literals to their gpu
/// representation and loads data to GPU on demand <returns name="statusCode">Finish status code of
/// the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::PolygonOperationColCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    std::cout << "Polygon operation: " << colNameRight << " " << colNameLeft << " " << reg << std::endl;

    int32_t loadFlag = LoadCol<U>(colNameLeft);
    if (loadFlag)
    {
        return loadFlag;
    }
    loadFlag = LoadCol<T>(colNameRight);
    if (loadFlag)
    {
        return loadFlag;
    }

    auto polygonLeft = FindComplexPolygon(colNameLeft);
    auto polygonRight = FindComplexPolygon(colNameRight);

    int32_t dataSize = std::min(std::get<1>(polygonLeft), std::get<1>(polygonRight));
    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUPolygon outPolygon;
        GPUPolygonClipping::ColCol<OP>(outPolygon, std::get<0>(polygonLeft), std::get<0>(polygonRight), dataSize);
        if (std::get<2>(polygonLeft) || std::get<2>(polygonRight))
        {
            int32_t bitMaskSize = ((dataSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            int8_t* combinedMask = AllocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
            FillPolygonRegister(outPolygon, reg, dataSize, false, combinedMask);
            if (std::get<2>(polygonLeft) && std::get<2>(polygonRight))
            {
                GPUArithmetic::colCol<ArithmeticOperations::bitwiseOr>(
                    combinedMask, reinterpret_cast<int8_t*>(std::get<2>(polygonLeft)),
                    reinterpret_cast<int8_t*>(std::get<2>(polygonRight)), bitMaskSize);
            }
            else if (std::get<2>(polygonLeft))
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(std::get<2>(polygonLeft)), bitMaskSize);
            }
            else if (std::get<2>(polygonRight))
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(std::get<2>(polygonRight)), bitMaskSize);
            }
        }
        else
        {
            FillPolygonRegister(outPolygon, reg, dataSize);
        }
    }
    return 0;
}

/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does
/// not meet this requrement) based on functor OP eg. INTRSECT(a,b), UNION(a,b) Implementation for
/// constant constant case Pops data from argument memory stream, converts geo literals to their gpu
/// representation and loads data to GPU on demand <returns name="statusCode">Finish status code of
/// the operation</returns>
template <typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::PolygonOperationConstConst()
{
    std::cout << "Polygon operation: " << std::endl;
    return 0;
}