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
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::PointColCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "PointColCol: " << colNameLeft << " " << colNameRight << " " << reg << '\n';

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<U>(colNameRight);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }
    loadFlag = LoadCol<T>(colNameLeft);
    if (loadFlag != InstructionStatus::CONTINUE)
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
            int64_t* combinedMask;
            pointCol = AllocateRegister<NativeGeoPoint>(reg, retSize, &combinedMask);
            int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
            if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
            {
                GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                    combinedMask, reinterpret_cast<int64_t*>(columnLeft.GpuNullMaskPtr),
                    reinterpret_cast<int64_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
            }
            else if (columnLeft.GpuNullMaskPtr)
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int64_t*>(columnLeft.GpuNullMaskPtr), bitMaskSize);
            }
            else if (columnRight.GpuNullMaskPtr)
            {
                GPUMemory::copyDeviceToDevice(combinedMask, reinterpret_cast<int64_t*>(columnRight.GpuNullMaskPtr),
                                              bitMaskSize);
            }
        }
        else
        {
            pointCol = AllocateRegister<NativeGeoPoint>(reg, retSize);
        }
        GPUConversion::Convert(pointCol, reinterpret_cast<T*>(columnLeft.GpuPtr),
                               reinterpret_cast<U*>(columnRight.GpuPtr), retSize);
    }

    FreeColumnIfRegister<U>(colNameRight);
    FreeColumnIfRegister<T>(colNameLeft);
    return InstructionStatus::CONTINUE;
}

/// Implementation of POINT(a, b) operation dispatching - concatenation of two numeric attributes to single point column
/// Implementation for column constant case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::PointColConst()
{
    U cnst = arguments_.Read<U>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "PointColConst: " << colNameLeft << " " << reg << '\n';

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<T>(colNameLeft);
    if (loadFlag != InstructionStatus::CONTINUE)
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
            int64_t* nullMask;
            pointCol = AllocateRegister<NativeGeoPoint>(reg, retSize, &nullMask);
            int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
            GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int64_t*>(columnLeft.GpuNullMaskPtr),
                                          bitMaskSize);
        }
        else
        {
            pointCol = AllocateRegister<NativeGeoPoint>(reg, retSize);
        }
        GPUConversion::Convert(pointCol, reinterpret_cast<T*>(columnLeft.GpuPtr), cnst, retSize);
    }

    FreeColumnIfRegister<T>(colNameLeft);
    return InstructionStatus::CONTINUE;
}

/// Implementation of POINT(a, b) operation dispatching - concatenation of two numeric attributes to single point column
/// Implementation for onstant column case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::PointConstCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    T cnst = arguments_.Read<T>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "PointConstCol: " << colNameRight << " " << reg << '\n';

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<U>(colNameRight);
    if (loadFlag != InstructionStatus::CONTINUE)
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
            int64_t* nullMask;
            pointCol = AllocateRegister<NativeGeoPoint>(reg, retSize, &nullMask);
            int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
            GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int64_t*>(columnRight.GpuNullMaskPtr),
                                          bitMaskSize);
        }
        else
        {
            pointCol = AllocateRegister<NativeGeoPoint>(reg, retSize);
        }
        GPUConversion::Convert(pointCol, cnst, reinterpret_cast<U*>(columnRight.GpuPtr), retSize);
    }

    FreeColumnIfRegister<U>(colNameRight);
    return InstructionStatus::CONTINUE;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for column constant case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ContainsColConst()
{
    auto constWkt = arguments_.Read<std::string>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<T>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "ContainsColConst: " + colName << " " << constWkt << " " << reg << '\n';

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
            int64_t* nullMask;
            result = AllocateRegister<int8_t>(reg, retSize, &nullMask);
            int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
            GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int64_t*>(std::get<2>(polygonCol)), bitMaskSize);
        }
        else
        {
            result = AllocateRegister<int8_t>(reg, retSize);
        }
        GPUPolygonContains::contains(result, polygons, retSize, pointConstPtr, 1);
    }
    return InstructionStatus::CONTINUE;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for constant column case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ContainsConstCol()
{
    auto colName = arguments_.Read<std::string>();
    auto constWkt = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<U>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "ContainsConstCol: " + constWkt << " " << colName << " " << reg << '\n';

    PointerAllocation columnPoint = allocatedPointers_.at(colName);
    ColmnarDB::Types::ComplexPolygon polygonConst = ComplexPolygonFactory::FromWkt(constWkt);
    GPUMemory::GPUPolygon gpuPolygon = InsertConstPolygonGpu(polygonConst);

    int32_t retSize = columnPoint.ElementCount;

    if (!IsRegisterAllocated(reg))
    {
        int8_t* result;
        if (columnPoint.GpuNullMaskPtr)
        {
            int64_t* nullMask;
            result = AllocateRegister<int8_t>(reg, retSize, &nullMask);
            int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
            GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int64_t*>(columnPoint.GpuNullMaskPtr),
                                          bitMaskSize);
        }
        else
        {
            result = AllocateRegister<int8_t>(reg, retSize);
        }
        GPUPolygonContains::contains(result, gpuPolygon, 1,
                                     reinterpret_cast<NativeGeoPoint*>(columnPoint.GpuPtr), retSize);
    }
    return InstructionStatus::CONTINUE;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for column column case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ContainsColCol()
{
    auto colNamePoint = arguments_.Read<std::string>();
    auto colNamePolygon = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<U>(colNamePoint);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }
    loadFlag = LoadCol<T>(colNamePolygon);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "ContainsColCol: " + colNamePolygon << " " << colNamePoint << " " << reg << '\n';

    PointerAllocation pointCol = allocatedPointers_.at(colNamePoint);
    auto polygonCol = FindComplexPolygon(colNamePolygon);


    int32_t retSize = std::min(pointCol.ElementCount, std::get<1>(polygonCol));

    if (!IsRegisterAllocated(reg))
    {
        int8_t* result;
        if (pointCol.GpuNullMaskPtr || std::get<2>(polygonCol))
        {
            int64_t* combinedMask;
            result = AllocateRegister<int8_t>(reg, retSize, &combinedMask);
            int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
            if (pointCol.GpuNullMaskPtr && std::get<2>(polygonCol))
            {
                GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                    combinedMask, reinterpret_cast<int64_t*>(pointCol.GpuNullMaskPtr),
                    reinterpret_cast<int64_t*>(std::get<2>(polygonCol)), bitMaskSize);
            }
            else if (pointCol.GpuNullMaskPtr)
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int64_t*>(pointCol.GpuNullMaskPtr), bitMaskSize);
            }
            else if (std::get<2>(polygonCol))
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int64_t*>(std::get<2>(polygonCol)), bitMaskSize);
            }
        }
        else
        {
            result = AllocateRegister<int8_t>(reg, retSize);
        }
        GPUPolygonContains::contains(result, std::get<0>(polygonCol), std::get<1>(polygonCol),
                                     reinterpret_cast<NativeGeoPoint*>(pointCol.GpuPtr), pointCol.ElementCount);
    }
    return InstructionStatus::CONTINUE;
}

/// Implementation of CONTAINS(a, b) operation dispatching - point in polygon
/// Implementation for constant constant case
/// Pops data from argument memory stream, converts geo literals to their gpu representation and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ContainsConstConst()
{
    // TODO : Specialize kernel for all cases.
    auto constPointWkt = arguments_.Read<std::string>();
    auto constPolygonWkt = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "ContainsConstConst: " + constPolygonWkt << " " << constPointWkt << " " << reg << '\n';

    ColmnarDB::Types::Point constPoint = PointFactory::FromWkt(constPointWkt);
    ColmnarDB::Types::ComplexPolygon constPolygon = ComplexPolygonFactory::FromWkt(constPolygonWkt);

    NativeGeoPoint* constNativeGeoPoint = InsertConstPointGpu(constPoint);
    GPUMemory::GPUPolygon gpuPolygon = InsertConstPolygonGpu(constPolygon);

    int32_t retSize = GetBlockSize();
    if (retSize == 0)
    {
        return InstructionStatus::OUT_OF_BLOCKS;
    }
    if (!IsRegisterAllocated(reg))
    {
        int8_t* result = AllocateRegister<int8_t>(reg, retSize);
        GPUPolygonContains::containsConst(result, gpuPolygon, constNativeGeoPoint, retSize);
    }
    return InstructionStatus::CONTINUE;
}

/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does
/// not meet this requrement) based on functor OP eg. INTRSECT(a,b), UNION(a,b) Implementation for
/// column constant case Pops data from argument memory stream, converts geo literals to their gpu
/// representation and loads data to GPU on demand <returns name="statusCode">Finish status code of
/// the operation</returns>
template <typename OP, typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::PolygonOperationColConst()
{
    auto constWkt = arguments_.Read<std::string>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<T>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "PolygonOPConstCol: " + constWkt << " " << colName << " " << reg << '\n';

    auto polygonCol = FindComplexPolygon(colName);
    ColmnarDB::Types::ComplexPolygon polygonConst = ComplexPolygonFactory::FromWkt(constWkt);
    GPUMemory::GPUPolygon gpuPolygonConst = InsertConstPolygonGpu(polygonConst);

    int32_t dataSize = std::get<1>(polygonCol);

    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUPolygon outPolygon;
        GPUPolygonClipping::ColConst<OP>(outPolygon, std::get<0>(polygonCol), gpuPolygonConst, dataSize);
        if (std::get<2>(polygonCol))
        {
            int32_t bitMaskSize = NullValues::GetNullBitMaskSize(dataSize);
            int64_t* combinedMask = AllocateRegister<int64_t>(reg + NULL_SUFFIX, bitMaskSize);
            GPUMemory::copyDeviceToDevice(combinedMask,
                                          reinterpret_cast<int64_t*>(std::get<2>(polygonCol)), bitMaskSize);
            FillPolygonRegister(outPolygon, reg, dataSize, false, combinedMask);
        }
        else
        {
            FillPolygonRegister(outPolygon, reg, dataSize);
        }
    }
    return InstructionStatus::CONTINUE;
}

/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does
/// not meet this requrement) based on functor OP eg. INTRSECT(a,b), UNION(a,b) Implementation for
/// constant column case Pops data from argument memory stream, converts geo literals to their gpu
/// representation and loads data to GPU on demand <returns name="statusCode">Finish status code of
/// the operation</returns>
template <typename OP, typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::PolygonOperationConstCol()
{
    auto colName = arguments_.Read<std::string>();
    auto constWkt = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<U>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    auto polygonCol = FindComplexPolygon(colName);
    ColmnarDB::Types::ComplexPolygon polygonConst = ComplexPolygonFactory::FromWkt(constWkt);
    GPUMemory::GPUPolygon gpuPolygonConst = InsertConstPolygonGpu(polygonConst);

    int32_t dataSize = std::get<1>(polygonCol);

    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUPolygon outPolygon;
        GPUPolygonClipping::ColConst<OP>(outPolygon, std::get<0>(polygonCol), gpuPolygonConst, dataSize);
        if (std::get<2>(polygonCol))
        {
            int32_t bitMaskSize = NullValues::GetNullBitMaskSize(dataSize);
            int64_t* combinedMask = AllocateRegister<int64_t>(reg + NULL_SUFFIX, bitMaskSize);
            GPUMemory::copyDeviceToDevice(combinedMask,
                                          reinterpret_cast<int64_t*>(std::get<2>(polygonCol)), bitMaskSize);
            FillPolygonRegister(outPolygon, reg, dataSize, false, combinedMask);
        }
        else
        {
            FillPolygonRegister(outPolygon, reg, dataSize);
        }
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Polygon operation: " << '\n';

    return InstructionStatus::CONTINUE;
}
/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does
/// not meet this requrement) based on functor OP eg. INTRSECT(a,b), UNION(a,b) Implementation for
/// column column case Pops data from argument memory stream, converts geo literals to their gpu
/// representation and loads data to GPU on demand <returns name="statusCode">Finish status code of
/// the operation</returns>
template <typename OP, typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::PolygonOperationColCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "Polygon operation: " << colNameRight << " " << colNameLeft << " " << reg << '\n';

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<T>(colNameLeft);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }
    loadFlag = LoadCol<U>(colNameRight);
    if (loadFlag != InstructionStatus::CONTINUE)
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
            int32_t bitMaskSize = NullValues::GetNullBitMaskSize(dataSize);
            int64_t* combinedMask = AllocateRegister<int64_t>(reg + NULL_SUFFIX, bitMaskSize);
            FillPolygonRegister(outPolygon, reg, dataSize, false, combinedMask);
            if (std::get<2>(polygonLeft) && std::get<2>(polygonRight))
            {
                GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                    combinedMask, reinterpret_cast<int64_t*>(std::get<2>(polygonLeft)),
                    reinterpret_cast<int64_t*>(std::get<2>(polygonRight)), bitMaskSize);
            }
            else if (std::get<2>(polygonLeft))
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int64_t*>(std::get<2>(polygonLeft)), bitMaskSize);
            }
            else if (std::get<2>(polygonRight))
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int64_t*>(std::get<2>(polygonRight)), bitMaskSize);
            }
        }
        else
        {
            FillPolygonRegister(outPolygon, reg, dataSize);
        }
    }
    return InstructionStatus::CONTINUE;
}

/// Implementation of genric polygon operation (operation which also outputs polygon - CONTAINS does
/// not meet this requrement) based on functor OP eg. INTRSECT(a,b), UNION(a,b) Implementation for
/// constant constant case Pops data from argument memory stream, converts geo literals to their gpu
/// representation and loads data to GPU on demand <returns name="statusCode">Finish status code of
/// the operation</returns>
template <typename OP, typename T, typename U>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::PolygonOperationConstConst()
{
    
    auto constWktRight = arguments_.Read<std::string>();
    auto constWktLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    ColmnarDB::Types::ComplexPolygon polygonConstLeft = ComplexPolygonFactory::FromWkt(constWktLeft);
    ColmnarDB::Types::ComplexPolygon polygonConstRight = ComplexPolygonFactory::FromWkt(constWktRight);
    GPUMemory::GPUPolygon gpuPolygonConstLeft = InsertConstPolygonGpu(polygonConstLeft);
    GPUMemory::GPUPolygon gpuPolygonConstRight = InsertConstPolygonGpu(polygonConstRight);

    int32_t dataSize = 1;

    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUPolygon outPolygon;
        GPUPolygonClipping::ConstConst<OP>(outPolygon, gpuPolygonConstLeft, gpuPolygonConstRight, dataSize);
        FillPolygonRegister(outPolygon, reg, dataSize);
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Polygon operation: " << '\n';

    return InstructionStatus::CONTINUE;
}