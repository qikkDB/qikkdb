#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"
#include "../../QueryEngine/GPUCore/GPUPolygonContains.cuh"
#include "../../QueryEngine/GPUCore/GPUConversion.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include "../../ComplexPolygonFactory.h"
#include "../../PointFactory.h"
#include "../../Database.h"

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

    auto polygonCol = FindCompositeDataTypeAllocation<ColmnarDB::Types::ComplexPolygon>(colName);
    ColmnarDB::Types::Point pointConst = PointFactory::FromWkt(constWkt);

    NativeGeoPoint* pointConstPtr = InsertConstPointGpu(pointConst);
    int32_t retSize = polygonCol.ElementCount;

    if (!IsRegisterAllocated(reg))
    {
        int8_t* result;
        if (polygonCol.GpuNullMaskPtr)
        {
            int8_t* nullMask;
            result = AllocateRegister<int8_t>(reg, retSize, &nullMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(polygonCol.GpuNullMaskPtr), bitMaskSize);
        }
        else
        {
            result = AllocateRegister<int8_t>(reg, retSize);
        }
        GPUPolygonContains::contains(result, polygonCol.GpuPtr, retSize, pointConstPtr, 1);
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
    GPUMemory::GPUPolygon gpuPolygon = InsertConstCompositeDataType<ColmnarDB::Types::ComplexPolygon>(constWkt);

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
    auto polygonCol = FindCompositeDataTypeAllocation<ColmnarDB::Types::ComplexPolygon>(colNamePolygon);


    int32_t retSize = std::min(pointCol.ElementCount, polygonCol.ElementCount);

    if (!IsRegisterAllocated(reg))
    {
        int8_t* result;
        if (pointCol.GpuNullMaskPtr || polygonCol.GpuNullMaskPtr)
        {
            int8_t* combinedMask;
            result = AllocateRegister<int8_t>(reg, retSize, &combinedMask);
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            if (pointCol.GpuNullMaskPtr && polygonCol.GpuNullMaskPtr)
            {
                GPUArithmetic<ArithmeticOperations::bitwiseOr, int8_t, int8_t*, int8_t*>::Arithmetic(
                    combinedMask, reinterpret_cast<int8_t*>(pointCol.GpuNullMaskPtr),
                    reinterpret_cast<int8_t*>(polygonCol.GpuNullMaskPtr), bitMaskSize);
            }
            else if (pointCol.GpuNullMaskPtr)
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(pointCol.GpuNullMaskPtr), bitMaskSize);
            }
            else if (polygonCol.GpuNullMaskPtr)
            {
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(polygonCol.GpuNullMaskPtr), bitMaskSize);
            }
        }
        else
        {
            result = AllocateRegister<int8_t>(reg, retSize);
        }
        GPUPolygonContains::contains(result, polygonCol.GpuPtr, polygonCol.ElementCount,
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

    NativeGeoPoint* constNativeGeoPoint = InsertConstPointGpu(constPoint);
    GPUMemory::GPUPolygon gpuPolygon =
        InsertConstCompositeDataType<ColmnarDB::Types::ComplexPolygon>(constPolygonWkt);

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