#include "GpuSqlDispatcherCastFunctions.h"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include <array>
#include "DispatcherMacros.h"
BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToIntFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int32_t, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int32_t, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int32_t, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int32_t, double)
DISPATCHER_UNARY_ERROR(int32_t, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(int32_t, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastString, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int32_t, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToLongFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int64_t, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int64_t, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int64_t, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int64_t, double)
DISPATCHER_UNARY_ERROR(int64_t, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(int64_t, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastString, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int64_t, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToFloatFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, float, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, float, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, float, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, float, double)
DISPATCHER_UNARY_ERROR(float, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(float, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastString, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, float, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToDoubleFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, double, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, double, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, double, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, double, double)
DISPATCHER_UNARY_ERROR(double, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(double, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastString, double)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, double, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToPointFunctions_)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, int32_t)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, int64_t)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, float)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, double)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastString, NativeGeoPoint)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::Point, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToPolygonFunctions_)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, int32_t)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, int64_t)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, float)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, double)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, std::string)
DISPATCHER_UNARY_ERROR(ColmnarDB::Types::ComplexPolygon, int8_t)
END_DISPATCH_TABLE


BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToStringFunctions_)
DISPATCHER_ERR(Const, std::string, int32_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::CastNumericToStringCol, int32_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(Const, std::string, int64_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::CastNumericToStringCol, int64_t)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(Const, std::string, float)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::CastNumericToStringCol, float)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_ERR(Const, std::string, double)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_FUN(GpuSqlDispatcher::CastNumericToStringCol, double)
DISPATCH_ENTRY_SEPARATOR
DISPATCHER_UNARY_FUNCTION_NO_TEMPLATE(GpuSqlDispatcher::CastPoint)
DISPATCHER_UNARY_FUNCTION_NO_TEMPLATE(GpuSqlDispatcher::CastPolygon)
DISPATCHER_UNARY_ERROR(std::string, std::string)
DISPATCHER_UNARY_ERROR(std::string, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::castToInt8TFunctions_)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int8_t, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int8_t, int64_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int8_t, float)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int8_t, double)
DISPATCHER_UNARY_ERROR(int8_t, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(int8_t, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(int8_t, std::string)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::CastNumeric, int8_t, int8_t)
END_DISPATCH_TABLE

int32_t GpuSqlDispatcher::CastPolygonCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<ColmnarDB::Types::ComplexPolygon>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "CastPolygonCol: " << colName << " " << reg << '\n';

    auto column = FindComplexPolygon(colName);
    int32_t retSize = std::get<1>(column);

    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUString result;
        GPUReconstruct::ConvertPolyColToWKTCol(&result, std::get<0>(column), retSize);
        if (std::get<2>(column))
        {
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            int8_t* nullMask = AllocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
            FillStringRegister(result, reg, retSize, true, nullMask);
            GPUMemory::copyDeviceToDevice(nullMask, std::get<2>(column), bitMaskSize);
        }
        else
        {
            FillStringRegister(result, reg, retSize, true);
        }
    }

    return 0;
}

int32_t GpuSqlDispatcher::CastPolygonConst()
{
    auto constWkt = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "CastPolygonConst: " << constWkt << " " << reg << '\n';

    ColmnarDB::Types::ComplexPolygon constPolygon = ComplexPolygonFactory::FromWkt(constWkt);
    GPUMemory::GPUPolygon gpuPolygon = InsertConstPolygonGpu(constPolygon);

    int32_t retSize = GetBlockSize();

    if (retSize == 0)
    {
        return 1;
    }
    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUString result;
        GPUReconstruct::ConvertPolyColToWKTCol(&result, gpuPolygon, retSize);
        FillStringRegister(result, reg, retSize, true);
    }

    return 0;
}

int32_t GpuSqlDispatcher::CastPointCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<ColmnarDB::Types::Point>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "CastPointCol: " << colName << " " << reg << '\n';

    auto column = allocatedPointers_.at(colName);
    int32_t retSize = column.ElementCount;

    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUString result;
        GPUReconstruct::ConvertPointColToWKTCol(&result, reinterpret_cast<NativeGeoPoint*>(column.GpuPtr), retSize);
        if (column.GpuNullMaskPtr)
        {
            int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
            int8_t* nullMask = AllocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
            FillStringRegister(result, reg, retSize, true, nullMask);
            GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
        }
        else
        {
            FillStringRegister(result, reg, retSize, true);
        }
    }

    return 0;
}

int32_t GpuSqlDispatcher::CastPointConst()
{
    auto constWkt = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "CastPointConst: " << constWkt << " " << reg << '\n';

    ColmnarDB::Types::Point constPoint = PointFactory::FromWkt(constWkt);
    NativeGeoPoint* gpuPoint = InsertConstPointGpu(constPoint);

    int32_t retSize = GetBlockSize();
    if (retSize == 0)
    {
        return 1;
    }
    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUString result;
        GPUReconstruct::ConvertPointColToWKTCol(&result, gpuPoint, retSize);
        FillStringRegister(result, reg, retSize, true);
    }

    return 0;
}
