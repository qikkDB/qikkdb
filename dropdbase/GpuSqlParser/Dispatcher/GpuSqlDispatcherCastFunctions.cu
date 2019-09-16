#include "GpuSqlDispatcherCastFunctions.h"
#include "../../QueryEngine/GPUCore/GPUReconstruct.cuh"
#include <array>

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::castToIntFunctions_ = {
    &GpuSqlDispatcher::CastNumericConst<int32_t, int32_t>,
    &GpuSqlDispatcher::CastNumericConst<int32_t, int64_t>,
    &GpuSqlDispatcher::CastNumericConst<int32_t, float>,
    &GpuSqlDispatcher::CastNumericConst<int32_t, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<int32_t, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<int32_t, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::CastStringConst<int32_t>,
    &GpuSqlDispatcher::CastNumericConst<int32_t, int8_t>,
    &GpuSqlDispatcher::CastNumericCol<int32_t, int32_t>,
    &GpuSqlDispatcher::CastNumericCol<int32_t, int64_t>,
    &GpuSqlDispatcher::CastNumericCol<int32_t, float>,
    &GpuSqlDispatcher::CastNumericCol<int32_t, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<int32_t, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<int32_t, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::CastStringCol<int32_t>,
    &GpuSqlDispatcher::CastNumericCol<int32_t, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::castToLongFunctions_ = {
    &GpuSqlDispatcher::CastNumericConst<int64_t, int32_t>,
    &GpuSqlDispatcher::CastNumericConst<int64_t, int64_t>,
    &GpuSqlDispatcher::CastNumericConst<int64_t, float>,
    &GpuSqlDispatcher::CastNumericConst<int64_t, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<int64_t, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<int64_t, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::CastStringConst<int64_t>,
    &GpuSqlDispatcher::CastNumericConst<int64_t, int8_t>,
    &GpuSqlDispatcher::CastNumericCol<int64_t, int32_t>,
    &GpuSqlDispatcher::CastNumericCol<int64_t, int64_t>,
    &GpuSqlDispatcher::CastNumericCol<int64_t, float>,
    &GpuSqlDispatcher::CastNumericCol<int64_t, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<int64_t, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<int64_t, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::CastStringCol<int64_t>,
    &GpuSqlDispatcher::CastNumericCol<int64_t, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::castToFloatFunctions_ = {
    &GpuSqlDispatcher::CastNumericConst<float, int32_t>,
    &GpuSqlDispatcher::CastNumericConst<float, int64_t>,
    &GpuSqlDispatcher::CastNumericConst<float, float>,
    &GpuSqlDispatcher::CastNumericConst<float, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<float, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<float, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::CastStringConst<float>,
    &GpuSqlDispatcher::CastNumericConst<float, int8_t>,
    &GpuSqlDispatcher::CastNumericCol<float, int32_t>,
    &GpuSqlDispatcher::CastNumericCol<float, int64_t>,
    &GpuSqlDispatcher::CastNumericCol<float, float>,
    &GpuSqlDispatcher::CastNumericCol<float, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<float, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<float, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::CastStringCol<float>,
    &GpuSqlDispatcher::CastNumericCol<float, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::castToDoubleFunctions_ = {
    &GpuSqlDispatcher::CastNumericConst<double, int32_t>,
    &GpuSqlDispatcher::CastNumericConst<double, int64_t>,
    &GpuSqlDispatcher::CastNumericConst<double, float>,
    &GpuSqlDispatcher::CastNumericConst<double, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<double, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<double, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::CastStringConst<double>,
    &GpuSqlDispatcher::CastNumericConst<double, int8_t>,
    &GpuSqlDispatcher::CastNumericCol<double, int32_t>,
    &GpuSqlDispatcher::CastNumericCol<double, int64_t>,
    &GpuSqlDispatcher::CastNumericCol<double, float>,
    &GpuSqlDispatcher::CastNumericCol<double, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<double, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<double, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::CastStringCol<double>,
    &GpuSqlDispatcher::CastNumericCol<double, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::castToPointFunctions_ = {
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point, int32_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point, int64_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::CastStringConst<NativeGeoPoint>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point, int8_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point, int32_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point, int64_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::CastStringCol<NativeGeoPoint>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::castToPolygonFunctions_ = {
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon, int32_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon, int64_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon, int8_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon, int32_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon, int64_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::castToStringFunctions_ = {
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<std::string, int32_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<std::string, int64_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<std::string, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<std::string, double>,
    &GpuSqlDispatcher::CastPointConst,
    &GpuSqlDispatcher::CastPolygonConst,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<std::string, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<std::string, int8_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<std::string, int32_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<std::string, int64_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<std::string, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<std::string, double>,
    &GpuSqlDispatcher::CastPointCol,
    &GpuSqlDispatcher::CastPolygonCol,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<std::string, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<std::string, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::castToInt8TFunctions_ = {
    &GpuSqlDispatcher::CastNumericConst<int8_t, int32_t>,
    &GpuSqlDispatcher::CastNumericConst<int8_t, int64_t>,
    &GpuSqlDispatcher::CastNumericConst<int8_t, float>,
    &GpuSqlDispatcher::CastNumericConst<int8_t, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<int8_t, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<int8_t, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<int8_t, std::string>,
    &GpuSqlDispatcher::CastNumericConst<int8_t, int8_t>,
    &GpuSqlDispatcher::CastNumericCol<int8_t, int32_t>,
    &GpuSqlDispatcher::CastNumericCol<int8_t, int64_t>,
    &GpuSqlDispatcher::CastNumericCol<int8_t, float>,
    &GpuSqlDispatcher::CastNumericCol<int8_t, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<int8_t, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<int8_t, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<int8_t, std::string>,
    &GpuSqlDispatcher::CastNumericCol<int8_t, int8_t>};


int32_t GpuSqlDispatcher::CastPolygonCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<ColmnarDB::Types::ComplexPolygon>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::info) << "CastPolygonCol: " << colName << " " << reg << '\n';

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

    CudaLogBoost::getInstance(CudaLogBoost::info) << "CastPolygonConst: " << constWkt << " " << reg << '\n';

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

    CudaLogBoost::getInstance(CudaLogBoost::info) << "CastPointCol: " << colName << " " << reg << '\n';

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

    CudaLogBoost::getInstance(CudaLogBoost::info) << "CastPointConst: " << constWkt << " " << reg << '\n';

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
