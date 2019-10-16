#include "GpuSqlDispatcherDateFunctions.h"
#include <array>
#include "DispatcherMacros.h"

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::dateToStringFunctions_ = {
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, int32_t>,
    &GpuSqlDispatcher::DateToStringConst,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::toString, int8_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, int32_t>,
    &GpuSqlDispatcher::DateToStringCol,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::toString, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::yearFunctions_ = {
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, int32_t>,
    &GpuSqlDispatcher::DateExtractConst<DateOperations::year>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::year, int8_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, int32_t>,
    &GpuSqlDispatcher::DateExtractCol<DateOperations::year>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::year, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::monthFunctions_ = {
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, int32_t>,
    &GpuSqlDispatcher::DateExtractConst<DateOperations::month>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::month, int8_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, int32_t>,
    &GpuSqlDispatcher::DateExtractCol<DateOperations::month>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::month, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::dayFunctions_ = {
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, int32_t>,
    &GpuSqlDispatcher::DateExtractConst<DateOperations::day>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::day, int8_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, int32_t>,
    &GpuSqlDispatcher::DateExtractCol<DateOperations::day>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::day, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::hourFunctions_ = {
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, int32_t>,
    &GpuSqlDispatcher::DateExtractConst<DateOperations::hour>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::hour, int8_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, int32_t>,
    &GpuSqlDispatcher::DateExtractCol<DateOperations::hour>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::hour, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::minuteFunctions_ = {
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, int32_t>,
    &GpuSqlDispatcher::DateExtractConst<DateOperations::minute>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::minute, int8_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, int32_t>,
    &GpuSqlDispatcher::DateExtractCol<DateOperations::minute>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::minute, int8_t>};
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::secondFunctions_ = {
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, int32_t>,
    &GpuSqlDispatcher::DateExtractConst<DateOperations::second>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<DateOperations::second, int8_t>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, int32_t>,
    &GpuSqlDispatcher::DateExtractCol<DateOperations::second>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, float>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, double>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, ColmnarDB::Types::Point>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, ColmnarDB::Types::ComplexPolygon>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, std::string>,
    &GpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<DateOperations::second, int8_t>};

int32_t GpuSqlDispatcher::DateToStringCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    int32_t loadFlag = LoadCol<int64_t>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "DateToStringCol: " << colName << " " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
            groupByColumns_.end() &&
        !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            PointerAllocation column = allocatedPointers_.at(colName + KEYS_SUFFIX);
            int32_t retSize = column.ElementCount;
            GPUMemory::GPUString result;
            GPUDate::DateToString(&result, reinterpret_cast<int64_t*>(column.GpuPtr), retSize);
            if (column.GpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = AllocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                FillStringRegister(result, reg + KEYS_SUFFIX, retSize, true, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                FillStringRegister(result, reg + KEYS_SUFFIX, retSize, true);
            }
            groupByColumns_.push_back({reg, DataType::COLUMN_STRING});
        }
    }
    else if (isOverallLastBlock_ || !usingGroupBy_ || insideGroupBy_ || insideAggregation_)
    {
        PointerAllocation column = allocatedPointers_.at(colName);
        int32_t retSize = column.ElementCount;

        if (!IsRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUDate::DateToString(&result, reinterpret_cast<int64_t*>(column.GpuPtr), retSize);
            if (column.GpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = AllocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
                FillStringRegister(result, reg, retSize, true, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                FillStringRegister(result, reg, retSize, true);
            }
        }
    }
    FreeColumnIfRegister<int64_t>(colName);
    return 0;
}

int32_t GpuSqlDispatcher::DateToStringConst()
{
    int64_t cnst = arguments_.Read<int64_t>();
    auto reg = arguments_.Read<std::string>();
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "DateToStringConst: " << cnst << " " << reg << '\n';

    int32_t retSize = GetBlockSize();
    if (retSize == 0)
    {
        return 1;
    }
    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUString result;
        GPUDate::DateToString(&result, cnst, retSize);
        FillStringRegister(result, reg, retSize, true);
    }
    return 0;
}
