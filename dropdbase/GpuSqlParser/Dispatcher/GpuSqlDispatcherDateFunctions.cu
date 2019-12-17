#include "GpuSqlDispatcherDateFunctions.h"
#include <array>
#include "DispatcherMacros.h"

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::dateToStringFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::toString, int32_t)
DISPATCHER_UNARY_FUNCTION_NO_TEMPLATE(GpuSqlDispatcher::DateToString)
DISPATCHER_UNARY_ERROR(DateOperations::toString, float)
DISPATCHER_UNARY_ERROR(DateOperations::toString, double)
DISPATCHER_UNARY_ERROR(DateOperations::toString, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::toString, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::toString, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::toString, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::yearFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::year, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::DateExtract, DateOperations::year)
DISPATCHER_UNARY_ERROR(DateOperations::year, float)
DISPATCHER_UNARY_ERROR(DateOperations::year, double)
DISPATCHER_UNARY_ERROR(DateOperations::year, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::year, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::year, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::year, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::monthFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::month, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::DateExtract, DateOperations::month)
DISPATCHER_UNARY_ERROR(DateOperations::month, float)
DISPATCHER_UNARY_ERROR(DateOperations::month, double)
DISPATCHER_UNARY_ERROR(DateOperations::month, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::month, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::month, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::month, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::dayFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::day, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::DateExtract, DateOperations::day)
DISPATCHER_UNARY_ERROR(DateOperations::day, float)
DISPATCHER_UNARY_ERROR(DateOperations::day, double)
DISPATCHER_UNARY_ERROR(DateOperations::day, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::day, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::day, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::day, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::hourFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::hour, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::DateExtract, DateOperations::hour)
DISPATCHER_UNARY_ERROR(DateOperations::hour, float)
DISPATCHER_UNARY_ERROR(DateOperations::hour, double)
DISPATCHER_UNARY_ERROR(DateOperations::hour, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::hour, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::hour, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::hour, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::minuteFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::minute, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::DateExtract, DateOperations::minute)
DISPATCHER_UNARY_ERROR(DateOperations::minute, float)
DISPATCHER_UNARY_ERROR(DateOperations::minute, double)
DISPATCHER_UNARY_ERROR(DateOperations::minute, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::minute, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::minute, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::minute, int8_t)
END_DISPATCH_TABLE

BEGIN_UNARY_DISPATCH_TABLE(GpuSqlDispatcher::secondFunctions_)
DISPATCHER_UNARY_ERROR(DateOperations::second, int32_t)
DISPATCHER_UNARY_FUNCTION(GpuSqlDispatcher::DateExtract, DateOperations::second)
DISPATCHER_UNARY_ERROR(DateOperations::second, float)
DISPATCHER_UNARY_ERROR(DateOperations::second, double)
DISPATCHER_UNARY_ERROR(DateOperations::second, ColmnarDB::Types::Point)
DISPATCHER_UNARY_ERROR(DateOperations::second, ColmnarDB::Types::ComplexPolygon)
DISPATCHER_UNARY_ERROR(DateOperations::second, std::string)
DISPATCHER_UNARY_ERROR(DateOperations::second, int8_t)
END_DISPATCH_TABLE


GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::DateToStringCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<int64_t>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
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
    return InstructionStatus::CONTINUE;
}

GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::DateToStringConst()
{
    int64_t cnst = arguments_.Read<int64_t>();
    auto reg = arguments_.Read<std::string>();
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "DateToStringConst: " << cnst << " " << reg << '\n';

    int32_t retSize = GetBlockSize();
    if (retSize == 0)
    {
        return InstructionStatus::OUT_OF_BLOCKS;
    }
    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUString result;
        GPUDate::DateToString(&result, cnst, retSize);
        FillStringRegister(result, reg, retSize, true);
    }
    return InstructionStatus::CONTINUE;
}
