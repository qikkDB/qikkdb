#pragma once

#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUCast.cuh"

template <typename OUT, typename IN>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::CastNumericCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<IN>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "CastNumericCol: " << colName << " " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
            groupByColumns_.end() &&
        !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            PointerAllocation column = allocatedPointers_.at(colName + KEYS_SUFFIX);
            int32_t retSize = column.ElementCount;
            OUT* result;

            if (column.GpuNullMaskPtr)
            {
                int8_t* nullMask;
                result = AllocateRegister<OUT>(reg + KEYS_SUFFIX, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                result = AllocateRegister<OUT>(reg + KEYS_SUFFIX, retSize);
            }

            GPUCast::CastNumeric(result, reinterpret_cast<IN*>(column.GpuPtr), retSize);
            groupByColumns_.push_back({reg, ::GetColumnType<OUT>()});
        }
    }
    else if (isOverallLastBlock_ || !usingGroupBy_ || insideGroupBy_ || insideAggregation_)
    {
        PointerAllocation column = allocatedPointers_.at(colName);
        int32_t retSize = column.ElementCount;

        if (!IsRegisterAllocated(reg))
        {
            OUT* result;
            if (column.GpuNullMaskPtr)
            {
                int8_t* nullMask;
                result = AllocateRegister<OUT>(reg, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                result = AllocateRegister<OUT>(reg, retSize);
            }
            GPUCast::CastNumeric(result, reinterpret_cast<IN*>(column.GpuPtr), retSize);
        }
    }

    FreeColumnIfRegister<IN>(colName);
    return InstructionStatus::CONTINUE;
}

template <typename OUT, typename IN>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::CastNumericConst()
{
    IN cnst = arguments_.Read<IN>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "CastNumericConst: " << reg << '\n';

    int32_t retSize = GetBlockSize();

    if (!IsRegisterAllocated(reg))
    {
        OUT* result = AllocateRegister<OUT>(reg, retSize);
        GPUCast::CastNumeric(result, cnst, retSize);
    }
    return InstructionStatus::CONTINUE;
}

template <typename OUT>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::CastStringCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<std::string>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "CastStringCol: " << colName << " " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
            groupByColumns_.end() &&
        !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            auto column = FindStringColumn(colName + KEYS_SUFFIX);
            int32_t retSize = std::get<1>(column);
            OUT* result;

            if (std::get<2>(column))
            {
                int8_t* nullMask;
                result = AllocateRegister<OUT>(reg + KEYS_SUFFIX, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, std::get<2>(column), bitMaskSize);
            }
            else
            {
                result = AllocateRegister<OUT>(reg + KEYS_SUFFIX, retSize);
            }

            GPUCast::CastString(result, std::get<0>(column), retSize);
            groupByColumns_.push_back({reg, ::GetColumnType<OUT>()});
        }
    }
    else if (isOverallLastBlock_ || !usingGroupBy_ || insideGroupBy_ || insideAggregation_)
    {
        auto column = FindStringColumn(colName);
        int32_t retSize = std::get<1>(column);

        if (!IsRegisterAllocated(reg))
        {
            OUT* result;
            if (std::get<2>(column))
            {
                int8_t* nullMask;
                result = AllocateRegister<OUT>(reg, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, std::get<2>(column), bitMaskSize);
            }
            else
            {
                result = AllocateRegister<OUT>(reg, retSize);
            }
            GPUCast::CastString(result, std::get<0>(column), retSize);
        }
    }
    return InstructionStatus::CONTINUE;
}

template <typename OUT>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::CastStringConst()
{
    std::string cnst = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "CastStringConst: " << reg << '\n';

    GPUMemory::GPUString gpuString = InsertConstStringGpu(cnst);
    int32_t retSize = GetBlockSize();
    if (retSize == 0)
    {
        return InstructionStatus::OUT_OF_BLOCKS;
    }
    if (!IsRegisterAllocated(reg))
    {
        OUT* result = AllocateRegister<OUT>(reg, retSize);
        GPUCast::CastString(result, gpuString, retSize);
    }
    return InstructionStatus::CONTINUE;
}

template <typename IN>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::CastNumericToStringCol()
{
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<IN>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "CastNumericToStringCol: " << colName << " " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
            groupByColumns_.end() &&
        !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            PointerAllocation column = allocatedPointers_.at(colName + KEYS_SUFFIX);
            int32_t retSize = column.ElementCount;
            GPUMemory::GPUString result;
            GPUCast::CastNumericToString(&result, reinterpret_cast<IN*>(column.GpuPtr), retSize);

            if (column.GpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* nullMask = AllocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                FillStringRegister(result, reg + KEYS_SUFFIX, retSize, true, nullMask);
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
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
            GPUCast::CastNumericToString(&result, reinterpret_cast<IN*>(column.GpuPtr), retSize);

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
    }
    FreeColumnIfRegister<IN>(colName);
    return InstructionStatus::CONTINUE;
}

template <typename IN>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::CastNumericToStringConst()
{
    return InstructionStatus::CONTINUE;
}