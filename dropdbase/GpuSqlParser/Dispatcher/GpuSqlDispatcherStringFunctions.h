#pragma once

#include "../ParserExceptions.h"
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"

#include "../../QueryEngine/GPUCore/GPUStringUnary.cuh"
#include "../../QueryEngine/GPUCore/GPUStringBinary.cuh"

template <typename OP, typename T>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::StringBinaryNumericColCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<std::string>(colNameLeft);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    loadFlag = LoadCol<T>(colNameRight);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "StringBinaryNumericColCol: " << colNameLeft << " " << colNameRight << " " << reg << '\n';

    bool isLeftKey = std::find_if(groupByColumns_.begin(), groupByColumns_.end(),
                                  StringDataTypeComp(colNameLeft)) != groupByColumns_.end() &&
                     !insideAggregation_;
    bool isRightKey = std::find_if(groupByColumns_.begin(), groupByColumns_.end(),
                                   StringDataTypeComp(colNameRight)) != groupByColumns_.end() &&
                      !insideAggregation_;

    if (isLeftKey || isRightKey)
    {
        if (isOverallLastBlock_)
        {
            auto columnLeft = FindStringColumn(colNameLeft + (isLeftKey ? KEYS_SUFFIX : ""));
            auto columnRight = allocatedPointers_.at(colNameRight + (isRightKey ? KEYS_SUFFIX : ""));
            int32_t retSize = std::min(columnLeft.ElementCount, columnRight.ElementCount);

            GPUMemory::GPUString result;
            GPUStringBinary::ColCol<OP>(result, columnLeft.GpuPtr,
                                        reinterpret_cast<T*>(columnRight.GpuPtr), retSize);
            if (columnLeft.GpuNullMaskPtr || columnRight.GpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = AllocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                FillStringRegister(result, reg + KEYS_SUFFIX, retSize, true, combinedMask);
                if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
                {
                    GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                        combinedMask, reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                        reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
                }
                else if (columnLeft.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                else if (columnRight.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
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
        auto columnLeft = FindStringColumn(colNameLeft);
        auto columnRight = allocatedPointers_.at(colNameRight);
        int32_t retSize = std::min(columnLeft.ElementCount, columnRight.ElementCount);

        if (!IsRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUStringBinary::ColCol<OP>(result, columnLeft.GpuPtr,
                                        reinterpret_cast<T*>(columnRight.GpuPtr), retSize);
            if (columnLeft.GpuNullMaskPtr || columnRight.GpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = AllocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
                FillStringRegister(result, reg, retSize, true, combinedMask);
                if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
                {
                    GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                        combinedMask, reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                        reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
                }
                else if (columnLeft.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                else if (columnRight.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
            }
            else
            {
                FillStringRegister(result, reg, retSize, true);
            }
        }
    }
    return InstructionStatus::CONTINUE;
}

template <typename OP, typename T>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::StringBinaryNumericColConst()
{
    T cnst = arguments_.Read<T>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<std::string>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "StringBinaryColConst: " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
            groupByColumns_.end() &&
        !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            auto column = FindStringColumn(colName + KEYS_SUFFIX);
            int32_t retSize = column.ElementCount;
            GPUMemory::GPUString result;
            GPUStringBinary::ColConst<OP>(result, column.GpuPtr, cnst, retSize);
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
        auto column = FindStringColumn(colName);
        int32_t retSize = column.ElementCount;

        if (!IsRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUStringBinary::ColConst<OP>(result, column.GpuPtr, cnst, retSize);
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
    return InstructionStatus::CONTINUE;
}

template <typename OP, typename T>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::StringBinaryNumericConstCol()
{
    auto colName = arguments_.Read<std::string>();
    std::string cnst = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<std::string>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "StringBinaryConstCol: " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
            groupByColumns_.end() &&
        !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            GPUMemory::GPUString gpuString = InsertConstStringGpu(cnst);
            PointerAllocation column = allocatedPointers_.at(colName + KEYS_SUFFIX);
            int32_t retSize = column.ElementCount;
            GPUMemory::GPUString result;
            GPUStringBinary::ConstCol<OP>(result, gpuString, reinterpret_cast<T*>(column.GpuPtr), retSize);
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
        GPUMemory::GPUString gpuString = InsertConstStringGpu(cnst);
        PointerAllocation column = allocatedPointers_.at(colName);
        int32_t retSize = column.ElementCount;

        if (!IsRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUStringBinary::ConstCol<OP>(result, gpuString, reinterpret_cast<T*>(column.GpuPtr), retSize);
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
    return InstructionStatus::CONTINUE;
}

template <typename OP, typename T>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::StringBinaryNumericConstConst()
{
    T cnstRight = arguments_.Read<T>();
    std::string cnstLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "StringBinaryConstConst: " << reg << '\n';
    int32_t retSize = GetBlockSize();
    if (retSize == 0)
    {
        return InstructionStatus::OUT_OF_BLOCKS;
    }
    GPUMemory::GPUString gpuString = InsertConstStringGpu(cnstLeft, retSize);

    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUString result;
        GPUStringBinary::ConstConst<OP>(result, gpuString, cnstRight, retSize);
        FillStringRegister(result, reg, retSize, true);
    }
    return InstructionStatus::CONTINUE;
}

template <typename OP>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::StringBinaryColCol()
{
    auto colNameRight = arguments_.Read<std::string>();
    auto colNameLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<std::string>(colNameLeft);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    loadFlag = LoadCol<std::string>(colNameRight);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug)
        << "StringBinaryColCol: " << colNameLeft << " " << colNameRight << " " << reg << '\n';

    bool isLeftKey = std::find_if(groupByColumns_.begin(), groupByColumns_.end(),
                                  StringDataTypeComp(colNameLeft)) != groupByColumns_.end() &&
                     !insideAggregation_;
    bool isRightKey = std::find_if(groupByColumns_.begin(), groupByColumns_.end(),
                                   StringDataTypeComp(colNameRight)) != groupByColumns_.end() &&
                      !insideAggregation_;

    if (isLeftKey || isRightKey)
    {
        if (isOverallLastBlock_)
        {
            auto columnLeft = FindStringColumn(colNameLeft + (isLeftKey ? KEYS_SUFFIX : ""));
            auto columnRight = FindStringColumn(colNameRight + (isRightKey ? KEYS_SUFFIX : ""));
            int32_t retSize = std::min(columnLeft.ElementCount, columnRight.ElementCount);

            GPUMemory::GPUString result;
            GPUStringBinary::ColCol<OP>(result, columnLeft.GpuPtr, columnRight.GpuPtr, retSize);
            if (columnLeft.GpuNullMaskPtr || columnRight.GpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = AllocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                FillStringRegister(result, reg + KEYS_SUFFIX, retSize, true, combinedMask);
                if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
                {
                    GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                        combinedMask, reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                        reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
                }
                else if (columnLeft.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                else if (columnRight.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
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
        auto columnLeft = FindStringColumn(colNameLeft);
        auto columnRight = FindStringColumn(colNameRight);
        int32_t retSize = std::min(columnLeft.ElementCount, columnRight.ElementCount);

        if (!IsRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUStringBinary::ColCol<OP>(result, columnLeft.GpuPtr, columnRight.GpuPtr, retSize);
            if (columnLeft.GpuNullMaskPtr || columnRight.GpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = AllocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
                FillStringRegister(result, reg, retSize, true, combinedMask);
                if (columnLeft.GpuNullMaskPtr && columnRight.GpuNullMaskPtr)
                {
                    GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                        combinedMask, reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                        reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr), bitMaskSize);
                }
                else if (columnLeft.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnLeft.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                else if (columnRight.GpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnRight.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
            }
            else
            {
                FillStringRegister(result, reg, retSize, true);
            }
        }
    }
    return InstructionStatus::CONTINUE;
}

template <typename OP>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::StringBinaryColConst()
{
    std::string cnst = arguments_.Read<std::string>();
    auto colName = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<std::string>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "StringBinaryColConst: " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
            groupByColumns_.end() &&
        !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            auto column = FindStringColumn(colName + KEYS_SUFFIX);
            int32_t retSize = column.ElementCount;
            GPUMemory::GPUString result;
            GPUMemory::GPUString cnstString = InsertConstStringGpu(cnst);
            GPUStringBinary::ColConst<OP>(result, column.GpuPtr, cnstString, retSize);
            if (column.GpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = AllocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                FillStringRegister(result, reg + KEYS_SUFFIX, retSize, false, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                FillStringRegister(result, reg + KEYS_SUFFIX, retSize, false);
            }
            groupByColumns_.push_back({reg, DataType::COLUMN_STRING});
        }
    }
    else if (isOverallLastBlock_ || !usingGroupBy_ || insideGroupBy_ || insideAggregation_)
    {
        auto column = FindStringColumn(colName);
        int32_t retSize = column.ElementCount;

        if (!IsRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUMemory::GPUString cnstString = InsertConstStringGpu(cnst);
            GPUStringBinary::ColConst<OP>(result, column.GpuPtr, cnstString, retSize);
            if (column.GpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = AllocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
                FillStringRegister(result, reg, retSize, false, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                FillStringRegister(result, reg, retSize, false);
            }
        }
    }
    return InstructionStatus::CONTINUE;
}

template <typename OP>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::StringBinaryConstCol()
{
    auto colName = arguments_.Read<std::string>();
    std::string cnst = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    GpuSqlDispatcher::InstructionStatus loadFlag = LoadCol<std::string>(colName);
    if (loadFlag != InstructionStatus::CONTINUE)
    {
        return loadFlag;
    }

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "StringBinaryConstCol: " << reg << '\n';

    if (std::find_if(groupByColumns_.begin(), groupByColumns_.end(), StringDataTypeComp(colName)) !=
            groupByColumns_.end() &&
        !insideAggregation_)
    {
        if (isOverallLastBlock_)
        {
            GPUMemory::GPUString gpuString = InsertConstStringGpu(cnst);
            auto column = FindStringColumn(colName + KEYS_SUFFIX);
            int32_t retSize = column.ElementCount;
            GPUMemory::GPUString result;
            GPUStringBinary::ConstCol<OP>(result, gpuString, column.GpuPtr, retSize);
            if (column.GpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = AllocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                FillStringRegister(result, reg + KEYS_SUFFIX, retSize, false, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                FillStringRegister(result, reg + KEYS_SUFFIX, retSize, false);
            }
            groupByColumns_.push_back({reg, DataType::COLUMN_STRING});
        }
    }
    else if (isOverallLastBlock_ || !usingGroupBy_ || insideGroupBy_ || insideAggregation_)
    {
        GPUMemory::GPUString gpuString = InsertConstStringGpu(cnst);
        auto column = FindStringColumn(colName);
        int32_t retSize = column.ElementCount;

        if (!IsRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUStringBinary::ConstCol<OP>(result, gpuString, column.GpuPtr, retSize);
            if (column.GpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = AllocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
                FillStringRegister(result, reg, retSize, false, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(column.GpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                FillStringRegister(result, reg, retSize, false);
            }
        }
    }
    return InstructionStatus::CONTINUE;
}

template <typename OP>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::StringBinaryConstConst()
{
    std::string cnstRight = arguments_.Read<std::string>();
    std::string cnstLeft = arguments_.Read<std::string>();
    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "StringBinaryConstConst: " << reg << '\n';
    int32_t retSize = GetBlockSize();
    if (retSize == 0)
    {
        return InstructionStatus::CONTINUE;
    }
    GPUMemory::GPUString gpuStringLeft = InsertConstStringGpu(cnstLeft, retSize);
    GPUMemory::GPUString gpuStringRight = InsertConstStringGpu(cnstRight, retSize);

    if (!IsRegisterAllocated(reg))
    {
        GPUMemory::GPUString result;
        GPUStringBinary::ConstConst<OP>(result, gpuStringLeft, gpuStringRight, retSize);
        FillStringRegister(result, reg, retSize, false);
    }
    return InstructionStatus::CONTINUE;
}
