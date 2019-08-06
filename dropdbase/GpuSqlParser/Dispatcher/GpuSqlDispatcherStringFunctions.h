#pragma once

#include "../ParserExceptions.h"
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"

#include "../../QueryEngine/GPUCore/GPUStringUnary.cuh"
#include "../../QueryEngine/GPUCore/GPUStringBinary.cuh"

template <typename OP>
int32_t GpuSqlDispatcher::StringUnaryCol()
{
    auto colName = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int32_t loadFlag = loadCol<std::string>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "StringUnaryCol: " << colName << " " << reg << std::endl;

    if (std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) !=
        groupByColumns.end())
    {
        if (isOverallLastBlock)
        {
            auto column = findStringColumn(colName + KEYS_SUFFIX);
            int32_t retSize = std::get<1>(column);
            GPUMemory::GPUString result;
            GPUStringUnary::Col<OP>(result, std::get<0>(column), retSize);
            if (std::get<2>(column))
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, true, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(std::get<2>(column)), bitMaskSize);
            }
            else
            {
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, true);
            }
            groupByColumns.push_back({reg, DataType::COLUMN_STRING});
        }
    }
    else if (isOverallLastBlock || !usingGroupBy || insideGroupBy || insideAggregation)
    {
        auto column = findStringColumn(colName);
        int32_t retSize = std::get<1>(column);

        if (!isRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUStringUnary::Col<OP>(result, std::get<0>(column), retSize);
            if (std::get<2>(column))
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg, retSize, true, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(std::get<2>(column)), bitMaskSize);
            }
            else
            {
                fillStringRegister(result, reg, retSize, true);
            }
        }
    }
    return 0;
}

template <typename OP>
int32_t GpuSqlDispatcher::StringUnaryConst()
{
    std::string cnst = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    std::cout << "StringUnaryConst: " << reg << std::endl;

    GPUMemory::GPUString gpuString = insertConstStringGpu(cnst);

    if (!isRegisterAllocated(reg))
    {
        GPUMemory::GPUString result;
        GPUStringUnary::Const<OP>(result, gpuString);
        fillStringRegister(result, reg, 1, true);
    }
    return 0;
}

template <typename OP>
int32_t GpuSqlDispatcher::StringUnaryNumericCol()
{
    auto colName = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int32_t loadFlag = loadCol<std::string>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "StringIntUnaryCol: " << colName << " " << reg << std::endl;

    if (std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) !=
        groupByColumns.end())
    {
        if (isOverallLastBlock)
        {
            auto column = findStringColumn(colName + KEYS_SUFFIX);
            int32_t retSize = std::get<1>(column);
            int32_t* result = allocateRegister<int32_t>(reg + KEYS_SUFFIX, retSize);
            GPUStringUnary::Col<OP>(result, std::get<0>(column), retSize);
            groupByColumns.push_back({reg, DataType::COLUMN_INT});
        }
    }
    else if (isOverallLastBlock || !usingGroupBy || insideGroupBy || insideAggregation)
    {
        auto column = findStringColumn(colName);
        int32_t retSize = std::get<1>(column);

        if (!isRegisterAllocated(reg))
        {
            int32_t* result;
            if (std::get<2>(column))
            {
                int8_t* nullMask;
                result = allocateRegister<int32_t>(reg, retSize, &nullMask);
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(std::get<2>(column)), bitMaskSize);
            }
            else
            {
                result = allocateRegister<int32_t>(reg, retSize);
            }
            GPUStringUnary::Col<OP>(result, std::get<0>(column), retSize);
        }
    }
    return 0;
}

template <typename OP>
int32_t GpuSqlDispatcher::StringUnaryNumericConst()
{
    std::string cnst = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    std::cout << "StringUnaryConst: " << reg << std::endl;

    GPUMemory::GPUString gpuString = insertConstStringGpu(cnst);

    if (!isRegisterAllocated(reg))
    {
        int32_t* result = allocateRegister<int32_t>(reg, 1);
        GPUStringUnary::Const<OP>(result, gpuString);
    }
    return 0;
}


template <typename OP, typename T>
int32_t GpuSqlDispatcher::StringBinaryNumericColCol()
{
    auto colNameRight = arguments.read<std::string>();
    auto colNameLeft = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int32_t loadFlag = loadCol<std::string>(colNameLeft);
    if (loadFlag)
    {
        return loadFlag;
    }

    loadFlag = loadCol<T>(colNameRight);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "StringBinaryNumericColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

    bool isLeftKey = std::find_if(groupByColumns.begin(), groupByColumns.end(),
                                  StringDataTypeComp(colNameLeft)) != groupByColumns.end();
    bool isRightKey = std::find_if(groupByColumns.begin(), groupByColumns.end(),
                                   StringDataTypeComp(colNameRight)) != groupByColumns.end();

    if (isLeftKey || isRightKey)
    {
        if (isOverallLastBlock)
        {
            auto columnLeft = findStringColumn(colNameLeft + (isLeftKey ? KEYS_SUFFIX : ""));
            auto columnRight = allocatedPointers.at(colNameRight + (isRightKey ? KEYS_SUFFIX : ""));
            int32_t retSize = std::min(std::get<1>(columnLeft), columnRight.elementCount);

            GPUMemory::GPUString result;
            GPUStringBinary::ColCol<OP>(result, std::get<0>(columnLeft),
                                        reinterpret_cast<T*>(columnRight.gpuPtr), retSize);
            if (std::get<2>(columnLeft) || columnRight.gpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, true, combinedMask);
                if (std::get<2>(columnLeft) && columnRight.gpuNullMaskPtr)
                {
                    GPUArithmetic::colCol<ArithmeticOperations::bitwiseOr>(
                        combinedMask, reinterpret_cast<int8_t*>(std::get<2>(columnLeft)),
                        reinterpret_cast<int8_t*>(columnRight.gpuNullMaskPtr), bitMaskSize);
                }
                else if (std::get<2>(columnLeft))
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(std::get<2>(columnLeft)),
                                                  bitMaskSize);
                }
                else if (columnRight.gpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnRight.gpuNullMaskPtr),
                                                  bitMaskSize);
                }
            }
            else
            {
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, true);
            }
            groupByColumns.push_back({reg, DataType::COLUMN_STRING});
        }
    }
    else if (isOverallLastBlock || !usingGroupBy || insideGroupBy || insideAggregation)
    {
        auto columnLeft = findStringColumn(colNameLeft);
        auto columnRight = allocatedPointers.at(colNameRight);
        int32_t retSize = std::min(std::get<1>(columnLeft), columnRight.elementCount);

        if (!isRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUStringBinary::ColCol<OP>(result, std::get<0>(columnLeft),
                                        reinterpret_cast<T*>(columnRight.gpuPtr), retSize);
            if (std::get<2>(columnLeft) || columnRight.gpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg, retSize, true, combinedMask);
                if (std::get<2>(columnLeft) && columnRight.gpuNullMaskPtr)
                {
                    GPUArithmetic::colCol<ArithmeticOperations::bitwiseOr>(
                        combinedMask, reinterpret_cast<int8_t*>(std::get<2>(columnLeft)),
                        reinterpret_cast<int8_t*>(columnRight.gpuNullMaskPtr), bitMaskSize);
                }
                else if (std::get<2>(columnLeft))
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(std::get<2>(columnLeft)),
                                                  bitMaskSize);
                }
                else if (columnRight.gpuNullMaskPtr)
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(columnRight.gpuNullMaskPtr),
                                                  bitMaskSize);
                }
            }
            else
            {
                fillStringRegister(result, reg, retSize, true);
            }
        }
    }
    return 0;
}

template <typename OP, typename T>
int32_t GpuSqlDispatcher::StringBinaryNumericColConst()
{
    T cnst = arguments.read<T>();
    auto colName = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int32_t loadFlag = loadCol<std::string>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "StringBinaryColConst: " << reg << std::endl;

    if (std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) !=
        groupByColumns.end())
    {
        if (isOverallLastBlock)
        {
            auto column = findStringColumn(colName + KEYS_SUFFIX);
            int32_t retSize = std::get<1>(column);
            GPUMemory::GPUString result;
            GPUStringBinary::ColConst<OP>(result, std::get<0>(column), cnst, retSize);
            if (std::get<2>(column))
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, true, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(std::get<2>(column)), bitMaskSize);
            }
            else
            {
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, true);
            }
            groupByColumns.push_back({reg, DataType::COLUMN_STRING});
        }
    }
    else if (isOverallLastBlock || !usingGroupBy || insideGroupBy || insideAggregation)
    {
        auto column = findStringColumn(colName);
        int32_t retSize = std::get<1>(column);

        if (!isRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUStringBinary::ColConst<OP>(result, std::get<0>(column), cnst, retSize);
            if (std::get<2>(column))
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg, retSize, true, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(std::get<2>(column)), bitMaskSize);
            }
            else
            {
                fillStringRegister(result, reg, retSize, true);
            }
        }
    }
    return 0;
}

template <typename OP, typename T>
int32_t GpuSqlDispatcher::StringBinaryNumericConstCol()
{
    auto colName = arguments.read<std::string>();
    std::string cnst = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int32_t loadFlag = loadCol<std::string>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "StringBinaryConstCol: " << reg << std::endl;

    if (std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) !=
        groupByColumns.end())
    {
        if (isOverallLastBlock)
        {
            GPUMemory::GPUString gpuString = insertConstStringGpu(cnst);
            PointerAllocation column = allocatedPointers.at(colName + KEYS_SUFFIX);
            int32_t retSize = column.elementCount;
            GPUMemory::GPUString result;
            GPUStringBinary::ConstCol<OP>(result, gpuString, reinterpret_cast<T*>(column.gpuPtr), retSize);
            if (column.gpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, true, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, true);
            }
            groupByColumns.push_back({reg, DataType::COLUMN_STRING});
        }
    }
    else if (isOverallLastBlock || !usingGroupBy || insideGroupBy || insideAggregation)
    {
        GPUMemory::GPUString gpuString = insertConstStringGpu(cnst);
        PointerAllocation column = allocatedPointers.at(colName);
        int32_t retSize = column.elementCount;

        if (!isRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUStringBinary::ConstCol<OP>(result, gpuString, reinterpret_cast<T*>(column.gpuPtr), retSize);
            if (column.gpuNullMaskPtr)
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg, retSize, true, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
            }
            else
            {
                fillStringRegister(result, reg, retSize, true);
            }
        }
    }
    return 0;
}

template <typename OP, typename T>
int32_t GpuSqlDispatcher::StringBinaryNumericConstConst()
{
    T cnstRight = arguments.read<T>();
    std::string cnstLeft = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    std::cout << "StringBinaryConstConst: " << reg << std::endl;

    GPUMemory::GPUString gpuString = insertConstStringGpu(cnstLeft);

    if (!isRegisterAllocated(reg))
    {
        GPUMemory::GPUString result;
        GPUStringBinary::ConstConst<OP>(result, gpuString, cnstRight, 1);
        fillStringRegister(result, reg, 1, true);
    }
    return 0;
}

template <typename OP>
int32_t GpuSqlDispatcher::StringBinaryColCol()
{
    auto colNameRight = arguments.read<std::string>();
    auto colNameLeft = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int32_t loadFlag = loadCol<std::string>(colNameLeft);
    if (loadFlag)
    {
        return loadFlag;
    }

    loadFlag = loadCol<std::string>(colNameRight);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "StringBinaryColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

    bool isLeftKey = std::find_if(groupByColumns.begin(), groupByColumns.end(),
                                  StringDataTypeComp(colNameLeft)) != groupByColumns.end();
    bool isRightKey = std::find_if(groupByColumns.begin(), groupByColumns.end(),
                                   StringDataTypeComp(colNameRight)) != groupByColumns.end();

    if (isLeftKey || isRightKey)
    {
        if (isOverallLastBlock)
        {
            auto columnLeft = findStringColumn(colNameLeft + (isLeftKey ? KEYS_SUFFIX : ""));
            auto columnRight = findStringColumn(colNameRight + (isRightKey ? KEYS_SUFFIX : ""));
            int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

            GPUMemory::GPUString result;
            GPUStringBinary::ColCol<OP>(result, std::get<0>(columnLeft), std::get<0>(columnRight), retSize);
            if (std::get<2>(columnLeft) || std::get<2>(columnRight))
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, true, combinedMask);
                if (std::get<2>(columnLeft) && std::get<2>(columnRight))
                {
                    GPUArithmetic::colCol<ArithmeticOperations::bitwiseOr>(
                        combinedMask, reinterpret_cast<int8_t*>(std::get<2>(columnLeft)),
                        reinterpret_cast<int8_t*>(std::get<2>(columnRight)), bitMaskSize);
                }
                else if (std::get<2>(columnLeft))
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(std::get<2>(columnLeft)),
                                                  bitMaskSize);
                }
                else if (std::get<2>(columnRight))
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(std::get<2>(columnRight)),
                                                  bitMaskSize);
                }
            }
            else
            {
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, true);
            }
            groupByColumns.push_back({reg, DataType::COLUMN_STRING});
        }
    }
    else if (isOverallLastBlock || !usingGroupBy || insideGroupBy || insideAggregation)
    {
        auto columnLeft = findStringColumn(colNameLeft);
        auto columnRight = findStringColumn(colNameRight);
        int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

        if (!isRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUStringBinary::ColCol<OP>(result, std::get<0>(columnLeft), std::get<0>(columnRight), retSize);
            if (std::get<2>(columnLeft) || std::get<2>(columnRight))
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg, retSize, true, combinedMask);
                if (std::get<2>(columnLeft) && std::get<2>(columnRight))
                {
                    GPUArithmetic::colCol<ArithmeticOperations::bitwiseOr>(
                        combinedMask, reinterpret_cast<int8_t*>(std::get<2>(columnLeft)),
                        reinterpret_cast<int8_t*>(std::get<2>(columnRight)), bitMaskSize);
                }
                else if (std::get<2>(columnLeft))
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(std::get<2>(columnLeft)),
                                                  bitMaskSize);
                }
                else if (std::get<2>(columnRight))
                {
                    GPUMemory::copyDeviceToDevice(combinedMask,
                                                  reinterpret_cast<int8_t*>(std::get<2>(columnRight)),
                                                  bitMaskSize);
                }
            }
            else
            {
                fillStringRegister(result, reg, retSize, true);
            }
        }
    }
    return 0;
}

template <typename OP>
int32_t GpuSqlDispatcher::StringBinaryColConst()
{
    std::string cnst = arguments.read<std::string>();
    auto colName = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int32_t loadFlag = loadCol<std::string>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "StringBinaryColConst: " << reg << std::endl;

    if (std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) !=
        groupByColumns.end())
    {
        if (isOverallLastBlock)
        {
            auto column = findStringColumn(colName + KEYS_SUFFIX);
            int32_t retSize = std::get<1>(column);
            GPUMemory::GPUString result;
            GPUMemory::GPUString cnstString = insertConstStringGpu(cnst);
            GPUStringBinary::ColConst<OP>(result, std::get<0>(column), cnstString, retSize);
            if (std::get<2>(column))
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, false, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(std::get<2>(column)), bitMaskSize);
            }
            else
            {
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, false);
            }
            groupByColumns.push_back({reg, DataType::COLUMN_STRING});
        }
    }
    else if (isOverallLastBlock || !usingGroupBy || insideGroupBy || insideAggregation)
    {
        auto column = findStringColumn(colName);
        int32_t retSize = std::get<1>(column);

        if (!isRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUMemory::GPUString cnstString = insertConstStringGpu(cnst);
            GPUStringBinary::ColConst<OP>(result, std::get<0>(column), cnstString, retSize);
            if (std::get<2>(column))
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg, retSize, false, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(std::get<2>(column)), bitMaskSize);
            }
            else
            {
                fillStringRegister(result, reg, retSize, false);
            }
        }
    }
    return 0;
}

template <typename OP>
int32_t GpuSqlDispatcher::StringBinaryConstCol()
{
    auto colName = arguments.read<std::string>();
    std::string cnst = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    int32_t loadFlag = loadCol<std::string>(colName);
    if (loadFlag)
    {
        return loadFlag;
    }

    std::cout << "StringBinaryConstCol: " << reg << std::endl;

    if (std::find_if(groupByColumns.begin(), groupByColumns.end(), StringDataTypeComp(colName)) !=
        groupByColumns.end())
    {
        if (isOverallLastBlock)
        {
            GPUMemory::GPUString gpuString = insertConstStringGpu(cnst);
            auto column = findStringColumn(colName + KEYS_SUFFIX);
            int32_t retSize = std::get<1>(column);
            GPUMemory::GPUString result;
            GPUStringBinary::ConstCol<OP>(result, gpuString, std::get<0>(column), retSize);
            if (std::get<2>(column))
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + KEYS_SUFFIX + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, false, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(std::get<2>(column)), bitMaskSize);
            }
            else
            {
                fillStringRegister(result, reg + KEYS_SUFFIX, retSize, false);
            }
            groupByColumns.push_back({reg, DataType::COLUMN_STRING});
        }
    }
    else if (isOverallLastBlock || !usingGroupBy || insideGroupBy || insideAggregation)
    {
        GPUMemory::GPUString gpuString = insertConstStringGpu(cnst);
        auto column = findStringColumn(colName);
        int32_t retSize = std::get<1>(column);

        if (!isRegisterAllocated(reg))
        {
            GPUMemory::GPUString result;
            GPUStringBinary::ConstCol<OP>(result, gpuString, std::get<0>(column), retSize);
            if (std::get<2>(column))
            {
                int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                int8_t* combinedMask = allocateRegister<int8_t>(reg + NULL_SUFFIX, bitMaskSize);
                fillStringRegister(result, reg, retSize, false, combinedMask);
                GPUMemory::copyDeviceToDevice(combinedMask,
                                              reinterpret_cast<int8_t*>(std::get<2>(column)), bitMaskSize);
            }
            else
            {
                fillStringRegister(result, reg, retSize, false);
            }
        }
    }
    return 0;
}

template <typename OP>
int32_t GpuSqlDispatcher::StringBinaryConstConst()
{
    std::string cnstRight = arguments.read<std::string>();
    std::string cnstLeft = arguments.read<std::string>();
    auto reg = arguments.read<std::string>();

    std::cout << "StringBinaryConstConst: " << reg << std::endl;

    GPUMemory::GPUString gpuStringLeft = insertConstStringGpu(cnstLeft);
    GPUMemory::GPUString gpuStringRight = insertConstStringGpu(cnstRight);

    if (!isRegisterAllocated(reg))
    {
        GPUMemory::GPUString result;
        GPUStringBinary::ConstConst<OP>(result, gpuStringLeft, gpuStringRight, 1);
        fillStringRegister(result, reg, 1, false);
    }
    return 0;
}
