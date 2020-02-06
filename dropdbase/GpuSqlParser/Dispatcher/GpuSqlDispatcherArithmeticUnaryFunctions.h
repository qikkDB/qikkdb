#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmeticUnary.cuh"

template <typename OP, typename T>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ArithmeticUnary()
{
    std::tuple<T, PointerAllocation, InstructionStatus, std::string> left = LoadInstructionArgument<T>();

    if (std::get<2>(left) != InstructionStatus::CONTINUE)
    {
        return std::get<2>(left);
    }

    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "ArithmeticUnary: " << reg << '\n';

    // TODO STD conditional :: if OP == abs return type = T

    typedef typename std::conditional<OP::isFloatRetType, float, typename std::remove_pointer<T>::type>::type ResultType;

    if (std::is_pointer<T>::value)
    {
        if (std::get<0>(left))
        {
            const int32_t retSize = std::get<1>(left).ElementCount;
            const bool allocateNullMask = std::get<1>(left).GpuNullMaskPtr;
            std::pair<ResultType*, int8_t*> result =
                AllocateInstructionResult<ResultType>(reg, retSize, allocateNullMask, {std::get<3>(left)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                  reinterpret_cast<int8_t*>(std::get<1>(left).GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPUArithmeticUnary::ArithmeticUnary<OP, ResultType, T>(std::get<0>(result),
                                                                       std::get<0>(left), retSize);
            }
        }
        FreeColumnIfRegister<T>(std::get<3>(left));
    }
    else
    {
        const int32_t retSize = GetBlockSize();
        if (retSize == 0)
        {
            return InstructionStatus::OUT_OF_BLOCKS;
        }

        std::pair<ResultType*, int8_t*> result =
            AllocateInstructionResult<ResultType>(reg, retSize, false, {});

        if (std::get<0>(result))
        {
            GPUArithmeticUnary::ArithmeticUnary<OP, ResultType, T>(std::get<0>(result),
                                                                   std::get<0>(left), retSize);
        }
    }

    return InstructionStatus::CONTINUE;
}