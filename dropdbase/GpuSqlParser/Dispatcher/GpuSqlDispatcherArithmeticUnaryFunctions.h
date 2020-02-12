#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmeticUnary.cuh"

template <typename OP, typename T>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::ArithmeticUnary()
{
    InstructionArgument<T> left = DispatcherInstructionHelper<T>::LoadInstructionArgument(*this);

    if (std::get<2>(left) != InstructionStatus::CONTINUE)
    {
        return std::get<2>(left);
    }

    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "ArithmeticUnary: " << reg << '\n';

    // TODO STD conditional :: if OP == abs return type = T

    typedef typename std::conditional<std::is_same<typename OP::RetType, void>::value,
                                      typename std::remove_pointer<T>::type, typename OP::RetType>::type ResultType;

    if constexpr (std::is_pointer<T>::value)
    {
        if (std::get<0>(left))
        {
            const int32_t retSize = std::get<1>(left).ElementCount;
            const bool allocateNullMask = std::get<1>(left).GpuNullMaskPtr;
            InstructionResult<ResultType> result =
                DispatcherInstructionHelper<ResultType>::AllocateInstructionResult(*this, reg, retSize, allocateNullMask,
                                                                                   {std::get<3>(left)});
            if (isCompositeDataType<ResultType> || std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                  reinterpret_cast<int8_t*>(std::get<1>(left).GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPUArithmeticUnary<OP, ResultType, T>::ArithmeticUnary(std::get<0>(result),
                                                                       std::get<0>(left), retSize);
                DispatcherInstructionHelper<ResultType>::StoreInstructionResult(result, *this, reg,
                                                                                retSize, allocateNullMask,
                                                                                {std::get<3>(left)});
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

        InstructionResult<ResultType> result =
            DispatcherInstructionHelper<ResultType>::AllocateInstructionResult(*this, reg, retSize, false, {});

        if (isCompositeDataType<ResultType> || std::get<0>(result))
        {
            GPUArithmeticUnary<OP, ResultType, T>::ArithmeticUnary(std::get<0>(result),
                                                                   std::get<0>(left), retSize);
            DispatcherInstructionHelper<ResultType>::StoreInstructionResult(result, *this, reg, retSize,
                                                                            false, {std::get<3>(left)});
        }
    }

    return InstructionStatus::CONTINUE;
}