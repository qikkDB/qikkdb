#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"

/// Implementation of generic binary arithmetic function dispatching given by the functor OP
/// Implementation for column constant case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename L, typename R>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::Arithmetic()
{
    std::tuple<R, PointerAllocation, InstructionStatus, std::string> right = LoadInstructionArgument<R>();
    std::tuple<L, PointerAllocation, InstructionStatus, std::string> left = LoadInstructionArgument<L>();

    if (std::get<2>(left) != InstructionStatus::CONTINUE)
    {
        return std::get<2>(left);
    }

    if (std::get<2>(right) != InstructionStatus::CONTINUE)
    {
        return std::get<2>(right);
    }

    auto reg = arguments_.Read<std::string>();
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Arithmetic: " << reg << '\n';

    constexpr bool bothTypesFloatOrBothIntegral =
        std::is_floating_point<typename std::remove_pointer<L>::type>::value &&
            std::is_floating_point<typename std::remove_pointer<R>::type>::value ||
        std::is_integral<typename std::remove_pointer<L>::type>::value &&
            std::is_integral<typename std::remove_pointer<R>::type>::value;
    typedef typename std::conditional<
        bothTypesFloatOrBothIntegral,
        typename std::conditional<sizeof(typename std::remove_pointer<L>::type) >= sizeof(typename std::remove_pointer<R>::type),
                                  typename std::remove_pointer<L>::type, typename std::remove_pointer<R>::type>::type,
        typename std::conditional<std::is_floating_point<typename std::remove_pointer<L>::type>::value, typename std::remove_pointer<L>::type,
                                  typename std::conditional<std::is_floating_point<typename std::remove_pointer<R>::type>::value,
                                                            typename std::remove_pointer<R>::type, void>::type>::type>::type ResultType;

    if constexpr (std::is_pointer<L>::value && std::is_pointer<R>::value)
    {
        if (std::get<0>(left) && std::get<0>(right))
        {
            const int32_t retSize = std::min(std::get<1>(left).ElementCount, std::get<1>(right).ElementCount);
            const bool allocateNullMask = std::get<1>(left).GpuNullMaskPtr || std::get<1>(right).GpuNullMaskPtr;
            std::pair<ResultType*, int8_t*> result =
                AllocateInstructionResult<ResultType>(reg, retSize, allocateNullMask,
                                                      {std::get<3>(left), std::get<3>(right)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    if (std::get<1>(left).GpuNullMaskPtr && std::get<1>(right).GpuNullMaskPtr)
                    {
                        GPUArithmetic::Arithmetic<ArithmeticOperations::bitwiseOr>(
                            std::get<1>(result), reinterpret_cast<int8_t*>(std::get<1>(left).GpuNullMaskPtr),
                            reinterpret_cast<int8_t*>(std::get<1>(right).GpuNullMaskPtr), bitMaskSize);
                    }
                    else if (std::get<1>(left).GpuNullMaskPtr)
                    {
                        GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                      reinterpret_cast<int8_t*>(std::get<1>(left).GpuNullMaskPtr),
                                                      bitMaskSize);
                    }
                    else
                    {
                        GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                      reinterpret_cast<int8_t*>(std::get<1>(right).GpuNullMaskPtr),
                                                      bitMaskSize);
                    }
                }
                GPUArithmetic::Arithmetic<OP, ResultType, L, R>(std::get<0>(result), std::get<0>(left),
                                                                std::get<0>(right), retSize);
            }
        }
        FreeColumnIfRegister<L>(std::get<3>(left));
        FreeColumnIfRegister<R>(std::get<3>(right));
    }

    else if constexpr (std::is_pointer<L>::value)
    {
        if (std::get<0>(left))
        {
            const int32_t retSize = std::get<1>(left).ElementCount;
            const bool allocateNullMask = std::get<1>(left).GpuNullMaskPtr;
            std::pair<ResultType*, int8_t*> result =
                AllocateInstructionResult<ResultType>(reg, retSize, allocateNullMask,
                                                      {std::get<3>(left), std::get<3>(right)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                  reinterpret_cast<int8_t*>(std::get<1>(left).GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPUArithmetic::Arithmetic<OP, ResultType, L, R>(std::get<0>(result), std::get<0>(left),
                                                                std::get<0>(right), retSize);
            }
        }
        FreeColumnIfRegister<L>(std::get<3>(left));
    }

    else if constexpr (std::is_pointer<R>::value)
    {
        if (std::get<0>(right))
        {
            const int32_t retSize = std::get<1>(right).ElementCount;
            const bool allocateNullMask = std::get<1>(right).GpuNullMaskPtr;
            std::pair<ResultType*, int8_t*> result =
                AllocateInstructionResult<ResultType>(reg, retSize, allocateNullMask,
                                                      {std::get<3>(left), std::get<3>(right)});
            if (std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                  reinterpret_cast<int8_t*>(std::get<1>(right).GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPUArithmetic::Arithmetic<OP, ResultType, L, R>(std::get<0>(result), std::get<0>(left),
                                                                std::get<0>(right), retSize);
            }
        }
        FreeColumnIfRegister<R>(std::get<3>(right));
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
            GPUArithmetic::Arithmetic<OP, ResultType, L, R>(std::get<0>(result), std::get<0>(left),
                                                            std::get<0>(right), retSize);
        }
    }

    return InstructionStatus::CONTINUE;
}