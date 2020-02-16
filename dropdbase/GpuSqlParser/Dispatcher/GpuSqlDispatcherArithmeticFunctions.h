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
    InstructionArgument<R> right = DispatcherInstructionHelper<R>::LoadInstructionArgument(*this);
    InstructionArgument<L> left = DispatcherInstructionHelper<L>::LoadInstructionArgument(*this);

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
        std::is_same<typename OP::RetType, void>::value,
        typename std::conditional<
            bothTypesFloatOrBothIntegral,
            typename std::conditional<sizeof(typename std::remove_pointer<L>::type) >= sizeof(typename std::remove_pointer<R>::type),
                                      typename std::remove_pointer<L>::type, typename std::remove_pointer<R>::type>::type,
            typename std::conditional<std::is_floating_point<typename std::remove_pointer<L>::type>::value, typename std::remove_pointer<L>::type,
                                      typename std::conditional<std::is_floating_point<typename std::remove_pointer<R>::type>::value,
                                                                typename std::remove_pointer<R>::type, void>::type>::type>::type,
        typename OP::RetType>::type ResultType;

    if constexpr (std::is_pointer<L>::value && std::is_pointer<R>::value)
    {
        if (std::get<0>(left) && std::get<0>(right))
        {
            const int32_t retSize = std::min(std::get<1>(left).ElementCount, std::get<1>(right).ElementCount);
            const bool allocateNullMask = std::get<1>(left).GpuNullMaskPtr || std::get<1>(right).GpuNullMaskPtr;
            InstructionResult<ResultType> result = DispatcherInstructionHelper<ResultType>::AllocateInstructionResult(
                *this, reg, retSize, allocateNullMask, {std::get<3>(left), std::get<3>(right)});
            if (isCompositeDataType<ResultType> || std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    if (std::get<1>(left).GpuNullMaskPtr && std::get<1>(right).GpuNullMaskPtr)
                    {
                        GPUArithmetic<ArithmeticOperations::bitwiseOr, int8_t, int8_t*, int8_t*>::Arithmetic(
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
                GPUArithmetic<OP, ResultType, L, R>::Arithmetic(std::get<0>(result), std::get<0>(left),
                                                                std::get<0>(right), retSize);
                DispatcherInstructionHelper<ResultType>::StoreInstructionResult(
                    result, *this, reg, retSize, allocateNullMask, {std::get<3>(left), std::get<3>(right)});
            }
        }
        FreeColumnIfRegister<L>(std::get<3>(left));
        FreeColumnIfRegister<R>(std::get<3>(right));
    }

    else if constexpr (std::is_pointer<L>::value || std::is_pointer<R>::value)
    {
        typedef typename std::conditional<std::is_pointer<L>::value, L, R>::type ColType;
        InstructionArgument<ColType> col;
        if constexpr (std::is_pointer<L>::value)
        {
            col = left;
        }
        else
        {
            col = right;
        }

        if (std::get<0>(col))
        {
            const int32_t retSize = std::get<1>(col).ElementCount;
            const bool allocateNullMask = std::get<1>(col).GpuNullMaskPtr;
            InstructionResult<ResultType> result = DispatcherInstructionHelper<ResultType>::AllocateInstructionResult(
                *this, reg, retSize, allocateNullMask, {std::get<3>(left), std::get<3>(right)});
            if (isCompositeDataType<ResultType> || std::get<0>(result))
            {
                if (std::get<1>(result))
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    GPUMemory::copyDeviceToDevice(std::get<1>(result),
                                                  reinterpret_cast<int8_t*>(std::get<1>(col).GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPUArithmetic<OP, ResultType, L, R>::Arithmetic(std::get<0>(result), std::get<0>(left),
                                                                std::get<0>(right), retSize);
                DispatcherInstructionHelper<ResultType>::StoreInstructionResult(
                    result, *this, reg, retSize, allocateNullMask, {std::get<3>(left), std::get<3>(right)});
            }
        }
        FreeColumnIfRegister<ColType>(std::get<3>(col));
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
            GPUArithmetic<OP, ResultType, L, R>::Arithmetic(std::get<0>(result), std::get<0>(left),
                                                            std::get<0>(right), retSize);
            DispatcherInstructionHelper<ResultType>::StoreInstructionResult(result, *this, reg,
                                                                            retSize, false, {});
        }
    }

    return InstructionStatus::CONTINUE;
}