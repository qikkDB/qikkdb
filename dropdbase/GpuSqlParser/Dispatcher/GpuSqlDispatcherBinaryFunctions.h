#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUBinary.cuh"

/// Implementation of generic binary arithmetic function dispatching given by the functor OP
/// Implementation for column constant case
/// Pops data from argument memory stream and loads data to GPU on demand
/// <returns name="statusCode">Finish status code of the operation</returns>
template <typename OP, typename L, typename R>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::Binary()
{
    InstructionArgument<R> right = DispatcherInstructionHelper<R>::LoadInstructionArgument(*this);
    InstructionArgument<L> left = DispatcherInstructionHelper<L>::LoadInstructionArgument(*this);

    if (left.LoadStatus != InstructionStatus::CONTINUE)
    {
        return left.LoadStatus;
    }

    if (right.LoadStatus != InstructionStatus::CONTINUE)
    {
        return right.LoadStatus;
    }

    auto reg = arguments_.Read<std::string>();
    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Binary: " << reg << '\n';

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
        if (left.Data && right.Data)
        {
            const int32_t retSize =
                std::min(left.DataAllocation.ElementCount, right.DataAllocation.ElementCount);
            const bool allocateNullMask = left.DataAllocation.GpuNullMaskPtr || right.DataAllocation.GpuNullMaskPtr;
            InstructionResult<ResultType> result =
                DispatcherInstructionHelper<ResultType>::AllocateInstructionResult(*this, reg, retSize, allocateNullMask,
                                                                                   {left.Name, right.Name});
            if (isCompositeDataType<ResultType> || result.Data)
            {
                if (result.NullMaskPtr)
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    if (left.DataAllocation.GpuNullMaskPtr && right.DataAllocation.GpuNullMaskPtr)
                    {
                        GPUBinary<ArithmeticOperations::bitwiseOr, int8_t, int8_t*, int8_t*>::Binary(
                            result.NullMaskPtr, reinterpret_cast<int8_t*>(left.DataAllocation.GpuNullMaskPtr),
                            reinterpret_cast<int8_t*>(right.DataAllocation.GpuNullMaskPtr), bitMaskSize);
                    }
                    else if (left.DataAllocation.GpuNullMaskPtr)
                    {
                        GPUMemory::copyDeviceToDevice(result.NullMaskPtr,
                                                      reinterpret_cast<int8_t*>(left.DataAllocation.GpuNullMaskPtr),
                                                      bitMaskSize);
                    }
                    else if (right.DataAllocation.GpuNullMaskPtr)
                    {
                        GPUMemory::copyDeviceToDevice(result.NullMaskPtr,
                                                      reinterpret_cast<int8_t*>(right.DataAllocation.GpuNullMaskPtr),
                                                      bitMaskSize);
                    }
                }
                GPUBinary<OP, ResultType, L, R>::Binary(result.Data, left.Data, right.Data, retSize,
                                                        result.NullMaskPtr);
                DispatcherInstructionHelper<ResultType>::StoreInstructionResult(result, *this, reg,
                                                                                retSize, allocateNullMask,
                                                                                {left.Name, right.Name});
            }
        }
        FreeColumnIfRegister<L>(left.Name);
        FreeColumnIfRegister<R>(right.Name);
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

        if (col.Data)
        {
            const int32_t retSize = col.DataAllocation.ElementCount;
            const bool allocateNullMask = col.DataAllocation.GpuNullMaskPtr;
            InstructionResult<ResultType> result =
                DispatcherInstructionHelper<ResultType>::AllocateInstructionResult(*this, reg, retSize, allocateNullMask,
                                                                                   {left.Name, right.Name});
            if (isCompositeDataType<ResultType> || result.Data)
            {
                if (result.NullMaskPtr)
                {
                    const int32_t bitMaskSize = ((retSize + sizeof(int8_t) * 8 - 1) / (8 * sizeof(int8_t)));
                    GPUMemory::copyDeviceToDevice(result.NullMaskPtr,
                                                  reinterpret_cast<int8_t*>(col.DataAllocation.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPUBinary<OP, ResultType, L, R>::Binary(result.Data, left.Data, right.Data, retSize,
                                                        result.NullMaskPtr);
                DispatcherInstructionHelper<ResultType>::StoreInstructionResult(result, *this, reg,
                                                                                retSize, allocateNullMask,
                                                                                {left.Name, right.Name});
            }
        }
        FreeColumnIfRegister<ColType>(col.Name);
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
        if (isCompositeDataType<ResultType> || result.Data)
        {
            GPUBinary<OP, ResultType, L, R>::Binary(result.Data, left.Data, right.Data, retSize,
                                                    result.NullMaskPtr);
            DispatcherInstructionHelper<ResultType>::StoreInstructionResult(result, *this, reg,
                                                                            retSize, false, {});
        }
    }

    return InstructionStatus::CONTINUE;
}