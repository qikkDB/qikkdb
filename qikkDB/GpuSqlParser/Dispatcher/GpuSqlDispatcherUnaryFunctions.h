#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUUnary.cuh"

template <typename OP, typename T>
GpuSqlDispatcher::InstructionStatus GpuSqlDispatcher::Unary()
{
    InstructionArgument<T> left = DispatcherInstructionHelper<T>::LoadInstructionArgument(*this);

    if (left.LoadStatus != InstructionStatus::CONTINUE)
    {
        return left.LoadStatus;
    }

    auto reg = arguments_.Read<std::string>();

    CudaLogBoost::getInstance(CudaLogBoost::debug) << "Unary: " << reg << '\n';

    // TODO STD conditional :: if OP == abs return type = T

    typedef typename std::conditional<std::is_same<typename OP::RetType, void>::value,
                                      typename std::remove_pointer<T>::type, typename OP::RetType>::type ResultType;

    if constexpr (std::is_pointer<T>::value)
    {
        if (left.Data)
        {
            const int32_t retSize = left.DataAllocation.ElementCount;
            const bool allocateNullMask = left.DataAllocation.GpuNullMaskPtr;
            InstructionResult<ResultType> result =
                DispatcherInstructionHelper<ResultType>::AllocateInstructionResult(*this, reg, retSize, allocateNullMask,
                                                                                   {left.Name});
            if (isCompositeDataType<ResultType> || result.Data)
            {
                if (result.NullMaskPtr)
                {
                    const int32_t bitMaskSize = NullValues::GetNullBitMaskSize(retSize);
                    GPUMemory::copyDeviceToDevice(result.NullMaskPtr,
                                                  reinterpret_cast<nullmask_t*>(left.DataAllocation.GpuNullMaskPtr),
                                                  bitMaskSize);
                }
                GPUUnary<OP, ResultType, T>::Unary(result.Data, left.Data, retSize, result.NullMaskPtr);
                DispatcherInstructionHelper<ResultType>::StoreInstructionResult(result, *this, reg,
                                                                                retSize, allocateNullMask,
                                                                                {left.Name});

                if constexpr (std::is_same<OP, FilterConditions::logicalNot>::value)
                {
                    FreeRegisterNullMask(reg);
                }
            }
        }
        FreeColumnIfRegister<T>(left.Name);
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
            GPUUnary<OP, ResultType, T>::Unary(result.Data, left.Data, retSize, result.NullMaskPtr);
            DispatcherInstructionHelper<ResultType>::StoreInstructionResult(result, *this, reg, retSize,
                                                                            false, {left.Name});
        }
    }

    return InstructionStatus::CONTINUE;
}