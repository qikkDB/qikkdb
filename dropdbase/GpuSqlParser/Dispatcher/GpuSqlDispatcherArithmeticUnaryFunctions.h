#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmeticUnary.cuh"

/// Implementation of generic unary arithmetic function dispatching given by the functor OP
/// Implementation for column case
/// Pops data from argument memory stream and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T>
int32_t GpuSqlDispatcher::arithmeticUnaryCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	// TODO STD conditional :: if OP == abs return type = T

	typedef typename std::conditional < OP::isFloatRetType, float, T>::type ResultType;

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticUnaryCol: " << colName << " " << reg << std::endl;

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		if (isLastBlockOfDevice)
		{
			PointerAllocation column = allocatedPointers.at(colName + "_keys");
			int32_t retSize = column.elementCount;
			ResultType * result;
			if(column.gpuNullMaskPtr)
			{
				int8_t * nullMask;
				result = allocateRegister<ResultType>(reg + "_keys", retSize, &nullMask);
				int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
				GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
			}
			else
			{
				result = allocateRegister<ResultType>(reg + "_keys", retSize);
			}
			GPUArithmeticUnary::col<OP, ResultType, T>(result, reinterpret_cast<T*>(column.gpuPtr), retSize);
			groupByColumns.insert(reg);
		}
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		PointerAllocation column = allocatedPointers.at(colName);
		int32_t retSize = column.elementCount;
		if (!isRegisterAllocated(reg))
		{
			ResultType * result;
			if(column.gpuNullMaskPtr)
			{
				int8_t * nullMask;
				result = allocateRegister<ResultType>(reg + "_keys", retSize, &nullMask);
				int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
				GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
			}
			else
			{
				result = allocateRegister<ResultType>(reg + "_keys", retSize);
			}
			GPUArithmeticUnary::col<OP, ResultType, T>(result, reinterpret_cast<T*>(column.elementCount), retSize);
		}
	}
	freeColumnIfRegister<T>(colName);
	return 0;
}

/// Implementation of generic unary arithmetic function dispatching given by the functor OP
/// Implementation for constant case
/// Pops data from argument memory stream and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T>
int32_t GpuSqlDispatcher::arithmeticUnaryConst()
{
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	// TODO STD conditional :: if OP == abs return type = T
	typedef typename std::conditional < OP::isFloatRetType, float, T > ::type ResultType;

	std::cout << "ArithmeticUnaryConst: " << reg << std::endl;

	int32_t retSize = 1;

	if (!isRegisterAllocated(reg))
	{
		ResultType * result = allocateRegister<ResultType>(reg, retSize);
		GPUArithmeticUnary::cnst<OP, ResultType, T>(result, cnst, retSize);
	}

	return 0;
}