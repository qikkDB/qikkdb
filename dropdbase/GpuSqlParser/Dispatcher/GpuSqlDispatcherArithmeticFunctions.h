#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"

/// Implementation of generic binary arithmetic function dispatching given by the functor OP
/// Implementation for column constant case
/// Pops data from argument memory stream and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::arithmeticColConst()
{
	U cnst = arguments.read<U>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
	>::type ResultType;
	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticColConst: " << colName << " " << reg << std::endl;

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
			GPUArithmetic::colConst<OP, ResultType, T, U>(result, reinterpret_cast<T*>(column.gpuPtr), cnst, retSize);
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
				result = allocateRegister<ResultType>(reg, retSize, &nullMask);
				int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
				GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
			}
			else
			{
				result = allocateRegister<ResultType>(reg, retSize);
			}
			GPUArithmetic::colConst<OP, ResultType, T, U>(result, reinterpret_cast<T*>(column.gpuPtr), cnst, retSize);
		}
	}
	freeColumnIfRegister<T>(colName);
	return 0;
}

/// Implementation of generic binary arithmetic function dispatching given by the functor OP
/// Implementation for constant column case
/// Pops data from argument memory stream and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::arithmeticConstCol()
{
	auto colName = arguments.read<std::string>();
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();


	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
	>::type ResultType;
	int32_t loadFlag = loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticConstCol: " << colName << " " << reg << std::endl;

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
			GPUArithmetic::constCol<OP, ResultType, T, U>(result, cnst, reinterpret_cast<U*>(column.gpuPtr), retSize);
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
				result = allocateRegister<ResultType>(reg, retSize, &nullMask);
				int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
				GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
			}
			else
			{
				result = allocateRegister<ResultType>(reg, retSize);
			}
			GPUArithmetic::constCol<OP, ResultType, T, U>(result, cnst, reinterpret_cast<U*>(column.gpuPtr), retSize);
		}
	}
	freeColumnIfRegister<U>(colName);
	return 0;
}

/// Implementation of generic binary arithmetic function dispatching given by the functor OP
/// Implementation for column column case
/// Pops data from argument memory stream and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::arithmeticColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();
	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point<U>::value, U, void>::type>::type
	>::type ResultType;

	int32_t loadFlag = loadCol<U>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = loadCol<T>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "ArithmeticColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	if (groupByColumns.find(colNameRight) != groupByColumns.end())
	{
		if (isLastBlockOfDevice)
		{
			PointerAllocation columnRight = allocatedPointers.at(colNameRight + "_keys");
			PointerAllocation columnLeft = allocatedPointers.at(colNameLeft);
			int32_t retSize = std::min(columnLeft.elementCount, columnRight.elementCount);

			ResultType * result = allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::colCol<OP, ResultType, T, U>(result, reinterpret_cast<T*>(columnLeft.gpuPtr), reinterpret_cast<U*>(columnRight.gpuPtr), retSize);
			groupByColumns.insert(reg);
		}
	}
	else if (groupByColumns.find(colNameLeft) != groupByColumns.end())
	{
		if (isLastBlockOfDevice)
		{
			PointerAllocation columnRight = allocatedPointers.at(colNameRight);
			PointerAllocation columnLeft = allocatedPointers.at(colNameLeft + "_keys");
			int32_t retSize = std::min(columnLeft.elementCount, columnRight.elementCount);

			ResultType * result = allocateRegister<ResultType>(reg + "_keys", retSize);
			GPUArithmetic::colCol<OP, ResultType, T, U>(result, reinterpret_cast<T*>(columnLeft.gpuPtr), reinterpret_cast<U*>(columnRight.gpuPtr), retSize);
			groupByColumns.insert(reg);
		}
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		PointerAllocation columnRight = allocatedPointers.at(colNameRight);
		PointerAllocation columnLeft = allocatedPointers.at(colNameLeft);
		int32_t retSize = std::min(columnLeft.elementCount, columnRight.elementCount);

		if (!isRegisterAllocated(reg))
		{
			ResultType * result;
			if(columnLeft.gpuNullMaskPtr && columnRight.gpuNullMaskPtr)
			{
				int8_t * combinedMask;
				result = allocateRegister<ResultType>(reg, retSize, &combinedMask);
				int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
				GPUArithmetic::colCol<ArithmeticOperations::bitwiseOr>(combinedMask, reinterpret_cast<int8_t*>(columnLeft.gpuNullMaskPtr), reinterpret_cast<int8_t*>(columnRight.gpuNullMaskPtr), bitMaskSize);
				GPUFilter::colCol<OP, T, U>(mask, reinterpret_cast<T*>(columnLeft.gpuPtr), reinterpret_cast<U*>(columnRight.gpuPtr), combinedMask, retSize);
			}
			else if(columnLeft.gpuNullMaskPtr)
			{
				int8_t * combinedMask;
				result = allocateRegister<ResultType>(reg, retSize, &combinedMask);
				int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
				GPUMemory::copyDeviceToDevice(combinedMask, reinterpret_cast<int8_t*>(columnLeft.gpuNullMaskPtr), bitMaskSize);
			}
			else if(columnRight.gpuNullMaskPtr)
			{
				int8_t * combinedMask;
				int8_t * mask = allocateRegister<int8_t>(reg, retSize, &combinedMask);
				int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
				GPUMemory::copyDeviceToDevice(combinedMask, reinterpret_cast<int8_t*>(columnRight.gpuNullMaskPtr), bitMaskSize);
			}
			else
			{
				int8_t * mask = allocateRegister<int8_t>(reg, retSize);
			}
			GPUArithmetic::colCol<OP, ResultType, T, U>(result, reinterpret_cast<T*>(columnLeft.gpuPtr), reinterpret_cast<U*>(columnRight.gpuPtr), retSize);
		}
	}
	freeColumnIfRegister<T>(colNameLeft);
	freeColumnIfRegister<U>(colNameRight);
	return 0;
}

/// Implementation of generic binary arithmetic function dispatching given by the functor OP
/// Implementation for constant constant case
/// Pops data from argument memory stream and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::arithmeticConstConst()
{
	U constRight = arguments.read<U>();
	T constLeft = arguments.read<T>();
	auto reg = arguments.read<std::string>();
	constexpr bool bothTypesFloatOrBothIntegral =
		std::is_floating_point<T>::value && std::is_floating_point<U>::value ||
		std::is_integral<T>::value && std::is_integral<U>::value;
	typedef typename std::conditional< bothTypesFloatOrBothIntegral,
		typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type,
		typename std::conditional<std::is_floating_point<T>::value, T,
		typename std::conditional<std::is_floating_point< U>::value, U, void>::type>::type
	>::type ResultType;
	std::cout << "ArithmeticConstConst: " << reg << std::endl;

	int32_t retSize = 1;

	if (!isRegisterAllocated(reg))
	{
		ResultType * result = allocateRegister<ResultType>(reg, retSize);
		GPUArithmetic::constConst<OP, ResultType, T, U>(result, constLeft, constRight, retSize);
	}
	return 0;
}