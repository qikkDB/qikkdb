#pragma once
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPULogic.cuh"
#include "../../QueryEngine/GPUCore/GPUFilter.cuh"
#include "../../QueryEngine/GPUCore/GPUArithmetic.cuh"
#include "../../QueryEngine/GPUCore/GPUNullMask.cuh"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUFilterConditions.cuh"
#include "GpuSqlDispatcherVMFunctions.h"
#include <tuple>

/// Implementation of generic filter operation (<, >, =, ...) dispatching based on functor OP
/// Implementation for column constant case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::filterColConst()
{
	U cnst = arguments.read<U>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Filter: " << colName << " " << reg << std::endl;

	PointerAllocation column = allocatedPointers.at(getAllocatedRegisterName(colName));
	int32_t retSize = column.elementCount;

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask;
		if(column.gpuNullMaskPtr)
		{
			int8_t * nullMask;
			mask = allocateRegister<int8_t>(reg, retSize, &nullMask);
			int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
			GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
		}
		else
		{
			mask = allocateRegister<int8_t>(reg, retSize);
		}
		
		GPUFilter::colConst<OP, T, U>(mask, reinterpret_cast<T*>(column.gpuPtr), cnst, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), retSize);
	}

	freeColumnIfRegister<T>(colName);
	return 0;
}

/// Implementation of generic filter operation (<, >, =, ...) dispatching based on functor OP
/// Implementation for constant column case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::filterConstCol()
{
	auto colName = arguments.read<std::string>();
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "Filter: " << colName << " " << reg << std::endl;

	PointerAllocation column = allocatedPointers.at(getAllocatedRegisterName(colName));
	int32_t retSize = column.elementCount;

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask;
		if(column.gpuNullMaskPtr)
		{
			int8_t * nullMask;
			mask = allocateRegister<int8_t>(reg, retSize, &nullMask);
			int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
			GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
		}
		else
		{
			mask = allocateRegister<int8_t>(reg, retSize);
		}
		
		GPUFilter::constCol<OP, T, U>(mask, cnst, reinterpret_cast<U*>(column.gpuPtr), reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), retSize);
	}

	freeColumnIfRegister<U>(colName);
	return 0;
}

/// Implementation of generic filter operation (<, >, =, ...) dispatching based on functor OP
/// Implementation for column column case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::filterColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

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

	std::cout << "Filter: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	PointerAllocation columnRight = allocatedPointers.at(getAllocatedRegisterName(colNameRight));
	PointerAllocation columnLeft = allocatedPointers.at(getAllocatedRegisterName(colNameLeft));
	int32_t retSize = std::min(columnLeft.elementCount, columnRight.elementCount);

	if (!isRegisterAllocated(reg))
	{
		if(columnLeft.gpuNullMaskPtr || columnRight.gpuNullMaskPtr)
		{
			int8_t * combinedMask;
			int8_t * mask = allocateRegister<int8_t>(reg, retSize, &combinedMask);
			int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
			if(columnLeft.gpuNullMaskPtr && columnRight.gpuNullMaskPtr)
			{
				GPUArithmetic::colCol<ArithmeticOperations::bitwiseOr>(combinedMask, reinterpret_cast<int8_t*>(columnLeft.gpuNullMaskPtr), reinterpret_cast<int8_t*>(columnRight.gpuNullMaskPtr), bitMaskSize);
			}
			if(columnLeft.gpuNullMaskPtr)
			{
				GPUMemory::copyDeviceToDevice(combinedMask, reinterpret_cast<int8_t*>(columnLeft.gpuNullMaskPtr), bitMaskSize);
			}
			else if(columnRight.gpuNullMaskPtr)
			{
				GPUMemory::copyDeviceToDevice(combinedMask, reinterpret_cast<int8_t*>(columnRight.gpuNullMaskPtr), bitMaskSize);
			}
			GPUFilter::colCol<OP, T, U>(mask, reinterpret_cast<T*>(columnLeft.gpuPtr), reinterpret_cast<U*>(columnRight.gpuPtr), combinedMask, retSize);
		}
		else
		{
			int8_t * mask = allocateRegister<int8_t>(reg, retSize);
			GPUFilter::colCol<OP, T, U>(mask, reinterpret_cast<T*>(columnLeft.gpuPtr), reinterpret_cast<U*>(columnRight.gpuPtr), nullptr, retSize);
		}
		
	}

	freeColumnIfRegister<U>(colNameRight);
	freeColumnIfRegister<T>(colNameLeft);
	return 0;
}

/// Implementation of genric filter operation (<, >, =, ...) dispatching based on functor OP
/// Implementation for constant constant case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::filterConstConst()
{
	U constRight = arguments.read<U>();
	T constLeft = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, database->GetBlockSize());
		GPUFilter::constConst<OP, T, U>(mask, constLeft, constRight, database->GetBlockSize());
	}
	return 0;
}

template<typename OP>
int32_t GpuSqlDispatcher::filterStringColConst()
{
	std::string cnst = arguments.read<std::string>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<std::string>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "FilterStringColConst: " << colName << " " << cnst << " " << reg << std::endl;

	auto column = findStringColumn(getAllocatedRegisterName(colName));
	int32_t retSize = std::get<1>(column);
	int8_t* nullBitMask = std::get<2>(column);

	if (!isRegisterAllocated(reg))
	{
		GPUMemory::GPUString constString = insertConstStringGpu(cnst);
		int8_t * mask;
		if(nullBitMask)
		{
			int8_t * nullMask;
			mask = allocateRegister<int8_t>(reg, retSize, &nullMask);
			int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
			GPUMemory::copyDeviceToDevice(nullMask, nullBitMask, bitMaskSize);
		}
		else
		{
			mask = allocateRegister<int8_t>(reg, retSize);
		}
		GPUFilter::colConst<OP>(mask, std::get<0>(column), constString, nullBitMask, retSize);
	}
	return 0;
}

template<typename OP>
int32_t GpuSqlDispatcher::filterStringConstCol()
{
	auto colName = arguments.read<std::string>();
	std::string cnst = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<std::string>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "FilterStringConstCol: " << cnst << " " << colName << " " << reg << std::endl;

	std::tuple<GPUMemory::GPUString, int32_t, int8_t*> column = findStringColumn(getAllocatedRegisterName(colName));
	int32_t retSize = std::get<1>(column);
	int8_t* nullBitMask = std::get<2>(column);
	if (!isRegisterAllocated(reg))
	{
		GPUMemory::GPUString constString = insertConstStringGpu(cnst);
		int8_t * mask;
		if(nullBitMask)
		{
			int8_t * nullMask;
			mask = allocateRegister<int8_t>(reg, retSize, &nullMask);
			int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
			GPUMemory::copyDeviceToDevice(nullMask, nullBitMask, bitMaskSize);
		}
		else
		{
			mask = allocateRegister<int8_t>(reg, retSize);
		}
		GPUFilter::constCol<OP>(mask, constString, std::get<0>(column), nullBitMask, retSize);
	}
	return 0;
}

template<typename OP>
int32_t GpuSqlDispatcher::filterStringColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<std::string>(colNameRight);
	if (loadFlag)
	{
		return loadFlag;
	}
	loadFlag = loadCol<std::string>(colNameLeft);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "FilterStringColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	std::tuple<GPUMemory::GPUString, int32_t, int8_t*> columnLeft = findStringColumn(getAllocatedRegisterName(colNameLeft));
	std::tuple<GPUMemory::GPUString, int32_t, int8_t*> columnRight = findStringColumn(getAllocatedRegisterName(colNameRight));
	int32_t retSize = std::max(std::get<1>(columnLeft), std::get<1>(columnRight));
	int8_t* leftMask = std::get<2>(columnLeft);
	int8_t* rightMask = std::get<2>(columnRight);
	if (!isRegisterAllocated(reg))
	{
		if(leftMask || rightMask)
		{
			int8_t * combinedMask;
			int8_t * mask = allocateRegister<int8_t>(reg, retSize, &combinedMask);
			int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
			if(leftMask && rightMask)
			{
				GPUArithmetic::colCol<ArithmeticOperations::bitwiseOr>(combinedMask, leftMask, rightMask, bitMaskSize);
			}
			if(leftMask)
			{
				GPUMemory::copyDeviceToDevice(combinedMask, leftMask, bitMaskSize);
			}
			else if(rightMask)
			{
				GPUMemory::copyDeviceToDevice(combinedMask, rightMask, bitMaskSize);
			}
			GPUFilter::colCol<OP>(mask, std::get<0>(columnLeft), std::get<0>(columnRight), combinedMask, retSize);
		}

		else
		{
			int8_t * mask = allocateRegister<int8_t>(reg, retSize);
			GPUFilter::colCol<OP>(mask, std::get<0>(columnLeft), std::get<0>(columnRight), nullptr, retSize);
		}
		
	}
	return 0;
}


template<typename OP>
int32_t GpuSqlDispatcher::filterStringConstConst()
{
	std::string cnstRight = arguments.read<std::string>();
	std::string cnstLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	std::cout << "FilterStringConstConst: " << cnstLeft << " " << cnstRight << " " << reg << std::endl;

	if (!isRegisterAllocated(reg))
	{
		GPUMemory::GPUString constStringLeft = insertConstStringGpu(cnstLeft);
		GPUMemory::GPUString constStringRight = insertConstStringGpu(cnstRight);

		int8_t * mask = allocateRegister<int8_t>(reg, database->GetBlockSize());
		GPUFilter::constConst<OP>(mask, constStringLeft, constStringRight, database->GetBlockSize());
	}
	return 0;
}

/// Implementation of generic logical operation (AND, OR) dispatching based on functor OP
/// Implementation for column constant case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::logicalColConst()
{
	U cnst = arguments.read<U>();
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	PointerAllocation column = allocatedPointers.at(getAllocatedRegisterName(colName));
	int32_t retSize = column.elementCount;

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask;
		if(column.gpuNullMaskPtr)
		{
			int8_t * nullMask;
			mask = allocateRegister<int8_t>(reg, retSize, &nullMask);
			int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
			GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
		}
		else
		{
			mask = allocateRegister<int8_t>(reg, retSize);
		}
		
		GPULogic::colConst<OP, T, U>(mask, reinterpret_cast<T*>(column.gpuPtr), cnst, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), retSize);
	}

	freeColumnIfRegister<T>(colName);
	return 0;
}

/// Implementation of generic logical operation (AND, OR) dispatching based on functor OP
/// Implementation for constant column case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::logicalConstCol()
{
	auto colName = arguments.read<std::string>();
	T cnst = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<U>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	PointerAllocation column = allocatedPointers.at(getAllocatedRegisterName(colName));
	int32_t retSize = column.elementCount;

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask;
		if(column.gpuNullMaskPtr)
		{
			int8_t * nullMask;
			mask = allocateRegister<int8_t>(reg, retSize, &nullMask);
			int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
			GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
		}
		else
		{
			mask = allocateRegister<int8_t>(reg, retSize);
		}
		
		GPULogic::constCol<OP, T, U>(mask, cnst, reinterpret_cast<U*>(column.gpuPtr), reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), retSize);
	}

	freeColumnIfRegister<U>(colName);
	return 0;
}

/// Implementation of generic logical operation (AND, OR) dispatching based on functor OP
/// Implementation for column column case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::logicalColCol()
{
	auto colNameRight = arguments.read<std::string>();
	auto colNameLeft = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

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

	std::cout << "Logical: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	PointerAllocation columnRight = allocatedPointers.at(getAllocatedRegisterName(colNameRight));
	PointerAllocation columnLeft = allocatedPointers.at(getAllocatedRegisterName(colNameLeft));

	int32_t retSize = std::min(columnLeft.elementCount, columnRight.elementCount);

	if (!isRegisterAllocated(reg))
	{
		if(columnLeft.gpuNullMaskPtr || columnRight.gpuNullMaskPtr)
		{
			int8_t * combinedMask;
			int8_t * mask = allocateRegister<int8_t>(reg, retSize, &combinedMask);
			int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
			if(columnLeft.gpuNullMaskPtr && columnRight.gpuNullMaskPtr)
			{
				GPUArithmetic::colCol<ArithmeticOperations::bitwiseOr>(combinedMask, reinterpret_cast<int8_t*>(columnLeft.gpuNullMaskPtr), reinterpret_cast<int8_t*>(columnRight.gpuNullMaskPtr), bitMaskSize);
			}
			if(columnLeft.gpuNullMaskPtr)
			{
				GPUMemory::copyDeviceToDevice(combinedMask, reinterpret_cast<int8_t*>(columnLeft.gpuNullMaskPtr), bitMaskSize);
				
			}
			else if(columnRight.gpuNullMaskPtr)
			{
				GPUMemory::copyDeviceToDevice(combinedMask, reinterpret_cast<int8_t*>(columnRight.gpuNullMaskPtr), bitMaskSize);
			}
			GPULogic::colCol<OP, T, U>(mask, reinterpret_cast<T*>(columnLeft.gpuPtr), reinterpret_cast<U*>(columnRight.gpuPtr), combinedMask, retSize);
		}
		else
		{
			int8_t * mask = allocateRegister<int8_t>(reg, retSize);
			GPULogic::colCol<OP, T, U>(mask, reinterpret_cast<T*>(columnLeft.gpuPtr), reinterpret_cast<U*>(columnRight.gpuPtr), nullptr, retSize);
		}
	}

	freeColumnIfRegister<U>(colNameRight);
	freeColumnIfRegister<T>(colNameLeft);
	return 0;
}

/// Implementation of generic logical operation (AND, OR) dispatching based on functor OP
/// Implementation for constant constant case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename OP, typename T, typename U>
int32_t GpuSqlDispatcher::logicalConstConst()
{
	U constRight = arguments.read<U>();
	T constLeft = arguments.read<T>();
	auto reg = arguments.read<std::string>();

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask = allocateRegister<int8_t>(reg, database->GetBlockSize());
		GPULogic::constConst<OP, T, U>(mask, constLeft, constRight, database->GetBlockSize());
	}

	return 0;
}

/// Implementation of NOT operation dispatching
/// Implementation for column case
/// Pops data from argument memory stream, and loads data to GPU on demand 
/// <returns name="statusCode">Finish status code of the operation</returns>
template<typename T>
int32_t GpuSqlDispatcher::logicalNotCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<T>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "NotCol: " << colName << " " << reg << std::endl;

	PointerAllocation column = allocatedPointers.at(getAllocatedRegisterName(colName));
	int32_t retSize = column.elementCount;

	if (!isRegisterAllocated(reg))
	{
		int8_t * mask;
		if(column.gpuNullMaskPtr)
		{
			int8_t * nullMask;
			mask = allocateRegister<int8_t>(reg, retSize, &nullMask);
			int32_t bitMaskSize = ((retSize + sizeof(int8_t)*8 - 1) / (8*sizeof(int8_t)));
			GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), bitMaskSize);
		}
		else
		{
			mask = allocateRegister<int8_t>(reg, retSize);
		}
		GPULogic::not_col<int8_t, T>(mask, reinterpret_cast<T*>(column.gpuPtr), reinterpret_cast<int8_t*>(column.gpuNullMaskPtr), retSize);
	}

	freeColumnIfRegister<T>(colName);
	return 0;
}

template<typename T>
int32_t GpuSqlDispatcher::logicalNotConst()
{
	return 0;
}


template<typename OP>
int32_t GpuSqlDispatcher::nullMaskCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	std::cout << "NullMaskCol: " << colName << " " << reg << std::endl;

	if (colName.front() == '$')
	{
		throw NullMaskOperationInvalidOperandException();
	}

	int32_t loadFlag = loadColNullMask(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	PointerAllocation columnMask = allocatedPointers.at(colName + NULL_SUFFIX);
	size_t nullMaskSize = (columnMask.elementCount + 8 * sizeof(int8_t) - 1) / (8 * sizeof(int8_t));

	if (!isRegisterAllocated(reg))
	{
		int8_t * outFilterMask;
		
		int8_t * nullMask;
		outFilterMask = allocateRegister<int8_t>(reg, columnMask.elementCount, &nullMask);
		GPUMemory::copyDeviceToDevice(nullMask, reinterpret_cast<int8_t*>(columnMask.gpuPtr), nullMaskSize);
		GPUNullMask::Col<OP>(outFilterMask, reinterpret_cast<int8_t*>(columnMask.gpuPtr), nullMaskSize, columnMask.elementCount);
	}
	return 0;
}
