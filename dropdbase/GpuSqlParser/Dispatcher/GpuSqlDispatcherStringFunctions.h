#pragma once

#include "../ParserExceptions.h"
#include "../GpuSqlDispatcher.h"
#include "../../QueryEngine/GPUCore/GPUMemory.cuh"
#include "../../QueryEngine/GPUCore/GPUStringUnary.cuh"
#include "../../QueryEngine/GPUCore/GPUStringBinary.cuh"

template<typename OP>
int32_t GpuSqlDispatcher::stringUnaryCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<std::string>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}
	
	std::cout << "StringUnaryCol: " << colName << " " << reg << std::endl;

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		throw StringGroupByException();
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		auto column = findStringColumn(getAllocatedRegisterName(colName));
		int32_t retSize = std::get<1>(column);

		if (!isRegisterAllocated(reg))
		{
			GPUMemory::GPUString result;
			GPUStringUnary::Col<OP>(result, std::get<0>(column), retSize);
			fillStringRegister(result, reg, retSize, true);
		}
	}
	return 0;
}

template<typename OP>
int32_t GpuSqlDispatcher::stringUnaryConst()
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

template<typename OP>
int32_t GpuSqlDispatcher::stringUnaryNumericCol()
{
	auto colName = arguments.read<std::string>();
	auto reg = arguments.read<std::string>();

	int32_t loadFlag = loadCol<std::string>(colName);
	if (loadFlag)
	{
		return loadFlag;
	}

	std::cout << "StringIntUnaryCol: " << colName << " " << reg << std::endl;

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		throw StringGroupByException();
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		auto column = findStringColumn(getAllocatedRegisterName(colName));
		int32_t retSize = std::get<1>(column);

		if (!isRegisterAllocated(reg))
		{
			int32_t* result = allocateRegister<int32_t>(reg, retSize);
			GPUStringUnary::Col<OP>(result, std::get<0>(column), retSize);
		}
	}
	return 0;
}

template<typename OP>
int32_t GpuSqlDispatcher::stringUnaryNumericConst()
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


template<typename OP, typename T>
int32_t GpuSqlDispatcher::stringBinaryNumericColCol()
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

	std::cout << "StringBinaryColCol: " << colNameLeft << " " << colNameRight << " " << reg << std::endl;

	if (groupByColumns.find(colNameLeft) != groupByColumns.end() || groupByColumns.find(colNameRight) != groupByColumns.end())
	{
		throw StringGroupByException();
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		auto columnLeft = findStringColumn(getAllocatedRegisterName(colNameLeft));
		std::tuple<uintptr_t, int32_t, bool> columnRight = allocatedPointers.at(getAllocatedRegisterName(colNameRight));
		int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

		if (!isRegisterAllocated(reg))
		{
			GPUMemory::GPUString result;
			GPUStringBinary::ColCol<OP>(result, std::get<0>(columnLeft), reinterpret_cast<T*>(std::get<0>(columnRight)), retSize);
			fillStringRegister(result, reg, retSize, true);
		}
	}
	return 0;
}

template<typename OP, typename T>
int32_t GpuSqlDispatcher::stringBinaryNumericColConst()
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

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		throw StringGroupByException();
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		auto column = findStringColumn(getAllocatedRegisterName(colName));
		int32_t retSize = std::get<1>(column);

		if (!isRegisterAllocated(reg))
		{
			GPUMemory::GPUString result;
			GPUStringBinary::ColConst<OP>(result, std::get<0>(column), cnst, retSize);
			fillStringRegister(result, reg, retSize, true);
		}
	}
	return 0;
}

template<typename OP, typename T>
int32_t GpuSqlDispatcher::stringBinaryNumericConstCol()
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

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		throw StringGroupByException();
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		GPUMemory::GPUString gpuString = insertConstStringGpu(cnst);
		std::tuple<uintptr_t, int32_t, bool> column = allocatedPointers.at(getAllocatedRegisterName(colName));
		int32_t retSize = std::get<1>(column);

		if (!isRegisterAllocated(reg))
		{
			GPUMemory::GPUString result;
			GPUStringBinary::ConstCol<OP>(result, gpuString, reinterpret_cast<T*>(std::get<0>(column)), retSize);
			fillStringRegister(result, reg, retSize, true);
		}
	}
	return 0;
}

template<typename OP, typename T>
int32_t GpuSqlDispatcher::stringBinaryNumericConstConst()
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

template<typename OP>
int32_t GpuSqlDispatcher::stringBinaryColCol()
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

	if (groupByColumns.find(colNameLeft) != groupByColumns.end() || groupByColumns.find(colNameRight) != groupByColumns.end())
	{
		throw StringGroupByException();
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		auto columnLeft = findStringColumn(getAllocatedRegisterName(colNameLeft));
		auto columnRight = findStringColumn(getAllocatedRegisterName(colNameRight));
		int32_t retSize = std::min(std::get<1>(columnLeft), std::get<1>(columnRight));

		if (!isRegisterAllocated(reg))
		{
			GPUMemory::GPUString result;
			GPUStringBinary::ColCol<OP>(result, std::get<0>(columnLeft), std::get<0>(columnRight), retSize);
			fillStringRegister(result, reg, retSize, true);
		}
	}
	return 0;
}

template<typename OP>
int32_t GpuSqlDispatcher::stringBinaryColConst()
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

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		throw StringGroupByException();
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		auto column = findStringColumn(getAllocatedRegisterName(colName));
		int32_t retSize = std::get<1>(column);

		if (!isRegisterAllocated(reg))
		{
			GPUMemory::GPUString result;
			GPUMemory::GPUString cnstString = insertConstStringGpu(cnst);
			GPUStringBinary::ColConst<OP>(result, std::get<0>(column), cnstString, retSize);
			fillStringRegister(result, reg, retSize, true);
		}
	}
	return 0;
}

template<typename OP>
int32_t GpuSqlDispatcher::stringBinaryConstCol()
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

	if (groupByColumns.find(colName) != groupByColumns.end())
	{
		throw StringGroupByException();
	}
	else if (isLastBlockOfDevice || !usingGroupBy)
	{
		GPUMemory::GPUString gpuString = insertConstStringGpu(cnst);
		auto column = findStringColumn(getAllocatedRegisterName(colName));
		int32_t retSize = std::get<1>(column);

		if (!isRegisterAllocated(reg))
		{
			GPUMemory::GPUString result;
			GPUStringBinary::ConstCol<OP>(result, gpuString, std::get<0>(column), retSize);
			fillStringRegister(result, reg, retSize, true);
		}
	}
	return 0;
}

template<typename OP>
int32_t GpuSqlDispatcher::stringBinaryConstConst()
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
		fillStringRegister(result, reg, 1, true);
	}
	return 0;
}
