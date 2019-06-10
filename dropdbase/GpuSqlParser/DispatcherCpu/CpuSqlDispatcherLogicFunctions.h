#pragma once
#include "../CpuSqlDispatcher.h"
#include <tuple>
#include <stack>


std::stack<>

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