#include <memory>
#include <tuple>
#include <string>
#include <unoredered_map>
#include "MemoryStream.h"
#include "../DataType.h"

class CpuSqlDispatcher
{
private:
	typedef int32_t(CpuSqlDispatcher::*CpuDispatchFunction)();
	std::vector<CpuDispatchFunction> cpuDispatcherFunctions;
	MemoryStream arguments;

	std::unordered_map<std::string, std::tuple<std::uintptr_t, int32_t, bool>> allocatedPointers;

	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterEqualFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessEqualFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> equalFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> notEqualFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalAndFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalOrFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> mulFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> divFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> addFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> subFunctions;
	static std::array<CpuDispatchFunction,
		DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> modFunctions;

public:
	template<typename T>
	T* allocateRegister(const std::string& reg, int32_t size)
	{
		T * mask;
		GPUMemory::alloc<T>(&mask, size);
		allocatedPointers.insert({ reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(mask), size, true) });
		usedRegisterMemory += size * sizeof(T);
		return mask;
	}
};