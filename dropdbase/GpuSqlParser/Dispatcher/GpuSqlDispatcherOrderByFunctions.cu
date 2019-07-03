#include "GpuSqlDispatcherOrderByFunctions.h"

std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::orderByFunctions = { &GpuSqlDispatcher::orderByConst<int32_t>, &GpuSqlDispatcher::orderByConst<int64_t>, &GpuSqlDispatcher::orderByConst<float>, &GpuSqlDispatcher::orderByConst<double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<std::string>, &GpuSqlDispatcher::orderByConst<int8_t>, &GpuSqlDispatcher::orderByCol<int32_t>, &GpuSqlDispatcher::orderByCol<int64_t>, &GpuSqlDispatcher::orderByCol<float>, &GpuSqlDispatcher::orderByCol<double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<std::string>, &GpuSqlDispatcher::orderByCol<int8_t> };
std::array<GpuSqlDispatcher::DispatchFunction, DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::orderByReconstructFunctions = { &GpuSqlDispatcher::orderByReconstructConst<int32_t>, &GpuSqlDispatcher::orderByReconstructConst<int64_t>, &GpuSqlDispatcher::orderByReconstructConst<float>, &GpuSqlDispatcher::orderByReconstructConst<double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<std::string>, &GpuSqlDispatcher::orderByReconstructConst<int8_t>, &GpuSqlDispatcher::orderByReconstructCol<int32_t>, &GpuSqlDispatcher::orderByReconstructCol<int64_t>, &GpuSqlDispatcher::orderByReconstructCol<float>, &GpuSqlDispatcher::orderByReconstructCol<double>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon>, &GpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<std::string>, &GpuSqlDispatcher::orderByReconstructCol<int8_t> };

GpuSqlDispatcher::DispatchFunction GpuSqlDispatcher::freeOrderByTableFunction = &GpuSqlDispatcher::freeOrderByTable;

int32_t GpuSqlDispatcher::freeOrderByTable()
{
	std::cout << "Freeing order by table." << std::endl;
	orderByTable.release();
	return 0;
}

int32_t GpuSqlDispatcher::orderByReconstructInputColsGlobal()
{
	/*
	// Tieto polia nie su uniformne velke !!!, treba to zistit z IVariantArray
	std::unordered_map<std::string, std::vector<std::unique_ptr<IVariantArray>>> reconstructedOrderByColumnBlocks;
	std::unordered_map<std::string, std::unique_ptr<IVariantArray>> reconstructedOrderByColumnsMerged;
	std::unordered_map<int32_t, std::pair<std::string, OrderBy::Order>> orderByColumns;
	std::vector<std::vector<int32_t>> orderByIndices;
	*/





	return 0;
}
