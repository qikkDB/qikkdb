#include <memory>

#include "Context.h"
#include "EngineCore.h"

Context& Context::getInstance()
{
	// Static instance - constructor called only once
	static Context instance(std::make_unique<EngineCore>());
	return instance;
}


int32_t Context::calcGridDim(int32_t threadCount)
{
	int blockCount = (threadCount + queried_block_dimension_ - 1) / queried_block_dimension_;
	if (blockCount >= (DEFAULT_GRID_DIMENSION_LIMIT + 1))
	{
		blockCount = DEFAULT_GRID_DIMENSION_LIMIT;
	}
	return blockCount;
}