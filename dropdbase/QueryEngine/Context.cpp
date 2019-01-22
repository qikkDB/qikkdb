#include <memory>
#include "Context.cuh"
#include "EngineCore.cuh"


Context& Context::getInstance()
{
	// Static instance - constructor called only once
	static Context instance(std::make_unique<EngineCore>());
	return instance;
}
