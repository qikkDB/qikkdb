#include "QueryEngine/Context.h"
#include "QueryEngine/GPUCore/GPUMemory.cuh"

int main(int argc, char** argv)
{
	Context::getInstance();

	int32_t *foo;
	GPUMemory::alloc(&foo, 1);
	GPUMemory::fill(foo, 0, 1);

	GPUMemory::free(foo);

	return 0;
}