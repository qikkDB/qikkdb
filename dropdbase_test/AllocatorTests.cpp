#include "gtest/gtest.h"
#include "../dropdbase/QueryEngine/GPUMemoryCache.h"
#include "../dropdbase/DatabaseGenerator.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/Database.h"
#include "../dropdbase/ComplexPolygonFactory.h"
#include "../dropdbase/PointFactory.h"
#include "../dropdbase/ColumnBase.h"

TEST(AllocatorTests, AllocateDeallocate)
{
	auto& context = Context::getInstance();

	context.getAllocatorForCurrentDevice();
}