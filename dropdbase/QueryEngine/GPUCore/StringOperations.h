#pragma once

#include <string>
#include <utility>
#include <boost/algorithm/string.hpp>

namespace StringUnaryOperationsCpu
{
	struct ltrim
	{
		__host__ std::string operator()(const char* str, int32_t len) const
		{
			std::string strObj(str, len);
			boost::trim_left(strObj);
			return strObj;
		}
	};
}

namespace StringBinaryOperationsCpu
{

}