#include "CudaLogBoost.h"
#include <boost/log/trivial.hpp>

CudaLogBoost & CudaLogBoost::operator<<(const std::string & text)
{
	switch (severity)
	{
	case trace:
		BOOST_LOG_TRIVIAL(trace) << text;
		break;
	case debug:
		BOOST_LOG_TRIVIAL(debug) << text;
		break;
	case info:
		BOOST_LOG_TRIVIAL(info) << text;
		break;
	case warning:
		BOOST_LOG_TRIVIAL(warning) << text;
		break;
	case error:
		BOOST_LOG_TRIVIAL(error) << text;
		break;
	default:
		BOOST_LOG_TRIVIAL(info) << text;
		break;
	}
	return *this;
}

CudaLogBoost::CudaLogBoost(Severity severity) :
	severity(severity)
{
}
