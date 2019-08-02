#pragma once
#include <string>

class CudaLogBoost
{
public:
	enum Severity
	{
		trace,
		debug,
		info,
		warning,
		error,
		fatal,
	};

	static CudaLogBoost& getInstance(Severity severity)
	{
		static CudaLogBoost instanceTrace(Severity::trace);
		static CudaLogBoost instanceDebug(Severity::debug);
		static CudaLogBoost instanceInfo(Severity::info);
		static CudaLogBoost instanceWarning(Severity::warning);
		static CudaLogBoost instanceError(Severity::error);
		static CudaLogBoost instanceFatal(Severity::fatal);

		switch (severity)
		{
		case trace:
			return instanceTrace;
			break;
		case debug:
			return instanceDebug;
			break;
		case info:
			return instanceInfo;
			break;
		case warning:
			return instanceWarning;
			break;
		case error:
			return instanceError;
			break;
		case fatal:
			return instanceFatal;
			break;
		default:
			return instanceInfo;
			break;
		}
	}

	CudaLogBoost(CudaLogBoost const&) = delete;
	void operator=(CudaLogBoost const&) = delete;

	CudaLogBoost& operator<<(const std::string& text);

private:
	Severity severity;
	CudaLogBoost(Severity severity);
};