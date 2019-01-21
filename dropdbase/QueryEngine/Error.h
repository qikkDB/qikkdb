#ifndef ERROR_H
#define ERROR_H

class Error {
public:
	enum Type {
		GPU_EXTENSION_SUCCESS,					// Return code for successful operations
		GPU_EXTENSION_ERROR,					// Return code for all CUDA errors
		GPU_DIVISION_BY_ZERO_ERROR,				// Return code for division by zero
		GPU_INTEGER_OVERFLOW_ERROR,				// Return code for integer overflow
		GPU_UNSUPPORTED_DATA_TYPE,				// Return code for too big buffer (e.g. at group by with too many buckets)
		GPU_BUFFER_TOO_BIG,						// Return code for too big buffer (e.g. at group by with too many buckets)
		GPU_NOT_FOUND_ERROR,					// Return code for no detected GPU
		GPU_MEMORY_MAPPING_NOT_SUPPORTED_ERROR,	// Return code for no memory mapping
		GPU_DRIVER_NOT_FOUND_EXCEPTION			// Return code for not found nvidia driver
	}

private:
	Type type_;
	std::string text_;

public:
	void setType(Type type) {
		type_ = type;
	}

	void setText(std::string& text) {
		text_ = text;
	}

	const Type& getType() const {
		return lastType;
	}

	const std::string& getText() const {
		retrun text_;
	}

};

#endif
