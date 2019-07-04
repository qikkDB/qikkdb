#pragma once

#include <vector>
#include <memory>
#include "IVariantArray.h"

template<typename T>
class VariantArray : public IVariantArray
{
public:
	VariantArray(int32_t size) :
		data(std::make_unique<T[]>(size)),
		size(size)
	{

	}

	virtual int64_t GetSize() const override
	{
		return size;
	}

	virtual DataType GetType() const override
	{
		typedef typename std::conditional<
			std::is_same<T, int>::value, std::integral_constant<DataType, COLUMN_INT>,
			typename std::conditional<
			std::is_same<T, int64_t>::value, std::integral_constant<DataType, COLUMN_LONG>,
			typename std::conditional<
			std::is_same<T, float>::value, std::integral_constant<DataType, COLUMN_FLOAT>,
			typename std::conditional<
			std::is_same<T, double>::value, std::integral_constant<DataType, COLUMN_DOUBLE>,
			typename std::conditional<
			std::is_same<T, ColmnarDB::Types::Point>::value, std::integral_constant<DataType, COLUMN_POINT>,
			typename std::conditional<
			std::is_same<T, ColmnarDB::Types::ComplexPolygon>::value, std::integral_constant<DataType, COLUMN_POLYGON>,
			typename std::conditional<std::is_same<T, std::string>::value, std::integral_constant<DataType, COLUMN_STRING>,
			typename std::conditional<std::is_same<T, bool>::value, std::integral_constant<DataType, COLUMN_INT8_T>,
			typename std::conditional<std::is_same<T, int8_t>::value, std::integral_constant<DataType, COLUMN_INT8_T>, std::integral_constant<DataType, CONST_ERROR>>::type>::
			type>::type>::type>::type>::type>::type>::type>::type retConst;
		return retConst::value;
	};

	T* getData()
	{
		return data.get();
	}

	std::unique_ptr<T[]>& getDataRef()
	{
		return data;
	}

	void resize (int32_t newSize)
	{
		size = newSize;
	}

private:
	std::unique_ptr<T[]> data;
	int32_t size;
};