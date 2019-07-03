#pragma once

#include <vector>
#include "IVariantArray.h"

template<typename T>
class VariantArray : public IVariantArray
{
public:
	virtual int64_t GetSize() const override
	{
		return data.size();
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

	void resize(size_t size)
	{
		data.resize(size);
	}

	T* getData()
	{
		return data.data();
	}

private:
	std::vector<T> data;
};