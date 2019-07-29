#include "BlockBase.h"
#include "PointFactory.h"
#include "ComplexPolygonFactory.h"
#include "Types/ComplexPolygon.pb.h"
#include "Types/Point.pb.h"
#include <numeric>
#include <algorithm>
#include <cmath>


template<>
void BlockBase<int32_t>::setBlockStatistics(const std::vector<int32_t>& data)
{
	if (isCompressed_)
	{
		return;
	}
	
	if(!isNullable_)
	{	
		for (int32_t i = 0; i < data.size(); i++)
		{
			updateBlockStatistics(data[i], false);
		}
	}
	else
	{
		size_ += data.size();

		min_ = std::numeric_limits<int32_t>::max();
		max_ = std::numeric_limits<int32_t>::lowest();
		avg_ = 0;
		sum_ = 0;
		size_t count = 0;
		countOfNotNullValues_ = 0;
		isFullOfNullValue_ = true;
		for (int i = 0; i < size_; i++)
		{
			bool isNull = bitMask_[i / (sizeof(int8_t) * 8)] & (1 << (i % (sizeof(int8_t) * 8)));
			if (!isNull)
			{
				countOfNotNullValues_++;

				isFullOfNullValue_ = false;
				min_ = std::min(min_, data_[i]);
				max_ = std::max(max_, data_[i]);
				sum_ += data_[i];
				avg_ += data_[i];
				count++;
			}
		}
		avg_ /= count;
	}
}

template<>
void BlockBase<int64_t>::setBlockStatistics(const std::vector<int64_t>& data)
{
	if (isCompressed_)
	{
		return;
	}
	
	if(!isNullable_)
	{
		
		for (int32_t i = 0; i < data.size(); i++)
		{
			updateBlockStatistics(data[i], false);
		}
	}
	else
	{
		size_ += data.size();

		min_ = std::numeric_limits<int64_t>::max();
		max_ = std::numeric_limits<int64_t>::lowest();
		avg_ = 0;
		sum_ = 0;
		size_t count = 0;
		countOfNotNullValues_ = 0;
		isFullOfNullValue_ = true;
		for (int i = 0; i < size_; i++)
		{
			bool isNull = bitMask_[i / (sizeof(int8_t) * 8)] & (1 << (i % (sizeof(int8_t) * 8)));
			if (!isNull)
			{
				countOfNotNullValues_++;

				isFullOfNullValue_ = false;
				min_ = std::min(min_, data_[i]);
				max_ = std::max(max_, data_[i]);
				sum_ += data_[i];
				avg_ += data_[i];
				count++;
			}
		}
		avg_ /= count;
	}
}

template<>
void BlockBase<float>::setBlockStatistics(const std::vector<float>& data)
{
	if (isCompressed_)
	{
		return;
	}
	
	if(!isNullable_)
	{
		for (int32_t i = 0; i < data.size(); i++)
		{
			updateBlockStatistics(data[i], false);
		}
	}
	else
	{
		size_ += data.size();

		min_ = std::numeric_limits<float>::max();
		max_ = std::numeric_limits<float>::lowest();
		avg_ = 0;
		sum_ = 0;
		size_t count = 0;
		countOfNotNullValues_ = 0;
		isFullOfNullValue_ = true;
		for (int i = 0; i < size_; i++)
		{
			bool isNull = bitMask_[i / (sizeof(int8_t) * 8)] & (1 << (i % (sizeof(int8_t) * 8)));
			if (!isNull)
			{
				countOfNotNullValues_++;

				isFullOfNullValue_ = false;
				min_ = std::min(min_, data_[i]);
				max_ = std::max(max_, data_[i]);
				sum_ += data_[i];
				avg_ += data_[i];
				count++;
			}
		}
		avg_ /= count;
	}
}

template<>
void BlockBase<double>::setBlockStatistics(const std::vector<double>& data)
{
	if (isCompressed_)
	{
		return;
	}
	if(!isNullable_)
	{
		for (int32_t i = 0; i < data.size(); i++)
		{
			updateBlockStatistics(data[i], false);
		}
	}
	else
	{
		size_ += data.size();

		min_ = std::numeric_limits<double>::max();
		max_ = std::numeric_limits<double>::lowest();
		avg_ = 0;
		sum_ = 0;
		size_t count = 0;
		countOfNotNullValues_ = 0;
		isFullOfNullValue_ = true;
		for (int i = 0; i < size_; i++)
		{
			bool isNull = bitMask_[i / (sizeof(int8_t) * 8)] & (1 << (i % (sizeof(int8_t) * 8)));
			if (!isNull)
			{
				countOfNotNullValues_++;

				isFullOfNullValue_ = false;
				min_ = std::min(min_, data_[i]);
				max_ = std::max(max_, data_[i]);
				sum_ += data_[i];
				avg_ += data_[i];
				count++;
			}
		}
		avg_ /= count;
	}
}

template<>
void BlockBase<ColmnarDB::Types::Point>::setBlockStatistics(const std::vector<ColmnarDB::Types::Point>& data)
{
	for (int32_t i = 0; i < data.size(); i++)
	{
		updateBlockStatistics(data[i], false);
	}
}

template<>
void BlockBase<ColmnarDB::Types::ComplexPolygon>::setBlockStatistics(const std::vector<ColmnarDB::Types::ComplexPolygon>& data)
{
	for (int32_t i = 0; i < data.size(); i++)
	{
		updateBlockStatistics(data[i], false);
	}
}

template<>
void BlockBase<std::string>::setBlockStatistics(const std::vector<std::string>& data)
{
	for (int32_t i = 0; i < data.size(); i++)
	{
		updateBlockStatistics(data[i], false);
	}
}

template<>
void BlockBase<int8_t>::setBlockStatistics(const std::vector<int8_t>& data)
{
	if (isCompressed_)
	{
		return;
	}

	for (int32_t i = 0; i < data.size(); i++)
	{
		updateBlockStatistics(data[i], false);
	}
}

template<>
void BlockBase<int32_t>::updateBlockStatistics(const int32_t& data, bool isNullValue)
{
	if (isCompressed_)
	{
		return;
	}

	if (isNullValue)
	{
		size_++;
		return;
	}

	if (size_ == 0)
	{
		size_++;
		countOfNotNullValues_++;

		min_ = data;
		max_ = data;
		avg_ = data;
		sum_ = data;
	}
	else
	{
		size_++;
		countOfNotNullValues_++;

		min_ = (data < min_) ? data : min_;
		max_ = (data > max_) ? data : max_;
		sum_ += data;
		avg_ = sum_ / countOfNotNullValues_;
	}
}

template<>
void BlockBase<int64_t>::updateBlockStatistics(const int64_t& data, bool isNullValue)
{
	if (isCompressed_)
	{
		return;
	}

	if (isNullValue)
	{
		size_++;
		return;
	}

	if (size_ == 0)
	{
		size_++;
		countOfNotNullValues_++;

		min_ = data;
		max_ = data;
		avg_ = data;
		sum_ = data;
	}
	else
	{
		size_++;
		countOfNotNullValues_++;

		min_ = (data < min_) ? data : min_;
		max_ = (data > max_) ? data : max_;
		sum_ += data;
		avg_ = sum_ / countOfNotNullValues_;
	}
}

template<>
void BlockBase<float>::updateBlockStatistics(const float& data, bool isNullValue)
{
	if (isCompressed_)
	{
		return;
	}

	if (isNullValue)
	{
		size_++;
		return;
	}

	if (size_ == 0)
	{
		size_++;
		countOfNotNullValues_++;

		min_ = data;
		max_ = data;
		avg_ = data;
		sum_ = data;
	}
	else
	{
		size_++;
		countOfNotNullValues_++;

		min_ = (data < min_) ? data : min_;
		max_ = (data > max_) ? data : max_;
		sum_ += data;
		avg_ = sum_ / countOfNotNullValues_;
	}
}

template<>
void BlockBase<double>::updateBlockStatistics(const double& data, bool isNullValue)
{
	if (isCompressed_)
	{
		return;
	}

	if (isNullValue)
	{
		size_++;
		return;
	}

	if (size_ == 0)
	{
		size_++;
		countOfNotNullValues_++;

		min_ = data;
		max_ = data;
		avg_ = data;
		sum_ = data;
	}
	else
	{
		size_++;
		countOfNotNullValues_++;

		min_ = (data < min_) ? data : min_;
		max_ = (data > max_) ? data : max_;
		sum_ += data;
		avg_ = sum_ / countOfNotNullValues_;
	}
}

template<>
void BlockBase<ColmnarDB::Types::Point>::updateBlockStatistics(const ColmnarDB::Types::Point& data, bool isNullValue)
{
	if (size_ == 0)
	{
		min_ = PointFactory::FromWkt("POINT(0 0)");
		max_ = PointFactory::FromWkt("POINT(0 0)");
		avg_ = (float)0.0;
		sum_ = PointFactory::FromWkt("POINT(0 0)");
	}

	size_++;
}

template<>
void BlockBase<ColmnarDB::Types::ComplexPolygon>::updateBlockStatistics(const ColmnarDB::Types::ComplexPolygon& data, bool isNullValue)
{
	if (size_ == 0)
	{
		min_ = ComplexPolygonFactory::FromWkt("POLYGON((0 0),(0 0))");
		max_ = ComplexPolygonFactory::FromWkt("POLYGON((0 0),(0 0))");
		avg_ = (float)0.0;
		sum_ = ComplexPolygonFactory::FromWkt("POLYGON((0 0),(0 0))");
	}

	size_++;
}

template<>
void BlockBase<std::string>::updateBlockStatistics(const std::string& data, bool isNullValue)
{
	if (isCompressed_)
	{
		return;
	}

	if (isNullValue)
	{
		size_++;
		return;
	}

	if (size_ == 0)
	{
		size_++;
		countOfNotNullValues_++;

		min_ = data;
		max_ = data;
		avg_ = (float)0.0;
		sum_ = "";
	}
	else
	{
		size_++;
		countOfNotNullValues_++;

		min_ = (data < min_) ? data : min_;
		max_ = (data > max_) ? data : max_;
	}
}

template<>
void BlockBase<int8_t>::updateBlockStatistics(const int8_t& data, bool isNullValue)
{
	if (isCompressed_)
	{
		return;
	}

	if (isNullValue)
	{
		size_++;
		return;
	}

	if (size_ == 0)
	{
		size_++;
		countOfNotNullValues_++;

		min_ = data;
		max_ = data;
		avg_ = data;
		sum_ = data;
	}
	else
	{
		size_++;
		countOfNotNullValues_++;

		min_ = (data < min_) ? data : min_;
		max_ = (data > max_) ? data : max_;
		sum_ += data;
		avg_ = sum_ / countOfNotNullValues_;
	}
}