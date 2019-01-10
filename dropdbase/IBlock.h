#pragma once
#include <vector>
template <class T>
class IBlock
{
public:
	virtual std::vector<T> GetData() const = 0;

	virtual int EmptyBlockSpace() const = 0;

	virtual bool IsFull() const = 0;

	virtual void InsertData(const std::vector<T>& data) = 0;

	virtual ~IBlock() {};
};

