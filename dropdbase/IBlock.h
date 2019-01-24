#pragma once
#include <vector>
template <class T>
class IBlock
{
public:
	//TODO delete IBlock at all and refactor dependend code !!!!!!!!!!!!!!!!!!!!

	virtual std::vector<T>& GetData() = 0;

	virtual int EmptyBlockSpace() const = 0;

	virtual bool IsFull() const = 0;

	virtual void InsertData(const std::vector<T>& data) = 0;

	virtual ~IBlock() {};
};

