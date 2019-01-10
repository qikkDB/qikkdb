#pragma once
#include "IBlock.h"

template<class T>
class ColumnBase;

template<class T>
class BlockBase :
	public IBlock<T>
{
private:
	std::vector<T> data_;
	ColumnBase<T>& column_;
public:
	BlockBase(const std::vector<T>& data, const ColumnBase<T>& column);
	BlockBase(const ColumnBase<T>& column);

	virtual std::vector<T> GetData() const override;
	virtual int EmptyBlockSpace() const override;
	virtual bool IsFull() const override;
	virtual void InsertData(const std::vector<T>& data) override;
};

