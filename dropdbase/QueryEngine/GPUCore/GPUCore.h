#ifndef GPU_CORE_H
#define GPU_CORE_H

#include "../IEngineCore.h"

class GPUCore : public IEngineCore {
private:
	std::shared_ptr<IAggregation> iAggregation;
	std::shared_ptr<IArithmetic> iArithmetic;
	std::shared_ptr<IArithmeticConst> iArithmeticConst;
	std::shared_ptr<IFilter> iFilter;
	std::shared_ptr<IFilterConst> iFilterConst;
	std::shared_ptr<IGroupBy> iGroupBy;
	std::shared_ptr<ILogic> iLogic;
	std::shared_ptr<IMemory> iMemory;
	std::shared_ptr<IPolygon> iPolygon;
	std::shared_ptr<IReconstruct> iReconstruct;

public:
	GPUCore() :
		iAggregation(),
		iArithmetic(),
		iArithmeticConst(),
		iFilter(),
		iFilterConst(),
		iGroupBy(),
		iLogic(),
		iMemory(),
		iPolygon(),
		iReconstruct() {





	};

	virtual ~GPUCore() {};

	virtual const std::shared_ptr<IAggregation>& getIAggregation() const
	{

	}

	virtual const std::shared_ptr<IArithmetic>& getIArithmetic() const
	{

	}
	virtual const std::shared_ptr<IArithmeticConst>& getIArithmeticConst() const
	{

	}

	virtual const std::shared_ptr<IFilter>& getIFilter() const
	{

	}
	virtual const std::shared_ptr<IFilterConst>& getIFilterConst() const
	{

	}

	virtual const std::shared_ptr<IGroupBy>& getIGroupBy() const
	{

	}

	virtual const std::shared_ptr<ILogic>& getILogic() const
	{

	}

	virtual const std::shared_ptr<IMemory>& getIMemory() const
	{

	}

	virtual const std::shared_ptr<IPolygon>& getIPolygon() const
	{

	}

	virtual const std::shared_ptr<IReconstruct>& getIReconstruct() const
	{

	}
};

#endif

