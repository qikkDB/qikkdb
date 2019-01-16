#ifndef I_ENGINE_CORE_H
#define I_ENGINE_CORE_H

#include <memory>

#include "IAggregation.h"
#include "IArithmetic.h"
#include "IArtihmeticConst.h"
#include "IFilter.h"
#include "IFilterConst.h"
#include "IGroupBy.h"
#include "ILogic.h"
#include "IMemory.h"
#include "IPolygon.h"
#include "IReconstruct.h"

class IEngineCore {
public:
	virtual ~IEngineCore();

	virtual const std::shared_ptr<IAggregation>& getIAggregation() = 0;

	virtual const std::shared_ptr<IArithmetic>& getIArithmetic() = 0;

	virtual const std::shared_ptr<IArithmeticConst>& getIArithmeticConst() = 0;

	virtual const std::shared_ptr<IFilter>& getIFilter() = 0;

	virtual const std::shared_ptr<IFilterConst>& getIFilterConst() = 0;

	virtual const std::shared_ptr<IGroupBy>& getIGroupBy() = 0;

	virtual const std::shared_ptr<ILogic>& getILogic() = 0;

	virtual const std::shared_ptr<IMemory>& getIMemory() = 0;

	virtual const std::shared_ptr<IPolygon>& getIPolygon() = 0;

	virtual const std::shared_ptr<IReconstruct>& getIReconstruct() = 0;
};

#endif