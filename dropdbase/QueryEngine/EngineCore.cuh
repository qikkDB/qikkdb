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
#include "ITypeWidthManip.h"

#include "GPUCore/GPUAggregation.cuh"
#include "GPUCore/GPUArithmetic.cuh"
#include "GPUCore/GPUArtihmeticConst.cuh"
#include "GPUCore/GPUFilter.cuh"
#include "GPUCore/GPUFilterConst.cuh"
#include "GPUCore/GPUGroupBy.cuh"
#include "GPUCore/GPULogic.cuh"
#include "GPUCore/GPUMemory.cuh"
#include "GPUCore/GPUPolygon.cuh"
#include "GPUCore/GPUReconstruct.cuh"
#include "GPUCore/GPUTypeWidthManip.cuh"

#include "CPUCore/CPUAggregation.h"
#include "CPUCore/CPUArithmetic.h"
#include "CPUCore/CPUArtihmeticConst.h"
#include "CPUCore/CPUFilter.h"
#include "CPUCore/CPUFilterConst.h"
#include "CPUCore/CPUGroupBy.h"
#include "CPUCore/CPULogic.h"
#include "CPUCore/CPUMemory.h"
#include "CPUCore/CPUPolygon.h"
#include "CPUCore/CPUReconstruct.h"
#include "CPUCore/CPUTypeWidthManip.h"

class EngineCore {
public:
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
	std::shared_ptr<ITypeWidthManip> iTypeWidthManip;

public:
	enum Device { CPU, GPU };

	EngineCore(Device device)
	{
		switch (device) {
		case CPU:
			iAggregation = std::shared_ptr<IAggregation>(new CPUAggregation());
			iArithmetic = std::shared_ptr<IArithmetic>(new CPUArithmetic());
			iArithmeticConst = std::shared_ptr<IArithmeticConst>(new CPUArtihmeticConst());
			iFilter = std::shared_ptr<IFilter>(new CPUFilter());
			iFilterConst = std::shared_ptr<IFilterConst>(new CPUFilterConst());
			iGroupBy = std::shared_ptr<IGroupBy>(new CPUGroupBy());
			iLogic = std::shared_ptr<ILogic>(new CPULogic());
			iMemory = std::shared_ptr<IMemory>(new CPUMemory());
			iPolygon = std::shared_ptr<IPolygon>(new CPUPolygon());
			iReconstruct = std::shared_ptr<IReconstruct>(new CPUReconstruct());
			iTypeWidthManip = std::shared_ptr<ITypeWidthManip>(new CPUTypeWidthManip());
			break;
		case GPU:
			iAggregation = std::shared_ptr<IAggregation>(new GPUAggregation());
			iArithmetic = std::shared_ptr<IArithmetic>(new GPUArithmetic());
			iArithmeticConst = std::shared_ptr<IArithmeticConst>(new GPUArtihmeticConst());
			iFilter = std::shared_ptr<IFilter>(new GPUFilter());
			iFilterConst = std::shared_ptr<IFilterConst>(new GPUFilterConst());
			iGroupBy = std::shared_ptr<IGroupBy>(new GPUGroupBy());
			iLogic = std::shared_ptr<ILogic>(new GPULogic());
			iMemory = std::shared_ptr<IMemory>(new GPUMemory());
			iPolygon = std::shared_ptr<IPolygon>(new GPUPolygon());
			iReconstruct = std::shared_ptr<IReconstruct>(new GPUReconstruct());
			iTypeWidthManip = std::shared_ptr<ITypeWidthManip>(new GPUTypeWidthManip());
			break;
		}
	};

	~EngineCore() {};

	const std::shared_ptr<IAggregation>& getIAggregation() const { return iAggregation; }
	const std::shared_ptr<IArithmetic>& getIArithmetic() const { return iArithmetic; }
	const std::shared_ptr<IArithmeticConst>& getIArithmeticConst() const { return iArithmeticConst; }
	const std::shared_ptr<IFilter>& getIFilter() const { return iFilter; }
	const std::shared_ptr<IFilterConst>& getIFilterConst() const { return iFilterConst; }
	const std::shared_ptr<IGroupBy>& getIGroupBy() const { return iGroupBy; }
	const std::shared_ptr<ILogic>& getILogic() const { return iLogic; }
	const std::shared_ptr<IMemory>& getIMemory() const { return iMemory; }
	const std::shared_ptr<IPolygon>& getIPolygon() const { return iPolygon; }
	const std::shared_ptr<IReconstruct>& getIReconstruct() const { return iReconstruct; }
};

#endif