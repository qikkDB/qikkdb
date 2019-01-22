#ifndef I_ENGINE_CORE_H
#define I_ENGINE_CORE_H

#include <memory>

#include "InterfaceCore/IAggregation.h"
#include "InterfaceCore/IArithmetic.h"
#include "InterfaceCore/IArtihmeticConst.h"
#include "InterfaceCore/IFilter.h"
#include "InterfaceCore/IFilterConst.h"
#include "InterfaceCore/IGroupBy.h"
#include "InterfaceCore/ILogic.h"
#include "InterfaceCore/IMemory.h"
#include "InterfaceCore/IPolygon.h"
#include "InterfaceCore/IReconstruct.h"
#include "InterfaceCore/ITypeWidthManip.h"

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
	std::unique_ptr<IAggregation> iAggregation;
	std::unique_ptr<IArithmetic> iArithmetic;
	std::unique_ptr<IArithmeticConst> iArithmeticConst;
	std::unique_ptr<IFilter> iFilter;
	std::unique_ptr<IFilterConst> iFilterConst;
	std::unique_ptr<IGroupBy> iGroupBy;
	std::unique_ptr<ILogic> iLogic;
	std::unique_ptr<IMemory> iMemory;
	std::unique_ptr<IPolygon> iPolygon;
	std::unique_ptr<IReconstruct> iReconstruct;
	std::unique_ptr<ITypeWidthManip> iTypeWidthManip;

public:
	enum Device { CPU, GPU };

	EngineCore(Device device)
	{
		switch (device) {
		case CPU:
			iAggregation = std::make_unique<CPUAggregation>();
			iArithmetic = std::make_unique<CPUArithmetic>();
			iArithmeticConst = std::make_unique<CPUArtihmeticConst>();
			iFilter = std::make_unique<CPUFilter>();
			iFilterConst = std::make_unique<CPUFilterConst>();
			iGroupBy = std::make_unique<CPUGroupBy>();
			iLogic = std::make_unique<CPULogic>();
			iMemory = std::make_unique<CPUMemory>());
			iPolygon = std::make_unique<CPUPolygon>();
			iReconstruct = std::make_unique<CPUReconstruct>();
			iTypeWidthManip = std::make_unique<CPUTypeWidthManip>();
			break;
		case GPU:
			iAggregation = std::make_unique<GPUAggregation>();
			iArithmetic = std::make_unique<GPUArithmetic>();
			iArithmeticConst = std::make_unique<GPUArtihmeticConst>();
			iFilter = std::make_unique<GPUFilter>();
			iFilterConst = std::make_unique<GPUFilterConst>();
			iGroupBy = std::make_unique<GPUGroupBy>();
			iLogic = std::make_unique<GPULogic>();
			iMemory = std::make_unique<GPUMemory>();
			iPolygon = std::make_unique<GPUPolygon>();
			iReconstruct = std::make_unique<GPUReconstruct>();
			iTypeWidthManip = std::make_unique<GPUTypeWidthManip>();
			break;
		}
	};

	~EngineCore() {};

	const IAggregation& getIAggregation() const { return *iAggregation; }
	const IArithmetic& getIArithmetic() const { return *iArithmetic; }
	const IArithmeticConst& getIArithmeticConst() const { return *iArithmeticConst; }
	const IFilter& getIFilter() const { return *iFilter; }
	const IFilterConst& getIFilterConst() const { return *iFilterConst; }
	const IGroupBy& getIGroupBy() const { return *iGroupBy; }
	const ILogic& getILogic() const { return *iLogic; }
	const IMemory& getIMemory() const { return *iMemory; }
	const IPolygon& getIPolygon() const { return *iPolygon; }
	const IReconstruct& getIReconstruct() const { return *iReconstruct; }
};

#endif