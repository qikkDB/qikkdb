#ifndef I_ENGINE_CORE_H
#define I_ENGINE_CORE_H

#include <memory>

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

class EngineCore {
private:
	GPUAggregation gpuAggregation_;
	GPUArithmetic gpuArithmetic_;
	GPUArithmeticConst gpuArithmeticConst_;
	GPUFilter gpuFilter_;
	GPUFilterConst gpuFilterConst_;
	GPUGroupBy gpuGroupBy_;
	GPULogic gpuLogic_;
	GPUMemory gpuMemory_;
	GPUPolygon gpuPolygon_;
	GPUReconstruct gpuReconstruct_;
	GPUTypeWidthManip gpuTypeWidthManip_;

public:
	const GPUAggregation& getGPUAggregation() const { return gpuAggregation_; }
	const GPUArithmetic& getGPUArithmetic() const { return gpuArithmetic_; }
	const GPUArithmeticConst& getGPUArithmeticConst() const { return gpuArithmeticConst_; }
	const GPUFilter& getGPUFilter() const { return gpuFilter_; }
	const GPUFilterConst& getGPUFilterConst() const { return gpuFilterConst_; }
	const GPUGroupBy& getGPUGroupBy() const { return gpuGroupBy_; }
	const GPULogic& getGPULogic() const { return gpuLogic_; }
	const GPUMemory& getGPUMemory() const { return gpuMemory_; }
	const GPUPolygon& getGPUPolygon() const { return gpuPolygon_; }
	const GPUReconstruct& getGPUReconstruct() const { return gpuReconstruct_; }
};

#endif