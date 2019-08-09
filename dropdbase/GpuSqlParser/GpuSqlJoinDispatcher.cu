#include "GpuSqlJoinDispatcher.h"
#include "../QueryEngine/GPUCore/GPUFilterConditions.cuh"

GpuSqlJoinDispatcher::GpuSqlJoinDispatcher(const std::shared_ptr<Database>& database)
: database_(database), instructionPointer_(0)
{
}

void GpuSqlJoinDispatcher::Execute()
{
    int32_t err = 0;

    while (err == 0)
    {
        err = (this->*dispatcherFunctions_[instructionPointer_++])();

        if (err)
        {
            if (err != 1)
            {
                CudaLogBoost::getInstance(CudaLogBoost::info)
                    << "Error occured while producing join indices." << '\n';
            }
            break;
        }
    }
}

void GpuSqlJoinDispatcher::AddJoinFunction(DataType type, std::string op)
{
    if (op == "=")
    {
        dispatcherFunctions_.push_back(joinEqualFunctions_[type]);
    }
    else if (op == ">")
    {
        dispatcherFunctions_.push_back(joinGreaterFunctions_[type]);
    }
    else if (op == "<")
    {
        dispatcherFunctions_.push_back(joinLessFunctions_[type]);
    }
    else if (op == ">=")
    {
        dispatcherFunctions_.push_back(joinGreaterEqualFunctions_[type]);
    }
    else if (op == "<=")
    {
        dispatcherFunctions_.push_back(joinLessEqualFunctions_[type]);
    }
    else if (op == "=" || op == "<>")
    {
        dispatcherFunctions_.push_back(joinNotEqualFunctions_[type]);
    }
}

void GpuSqlJoinDispatcher::AddJoinDoneFunction()
{
    dispatcherFunctions_.push_back(joinDoneFunction_);
}

std::unordered_map<std::string, std::vector<std::vector<int32_t>>>* GpuSqlJoinDispatcher::GetJoinIndices()
{
    return &joinIndices_;
}

int32_t GpuSqlJoinDispatcher::JoinDone()
{
    CudaLogBoost::getInstance(CudaLogBoost::info) << "Join Done." << '\n';
    return 1;
}

std::pair<std::string, std::string> GpuSqlJoinDispatcher::SplitColumnName(const std::string& colName)
{
    const size_t splitIdx = colName.find(".");
    const std::string table = colName.substr(0, splitIdx);
    const std::string column = colName.substr(splitIdx + 1);
    return {table, column};
}
