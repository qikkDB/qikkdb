#pragma once

#include <functional>
#include <vector>
#include <iostream>
#include <memory>
#include <array>
#include <string>
#include <unordered_map>
#include "MemoryStream.h"
#include "../DataType.h"
#include "JoinType.h"
#include "../Database.h"
#include "../Table.h"
#include "../ColumnBase.h"
#include "ParserExceptions.h"


class GpuSqlJoinDispatcher
{
public:
    GpuSqlJoinDispatcher(const std::shared_ptr<Database>& database);

    void AddJoinFunction(DataType type, std::string op);

    void AddJoinDoneFunction();

    template <typename T>
    void AddArgument(T argument)
    {
        arguments_.Insert<T>(argument);
    }

    std::unordered_map<std::string, std::vector<std::vector<int32_t>>>* GetJoinIndices();

    void Execute();

private:
    typedef int32_t (GpuSqlJoinDispatcher::*DispatchJoinFunction)();
    std::vector<DispatchJoinFunction> dispatcherFunctions_;
    MemoryStream arguments_;
    const std::shared_ptr<Database>& database_;
    int32_t instructionPointer_;
    std::unordered_map<std::string, std::vector<std::vector<int32_t>>> joinIndices_;

    static std::array<DispatchJoinFunction, DataType::DATA_TYPE_SIZE> joinEqualFunctions_;
    static std::array<DispatchJoinFunction, DataType::DATA_TYPE_SIZE> joinGreaterFunctions_;
    static std::array<DispatchJoinFunction, DataType::DATA_TYPE_SIZE> joinLessFunctions_;
    static std::array<DispatchJoinFunction, DataType::DATA_TYPE_SIZE> joinGreaterEqualFunctions_;
    static std::array<DispatchJoinFunction, DataType::DATA_TYPE_SIZE> joinLessEqualFunctions_;
    static std::array<DispatchJoinFunction, DataType::DATA_TYPE_SIZE> joinNotEqualFunctions_;
    static DispatchJoinFunction joinDoneFunction_;

    template <typename OP, typename T>
    int32_t JoinCol();

    template <typename OP, typename T>
    int32_t JoinConst();

    int32_t JoinDone();

    template <typename OP, typename T>
    int32_t InvalidOperandTypesErrorHandlerCol()
    {
        auto colName = arguments_.Read<std::string>();

        throw InvalidOperandsException(colName, std::string(""), std::string(typeid(OP).name()));
    }

    template <typename OP, typename T>
    int32_t InvalidOperandTypesErrorHandlerConst()
    {
        T cnst = arguments_.Read<T>();

        throw InvalidOperandsException(std::string(""), std::string("cnst"), std::string(typeid(OP).name()));
    }

    std::pair<std::string, std::string> SplitColumnName(const std::string& colName);
};