#pragma once

#include <memory>
#include <array>
#include <tuple>
#include <string>
#include <unordered_map>
#include "MemoryStream.h"
#include "../DataType.h"
#include "../Database.h"
#include "../BlockBase.h"
#include "../ColumnBase.h"
#include "ParserExceptions.h"

class CpuSqlDispatcher
{
private:
    typedef int32_t (CpuSqlDispatcher::*CpuDispatchFunction)();
    std::vector<CpuDispatchFunction> cpuDispatcherFunctions_;
    const std::shared_ptr<Database>& database_;
    int32_t blockIndex_;
    int64_t whereResult_;
    MemoryStream arguments_;
    int32_t instructionPointer_;

    std::unordered_map<std::string, std::tuple<std::uintptr_t, int32_t, bool>> allocatedPointers_;
    bool IsRegisterAllocated(std::string& reg);
    std::pair<std::string, std::string> SplitColumnName(const std::string& name);
    std::pair<std::string, std::string> GetPointerNames(const std::string& colName);

    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterEqualFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessEqualFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> equalFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> notEqualFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalAndFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalOrFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> mulFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> divFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> addFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> subFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> modFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseOrFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseAndFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseXorFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseLeftShiftFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> bitwiseRightShiftFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logarithmFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> arctangent2Functions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> roundDecimalFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> powerFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> rootFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> pointFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> containsFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> intersectFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> unionFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> leftFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> rightFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> concatFunctions_;

    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> dateToStringFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> yearFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> monthFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> dayFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> hourFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> minuteFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> secondFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> weekdayFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> dayOfWeekFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> logicalNotFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> minusFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> absoluteFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> sineFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> cosineFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> tangentFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> cotangentFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> arcsineFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> arccosineFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> arctangentFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> logarithm10Functions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> logarithmNaturalFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> exponentialFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> squareRootFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> squareFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> signFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> roundFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> ceilFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> floorFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> ltrimFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> rtrimFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> lowerFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> upperFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> reverseFunctions_;
    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> lenFunctions_;
    static CpuDispatchFunction nullFunction;
    static std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> castToIntFunctions_;
    static std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> castToLongFunctions_;
    // static std::array<CpuSqlDispatcher::CpuDispatchFunction,
    //		DataType::DATA_TYPE_SIZE> castToDateFunctions;
    static std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> castToFloatFunctions_;
    static std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> castToDoubleFunctions_;
    static std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> castToStringFunctions_;
    static std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> castToPointFunctions_;
    static std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> castToPolygonFunctions_;
    static std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> castToInt8TFunctions_;

    static std::array<CpuDispatchFunction, DataType::DATA_TYPE_SIZE> whereResultFunctions_;

public:
    CpuSqlDispatcher(const std::shared_ptr<Database>& database);
    void AddBinaryOperation(DataType left, DataType right, size_t opType);
    void AddUnaryOperation(DataType type, size_t opType);
    void AddCastOperation(DataType inputType, DataType outputType, const std::string& outTypeStr);
    void AddWhereResultFunction(DataType dataType);
    int64_t Execute(int32_t index);
    void CopyExecutionDataTo(CpuSqlDispatcher& other);

    template <typename T>
    T* AllocateRegister(const std::string& reg, int32_t size, bool resultColColOperation)
    {
        void* allocatedMemory = operator new(size * sizeof(T));
        allocatedPointers_.insert({reg, std::make_tuple(reinterpret_cast<std::uintptr_t>(allocatedMemory),
                                                        size, resultColColOperation)});
        return reinterpret_cast<T*>(allocatedMemory);
    }

    template <typename T>
    T GetBlockMin(const std::string& tableName, const std::string& columnName)
    {
        auto col = dynamic_cast<const ColumnBase<T>*>(
            database_->GetTables().at(tableName).GetColumns().find(columnName)->second.get());
        auto block = dynamic_cast<BlockBase<T>*>(col->GetBlocksList()[blockIndex_]);

        return block->GetMin();
    }

    template <typename T>
    T GetBlockMax(const std::string& tableName, const std::string& columnName)
    {
        auto col = dynamic_cast<const ColumnBase<T>*>(
            database_->GetTables().at(tableName).GetColumns().find(columnName)->second.get());
        auto block = dynamic_cast<BlockBase<T>*>(col->GetBlocksList()[blockIndex_]);

        return block->GetMax();
    }

    ~CpuSqlDispatcher()
    {
        for (auto& pointer : allocatedPointers_)
        {
            operator delete(reinterpret_cast<void*>(std::get<0>(pointer.second)));
        }

        allocatedPointers_.clear();
    }

    template <typename T>
    int32_t LoadCol(std::string& colName)
    {
        if (allocatedPointers_.find(colName) == allocatedPointers_.end() && !colName.empty() &&
            colName.front() != '$')
        {
            std::string tableName;
            std::string columnName;

            std::tie(tableName, columnName) = SplitColumnName(colName);
            if (blockIndex_ >=
                database_->GetTables().at(tableName).GetColumns().at(columnName).get()->GetBlockCount())
            {
                return 1;
            }

            std::string reg_min = colName + "_min";
            std::string reg_max = colName + "_max";
            T* mask_min = AllocateRegister<T>(reg_min, 1, false);
            T* mask_max = AllocateRegister<T>(reg_max, 1, false);
            mask_min[0] = GetBlockMin<T>(tableName, columnName);
            mask_max[0] = GetBlockMax<T>(tableName, columnName);
        }
        return 0;
    }

    template <typename OP, typename T, typename U>
    int32_t FilterColConst();

    template <typename OP, typename T, typename U>
    int32_t FilterConstCol();

    template <typename OP, typename T, typename U>
    int32_t filterColCol();

    template <typename OP, typename T, typename U>
    int32_t filterConstConst();

    template <typename OP>
    int32_t FilterStringColConst();

    template <typename OP>
    int32_t FilterStringConstCol();

    template <typename OP>
    int32_t FilterStringColCol();

    template <typename OP>
    int32_t FilterStringConstConst();

    template <typename OP, typename T, typename U>
    int32_t LogicalColConst();

    template <typename OP, typename T, typename U>
    int32_t LogicalConstCol();

    template <typename OP, typename T, typename U>
    int32_t LogicalColCol();

    template <typename OP, typename T, typename U>
    int32_t LogicalConstConst();

    template <typename T>
    int32_t LogicalNotCol();

    template <typename T>
    int32_t LogicalNotConst();

    int32_t NullCol();

    template <typename OP, typename T, typename U>
    int32_t ArithmeticColConst();

    template <typename OP, typename T, typename U>
    int32_t arithmeticConstCol();

    template <typename OP, typename T, typename U>
    int32_t arithmeticColCol();

    template <typename OP, typename T, typename U>
    int32_t arithmeticConstConst();

    int32_t DateToStringCol();

    int32_t DateToStringConst();

    template <typename OP>
    int32_t DateExtractCol();

    template <typename OP>
    int32_t DateExtractConst();

    template <typename OP, typename T>
    int32_t ArithmeticUnaryCol();

    template <typename OP, typename T>
    int32_t ArithmeticUnaryConst();

    template <typename T, typename U>
    int32_t PointColCol();

    template <typename T, typename U>
    int32_t PointColConst();

    template <typename T, typename U>
    int32_t PointConstCol();

    template <typename T, typename U>
    int32_t ContainsColConst();

    template <typename T, typename U>
    int32_t ContainsConstCol();

    template <typename T, typename U>
    int32_t ContainsColCol();

    template <typename T, typename U>
    int32_t ContainsConstConst();

    template <typename OP, typename T, typename U>
    int32_t PolygonOperationColConst();

    template <typename OP, typename T, typename U>
    int32_t PolygonOperationConstCol();

    template <typename OP, typename T, typename U>
    int32_t PolygonOperationColCol();

    template <typename OP, typename T, typename U>
    int32_t PolygonOperationConstConst();

    template <typename OP>
    int32_t StringUnaryCol();

    template <typename OP>
    int32_t StringUnaryConst();

    template <typename OP>
    int32_t StringUnaryNumericCol();

    template <typename OP>
    int32_t StringUnaryNumericConst();

    template <typename OP, typename T>
    int32_t StringBinaryNumericColCol();

    template <typename OP, typename T>
    int32_t StringBinaryNumericColConst();

    template <typename OP, typename T>
    int32_t StringBinaryNumericConstCol();

    template <typename OP, typename T>
    int32_t StringBinaryNumericConstConst();

    template <typename OP>
    int32_t StringBinaryColCol();

    template <typename OP>
    int32_t StringBinaryColConst();

    template <typename OP>
    int32_t StringBinaryConstCol();

    template <typename OP>
    int32_t StringBinaryConstConst();
    template <typename OUT, typename IN>
    int32_t CastNumericCol();

    template <typename OUT, typename IN>
    int32_t CastNumericConst();

    template <typename T>
    int32_t WhereResultCol()
    {
        auto colName = arguments_.Read<std::string>();
        auto regMin = allocatedPointers_.at(colName + "_min");
        auto regMax = allocatedPointers_.at(colName + "_max");
        T* resultMin = reinterpret_cast<T*>(std::get<0>(regMin));
        T* resultMax = reinterpret_cast<T*>(std::get<0>(regMax));

        int64_t whereResultMin = std::get<2>(regMin) ? 1 : static_cast<int64_t>(resultMin[0]);
        int64_t whereResultMax = std::get<2>(regMax) ? 1 : static_cast<int64_t>(resultMax[0]);

        whereResult_ = whereResultMin || whereResultMax;

        CudaLogBoost::getInstance(CudaLogBoost::info)
            << "Where result: " << colName << ", " << whereResult_ << '\n';

        return 1;
    }

    template <typename T>
    int32_t WhereResultConst()
    {
        T cnst = arguments_.Read<T>();
        whereResult_ = static_cast<int64_t>(cnst);

        CudaLogBoost::getInstance(CudaLogBoost::info) << "Where result const: " << whereResult_ << '\n';

        return 1;
    }


    template <typename OP, typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerConstCol()
    {
        T cnst = arguments_.Read<T>();
        auto colName = arguments_.Read<std::string>();

        throw InvalidOperandsException(colName, std::string("cnst"), std::string(typeid(OP).name()));
    }

    template <typename OP, typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerColConst()
    {
        auto colName = arguments_.Read<std::string>();
        U cnst = arguments_.Read<U>();

        throw InvalidOperandsException(colName, std::string("cnst"), std::string(typeid(OP).name()));
    }

    template <typename OP, typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerConstConst()
    {
        T cnstLeft = arguments_.Read<T>();
        U cnstRight = arguments_.Read<U>();

        throw InvalidOperandsException(std::string("cnst"), std::string("cnst"),
                                       std::string(typeid(OP).name()));
    }

    template <typename OP, typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerColCol()
    {
        auto colNameLeft = arguments_.Read<std::string>();
        auto colNameRight = arguments_.Read<std::string>();

        throw InvalidOperandsException(colNameLeft, colNameRight, std::string(typeid(OP).name()));
    }

    template <typename T>
    int32_t InvalidOperandTypesErrorHandlerCol()
    {
        auto colName = arguments_.Read<std::string>();

        throw InvalidOperandsException(colName, std::string(""), std::string("operation"));
    }

    template <typename T>
    int32_t InvalidOperandTypesErrorHandlerConst()
    {
        T cnst = arguments_.Read<T>();

        throw InvalidOperandsException(std::string(""), std::string("cnst"), std::string("operation"));
    }

    template <typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerColConst()
    {
        U cnst = arguments_.Read<U>();
        auto colName = arguments_.Read<std::string>();

        throw InvalidOperandsException(colName, std::string("cnst"), std::string("operation"));
    }

    template <typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerConstCol()
    {
        auto colName = arguments_.Read<std::string>();
        T cnst = arguments_.Read<T>();

        throw InvalidOperandsException(colName, std::string("cnst"), std::string("operation"));
    }

    template <typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerColCol()
    {
        auto colNameRight = arguments_.Read<std::string>();
        auto colNameLeft = arguments_.Read<std::string>();

        throw InvalidOperandsException(colNameLeft, colNameRight, std::string("operation"));
    }

    template <typename T, typename U>
    int32_t InvalidOperandTypesErrorHandlerConstConst()
    {
        U cnstRight = arguments_.Read<U>();
        T cnstLeft = arguments_.Read<T>();

        throw InvalidOperandsException(std::string("cnst"), std::string("cnst"), std::string("operation"));
    }

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

    template <typename T>
    void AddArgument(T argument)
    {
        arguments_.Insert<T>(argument);
    }
};

template <>
int32_t CpuSqlDispatcher::LoadCol<std::string>(std::string& colName);