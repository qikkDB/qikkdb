#include "CpuSqlDispatcher.h"
#include "GpuSqlLexer.h"

std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::whereResultFunctions_ = {
    &CpuSqlDispatcher::WhereResultConst<int32_t>,
    &CpuSqlDispatcher::WhereResultConst<int64_t>,
    &CpuSqlDispatcher::WhereResultConst<float>,
    &CpuSqlDispatcher::WhereResultConst<double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerConst<std::string>,
    &CpuSqlDispatcher::WhereResultConst<int8_t>,
    &CpuSqlDispatcher::WhereResultCol<int32_t>,
    &CpuSqlDispatcher::WhereResultCol<int64_t>,
    &CpuSqlDispatcher::WhereResultCol<float>,
    &CpuSqlDispatcher::WhereResultCol<double>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::InvalidOperandTypesErrorHandlerCol<std::string>,
    &CpuSqlDispatcher::WhereResultCol<int8_t>};


CpuSqlDispatcher::CpuSqlDispatcher(const std::shared_ptr<Database>& database)
: database_(database), blockIndex_(0), instructionPointer_(0), whereResult_(1)
{
}

bool CpuSqlDispatcher::IsRegisterAllocated(std::string& reg)
{
    return allocatedPointers_.find(reg) != allocatedPointers_.end();
}

std::pair<std::string, std::string> CpuSqlDispatcher::SplitColumnName(const std::string& name)
{
    const size_t separatorPosition = name.find(".");
    const std::string table = name.substr(0, separatorPosition);
    const std::string column = name.substr(separatorPosition + 1);

    return std::make_pair(table, column);
}

void CpuSqlDispatcher::AddBinaryOperation(DataType left, DataType right, size_t opType)
{
    switch (opType)
    {
    case GpuSqlLexer::GREATER:
        cpuDispatcherFunctions_.push_back(greaterFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::LESS:
        cpuDispatcherFunctions_.push_back(lessFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::GREATEREQ:
        cpuDispatcherFunctions_.push_back(greaterEqualFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::LESSEQ:
        cpuDispatcherFunctions_.push_back(lessEqualFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::EQUALS:
        cpuDispatcherFunctions_.push_back(equalFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::NOTEQUALS:
    case GpuSqlLexer::NOTEQUALS_GT_LT:
        cpuDispatcherFunctions_.push_back(notEqualFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::AND:
        cpuDispatcherFunctions_.push_back(logicalAndFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::OR:
        cpuDispatcherFunctions_.push_back(logicalOrFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::ASTERISK:
        cpuDispatcherFunctions_.push_back(mulFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::DIVISION:
        cpuDispatcherFunctions_.push_back(divFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::PLUS:
        cpuDispatcherFunctions_.push_back(addFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::MINUS:
        cpuDispatcherFunctions_.push_back(subFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::MODULO:
        cpuDispatcherFunctions_.push_back(modFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::BIT_OR:
        cpuDispatcherFunctions_.push_back(bitwiseOrFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::BIT_AND:
        cpuDispatcherFunctions_.push_back(bitwiseAndFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::XOR:
        cpuDispatcherFunctions_.push_back(bitwiseXorFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::L_SHIFT:
        cpuDispatcherFunctions_.push_back(bitwiseLeftShiftFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::R_SHIFT:
        cpuDispatcherFunctions_.push_back(bitwiseRightShiftFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::POINT:
        cpuDispatcherFunctions_.push_back(pointFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::GEO_CONTAINS:
        cpuDispatcherFunctions_.push_back(containsFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::GEO_INTERSECT:
        cpuDispatcherFunctions_.push_back(intersectFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::GEO_UNION:
        cpuDispatcherFunctions_.push_back(unionFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::LOG:
        cpuDispatcherFunctions_.push_back(logarithmFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::POW:
        cpuDispatcherFunctions_.push_back(powerFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::ROOT:
        cpuDispatcherFunctions_.push_back(rootFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::ATAN2:
        cpuDispatcherFunctions_.push_back(arctangent2Functions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::LEFT:
        cpuDispatcherFunctions_.push_back(leftFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::RIGHT:
        cpuDispatcherFunctions_.push_back(rightFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::CONCAT:
        cpuDispatcherFunctions_.push_back(concatFunctions_[left * DataType::DATA_TYPE_SIZE + right]);
        break;
    default:
        break;
    }
}

void CpuSqlDispatcher::AddUnaryOperation(DataType type, size_t opType)
{
    switch (opType)
    {
    case GpuSqlLexer::LOGICAL_NOT:
        cpuDispatcherFunctions_.push_back(logicalNotFunctions_[type]);
        break;
    case GpuSqlLexer::ISNULL:
        cpuDispatcherFunctions_.push_back(nullFunction);
        break;
    case GpuSqlLexer::ISNOTNULL:
        cpuDispatcherFunctions_.push_back(nullFunction);
        break;
    case GpuSqlLexer::MINUS:
        cpuDispatcherFunctions_.push_back(minusFunctions_[type]);
        break;
    case GpuSqlLexer::YEAR:
        cpuDispatcherFunctions_.push_back(yearFunctions_[type]);
        break;
    case GpuSqlLexer::MONTH:
        cpuDispatcherFunctions_.push_back(monthFunctions_[type]);
        break;
    case GpuSqlLexer::DAY:
        cpuDispatcherFunctions_.push_back(dayFunctions_[type]);
        break;
    case GpuSqlLexer::HOUR:
        cpuDispatcherFunctions_.push_back(hourFunctions_[type]);
        break;
    case GpuSqlLexer::MINUTE:
        cpuDispatcherFunctions_.push_back(minuteFunctions_[type]);
        break;
    case GpuSqlLexer::SECOND:
        cpuDispatcherFunctions_.push_back(secondFunctions_[type]);
        break;
    case GpuSqlLexer::ABS:
        cpuDispatcherFunctions_.push_back(absoluteFunctions_[type]);
        break;
    case GpuSqlLexer::SIN:
        cpuDispatcherFunctions_.push_back(sineFunctions_[type]);
        break;
    case GpuSqlLexer::COS:
        cpuDispatcherFunctions_.push_back(cosineFunctions_[type]);
        break;
    case GpuSqlLexer::TAN:
        cpuDispatcherFunctions_.push_back(tangentFunctions_[type]);
        break;
    case GpuSqlLexer::COT:
        cpuDispatcherFunctions_.push_back(cotangentFunctions_[type]);
        break;
    case GpuSqlLexer::ASIN:
        cpuDispatcherFunctions_.push_back(arcsineFunctions_[type]);
        break;
    case GpuSqlLexer::ACOS:
        cpuDispatcherFunctions_.push_back(arccosineFunctions_[type]);
        break;
    case GpuSqlLexer::ATAN:
        cpuDispatcherFunctions_.push_back(arctangentFunctions_[type]);
        break;
    case GpuSqlLexer::LOG10:
        cpuDispatcherFunctions_.push_back(logarithm10Functions_[type]);
        break;
    case GpuSqlLexer::LOG:
        cpuDispatcherFunctions_.push_back(logarithmNaturalFunctions_[type]);
        break;
    case GpuSqlLexer::EXP:
        cpuDispatcherFunctions_.push_back(exponentialFunctions_[type]);
        break;
    case GpuSqlLexer::SQRT:
        cpuDispatcherFunctions_.push_back(squareRootFunctions_[type]);
        break;
    case GpuSqlLexer::SQUARE:
        cpuDispatcherFunctions_.push_back(squareFunctions_[type]);
        break;
    case GpuSqlLexer::SIGN:
        cpuDispatcherFunctions_.push_back(signFunctions_[type]);
        break;
    case GpuSqlLexer::ROUND:
        cpuDispatcherFunctions_.push_back(roundFunctions_[type]);
        break;
    case GpuSqlLexer::FLOOR:
        cpuDispatcherFunctions_.push_back(floorFunctions_[type]);
        break;
    case GpuSqlLexer::CEIL:
        cpuDispatcherFunctions_.push_back(ceilFunctions_[type]);
        break;
    case GpuSqlLexer::LTRIM:
        cpuDispatcherFunctions_.push_back(ltrimFunctions_[type]);
        break;
    case GpuSqlLexer::RTRIM:
        cpuDispatcherFunctions_.push_back(rtrimFunctions_[type]);
        break;
    case GpuSqlLexer::LOWER:
        cpuDispatcherFunctions_.push_back(lowerFunctions_[type]);
        break;
    case GpuSqlLexer::UPPER:
        cpuDispatcherFunctions_.push_back(upperFunctions_[type]);
        break;
    case GpuSqlLexer::LEN:
        cpuDispatcherFunctions_.push_back(lenFunctions_[type]);
        break;

    default:
        break;
    }
}

void CpuSqlDispatcher::AddCastOperation(DataType inputType, DataType outputType, const std::string& outTypeStr)
{
    switch (outputType)
    {
    case COLUMN_INT:
        cpuDispatcherFunctions_.push_back(castToIntFunctions_[inputType]);
        break;
    case COLUMN_LONG:
        if (outTypeStr == "DATE")
        {
            // dispatcher_.AddCastToDateFunction(operandType);
        }
        else
        {
            cpuDispatcherFunctions_.push_back(castToLongFunctions_[inputType]);
        }
        break;
    case COLUMN_FLOAT:
        cpuDispatcherFunctions_.push_back(castToFloatFunctions_[inputType]);
        break;
    case COLUMN_DOUBLE:
        cpuDispatcherFunctions_.push_back(castToDoubleFunctions_[inputType]);
        break;
    case COLUMN_STRING:
        cpuDispatcherFunctions_.push_back(castToStringFunctions_[inputType]);
        break;
    case COLUMN_POINT:
        cpuDispatcherFunctions_.push_back(castToPointFunctions_[inputType]);
        break;
    case COLUMN_POLYGON:
        cpuDispatcherFunctions_.push_back(castToPolygonFunctions_[inputType]);
        break;
    case COLUMN_INT8_T:
        cpuDispatcherFunctions_.push_back(castToInt8TFunctions_[inputType]);
        break;
    default:
        break;
    }
}

void CpuSqlDispatcher::AddWhereResultFunction(DataType dataType)
{
    cpuDispatcherFunctions_.push_back(whereResultFunctions_[dataType]);
}

int64_t CpuSqlDispatcher::Execute(int32_t index)
{
    blockIndex_ = index;

    int32_t err = 0;
    while (err == 0)
    {
        err = (this->*cpuDispatcherFunctions_[instructionPointer_++])();
    }
    instructionPointer_ = 0;
    arguments_.Reset();

    for (auto& pointer : allocatedPointers_)
    {
        operator delete(reinterpret_cast<void*>(std::get<0>(pointer.second)));
    }
    allocatedPointers_.clear();

    return whereResult_;
}

template <>
int32_t CpuSqlDispatcher::LoadCol<std::string>(std::string& colName)
{
    if (allocatedPointers_.find(colName) == allocatedPointers_.end() && !colName.empty() && colName.front() != '$')
    {
        std::string tableName;
        std::string columnName;

        std::tie(tableName, columnName) = SplitColumnName(colName);
        if (blockIndex_ >= database_->GetTables().at(tableName).GetColumns().at(columnName).get()->GetBlockCount())
        {
            return 1;
        }

        std::string reg_min = colName + "_min";
        std::string reg_max = colName + "_max";

        std::string blockMin = GetBlockMin<std::string>(tableName, columnName);
        std::string blockMax = GetBlockMax<std::string>(tableName, columnName);

        char* mask_min = AllocateRegister<char>(reg_min, blockMin.size() + 1, false);
        char* mask_max = AllocateRegister<char>(reg_max, blockMax.size() + 1, false);

        std::copy(blockMin.begin(), blockMin.end(), mask_min);
        mask_min[blockMin.size()] = '\0';
        std::copy(blockMax.begin(), blockMax.end(), mask_max);
        mask_max[blockMax.size()] = '\0';
    }
    return 0;
}

void CpuSqlDispatcher::CopyExecutionDataTo(CpuSqlDispatcher& other)
{
    other.cpuDispatcherFunctions_ = cpuDispatcherFunctions_;
    other.arguments_ = arguments_;
}


std::pair<std::string, std::string> CpuSqlDispatcher::GetPointerNames(const std::string& colName)
{
    return {colName + "_min", colName + "_max"};
}
