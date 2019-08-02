#include "CpuSqlDispatcher.h"
#include "GpuSqlLexer.h"

std::array<CpuSqlDispatcher::CpuDispatchFunction, DataType::DATA_TYPE_SIZE> CpuSqlDispatcher::whereResultFunctions = {
    &CpuSqlDispatcher::whereResultConst<int32_t>,
    &CpuSqlDispatcher::whereResultConst<int64_t>,
    &CpuSqlDispatcher::whereResultConst<float>,
    &CpuSqlDispatcher::whereResultConst<double>,
    &CpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::Point>,
    &CpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<ColmnarDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::invalidOperandTypesErrorHandlerConst<std::string>,
    &CpuSqlDispatcher::whereResultConst<int8_t>,
    &CpuSqlDispatcher::whereResultCol<int32_t>,
    &CpuSqlDispatcher::whereResultCol<int64_t>,
    &CpuSqlDispatcher::whereResultCol<float>,
    &CpuSqlDispatcher::whereResultCol<double>,
    &CpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::Point>,
    &CpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<ColmnarDB::Types::ComplexPolygon>,
    &CpuSqlDispatcher::invalidOperandTypesErrorHandlerCol<std::string>,
    &CpuSqlDispatcher::whereResultCol<int8_t>};


CpuSqlDispatcher::CpuSqlDispatcher(const std::shared_ptr<Database>& database)
: database(database), blockIndex(0), instructionPointer(0), whereResult(1)
{
}

bool CpuSqlDispatcher::isRegisterAllocated(std::string& reg)
{
    return allocatedPointers.find(reg) != allocatedPointers.end();
}

std::pair<std::string, std::string> CpuSqlDispatcher::splitColumnName(const std::string& name)
{
    const size_t separatorPosition = name.find(".");
    const std::string table = name.substr(0, separatorPosition);
    const std::string column = name.substr(separatorPosition + 1);

    return std::make_pair(table, column);
}

void CpuSqlDispatcher::addBinaryOperation(DataType left, DataType right, size_t opType)
{
    switch (opType)
    {
    case GpuSqlLexer::GREATER:
        cpuDispatcherFunctions.push_back(greaterFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::LESS:
        cpuDispatcherFunctions.push_back(lessFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::GREATEREQ:
        cpuDispatcherFunctions.push_back(greaterEqualFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::LESSEQ:
        cpuDispatcherFunctions.push_back(lessEqualFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::EQUALS:
        cpuDispatcherFunctions.push_back(equalFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::NOTEQUALS:
    case GpuSqlLexer::NOTEQUALS_GT_LT:
        cpuDispatcherFunctions.push_back(notEqualFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::AND:
        cpuDispatcherFunctions.push_back(logicalAndFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::OR:
        cpuDispatcherFunctions.push_back(logicalOrFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::ASTERISK:
        cpuDispatcherFunctions.push_back(mulFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::DIVISION:
        cpuDispatcherFunctions.push_back(divFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::PLUS:
        cpuDispatcherFunctions.push_back(addFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::MINUS:
        cpuDispatcherFunctions.push_back(subFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::MODULO:
        cpuDispatcherFunctions.push_back(modFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::BIT_OR:
        cpuDispatcherFunctions.push_back(bitwiseOrFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::BIT_AND:
        cpuDispatcherFunctions.push_back(bitwiseAndFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::XOR:
        cpuDispatcherFunctions.push_back(bitwiseXorFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::L_SHIFT:
        cpuDispatcherFunctions.push_back(bitwiseLeftShiftFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::R_SHIFT:
        cpuDispatcherFunctions.push_back(bitwiseRightShiftFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::POINT:
        cpuDispatcherFunctions.push_back(pointFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::GEO_CONTAINS:
        cpuDispatcherFunctions.push_back(containsFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::GEO_INTERSECT:
        cpuDispatcherFunctions.push_back(intersectFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::GEO_UNION:
        cpuDispatcherFunctions.push_back(unionFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::LOG:
        cpuDispatcherFunctions.push_back(logarithmFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::POW:
        cpuDispatcherFunctions.push_back(powerFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::ROOT:
        cpuDispatcherFunctions.push_back(rootFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::ATAN2:
        cpuDispatcherFunctions.push_back(arctangent2Functions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::LEFT:
        cpuDispatcherFunctions.push_back(leftFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::RIGHT:
        cpuDispatcherFunctions.push_back(rightFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;

    case GpuSqlLexer::CONCAT:
        cpuDispatcherFunctions.push_back(concatFunctions[left * DataType::DATA_TYPE_SIZE + right]);
        break;
    default:
        break;
    }
}

void CpuSqlDispatcher::addUnaryOperation(DataType type, size_t opType)
{
    switch (opType)
    {
    case GpuSqlLexer::LOGICAL_NOT:
        cpuDispatcherFunctions.push_back(logicalNotFunctions[type]);
        break;
    case GpuSqlLexer::ISNULL:
        cpuDispatcherFunctions.push_back(nullFunction);
        break;
    case GpuSqlLexer::ISNOTNULL:
        cpuDispatcherFunctions.push_back(nullFunction);
        break;
    case GpuSqlLexer::MINUS:
        cpuDispatcherFunctions.push_back(minusFunctions[type]);
        break;
    case GpuSqlLexer::YEAR:
        cpuDispatcherFunctions.push_back(yearFunctions[type]);
        break;
    case GpuSqlLexer::MONTH:
        cpuDispatcherFunctions.push_back(monthFunctions[type]);
        break;
    case GpuSqlLexer::DAY:
        cpuDispatcherFunctions.push_back(dayFunctions[type]);
        break;
    case GpuSqlLexer::HOUR:
        cpuDispatcherFunctions.push_back(hourFunctions[type]);
        break;
    case GpuSqlLexer::MINUTE:
        cpuDispatcherFunctions.push_back(minuteFunctions[type]);
        break;
    case GpuSqlLexer::SECOND:
        cpuDispatcherFunctions.push_back(secondFunctions[type]);
        break;
    case GpuSqlLexer::ABS:
        cpuDispatcherFunctions.push_back(absoluteFunctions[type]);
        break;
    case GpuSqlLexer::SIN:
        cpuDispatcherFunctions.push_back(sineFunctions[type]);
        break;
    case GpuSqlLexer::COS:
        cpuDispatcherFunctions.push_back(cosineFunctions[type]);
        break;
    case GpuSqlLexer::TAN:
        cpuDispatcherFunctions.push_back(tangentFunctions[type]);
        break;
    case GpuSqlLexer::COT:
        cpuDispatcherFunctions.push_back(cotangentFunctions[type]);
        break;
    case GpuSqlLexer::ASIN:
        cpuDispatcherFunctions.push_back(arcsineFunctions[type]);
        break;
    case GpuSqlLexer::ACOS:
        cpuDispatcherFunctions.push_back(arccosineFunctions[type]);
        break;
    case GpuSqlLexer::ATAN:
        cpuDispatcherFunctions.push_back(arctangentFunctions[type]);
        break;
    case GpuSqlLexer::LOG10:
        cpuDispatcherFunctions.push_back(logarithm10Functions[type]);
        break;
    case GpuSqlLexer::LOG:
        cpuDispatcherFunctions.push_back(logarithmNaturalFunctions[type]);
        break;
    case GpuSqlLexer::EXP:
        cpuDispatcherFunctions.push_back(exponentialFunctions[type]);
        break;
    case GpuSqlLexer::SQRT:
        cpuDispatcherFunctions.push_back(squareRootFunctions[type]);
        break;
    case GpuSqlLexer::SQUARE:
        cpuDispatcherFunctions.push_back(squareFunctions[type]);
        break;
    case GpuSqlLexer::SIGN:
        cpuDispatcherFunctions.push_back(signFunctions[type]);
        break;
    case GpuSqlLexer::ROUND:
        cpuDispatcherFunctions.push_back(roundFunctions[type]);
        break;
    case GpuSqlLexer::FLOOR:
        cpuDispatcherFunctions.push_back(floorFunctions[type]);
        break;
    case GpuSqlLexer::CEIL:
        cpuDispatcherFunctions.push_back(ceilFunctions[type]);
        break;
    case GpuSqlLexer::LTRIM:
        cpuDispatcherFunctions.push_back(ltrimFunctions[type]);
        break;
    case GpuSqlLexer::RTRIM:
        cpuDispatcherFunctions.push_back(rtrimFunctions[type]);
        break;
    case GpuSqlLexer::LOWER:
        cpuDispatcherFunctions.push_back(lowerFunctions[type]);
        break;
    case GpuSqlLexer::UPPER:
        cpuDispatcherFunctions.push_back(upperFunctions[type]);
        break;
    case GpuSqlLexer::LEN:
        cpuDispatcherFunctions.push_back(lenFunctions[type]);
        break;

    default:
        break;
    }
}

void CpuSqlDispatcher::addCastOperation(DataType inputType, DataType outputType, const std::string& outTypeStr)
{
    switch (outputType)
    {
    case COLUMN_INT:
        cpuDispatcherFunctions.push_back(castToIntFunctions[inputType]);
        break;
    case COLUMN_LONG:
        if (outTypeStr == "DATE")
        {
            // dispatcher.addCastToDateFunction(operandType);
        }
        else
        {
            cpuDispatcherFunctions.push_back(castToLongFunctions[inputType]);
        }
        break;
    case COLUMN_FLOAT:
        cpuDispatcherFunctions.push_back(castToFloatFunctions[inputType]);
        break;
    case COLUMN_DOUBLE:
        cpuDispatcherFunctions.push_back(castToDoubleFunctions[inputType]);
        break;
    case COLUMN_STRING:
        cpuDispatcherFunctions.push_back(castToStringFunctions[inputType]);
        break;
    case COLUMN_POINT:
        cpuDispatcherFunctions.push_back(castToPointFunctions[inputType]);
        break;
    case COLUMN_POLYGON:
        cpuDispatcherFunctions.push_back(castToPolygonFunctions[inputType]);
        break;
    case COLUMN_INT8_T:
        cpuDispatcherFunctions.push_back(castToInt8tFunctions[inputType]);
        break;
    default:
        break;
    }
}

void CpuSqlDispatcher::addWhereResultFunction(DataType dataType)
{
    cpuDispatcherFunctions.push_back(whereResultFunctions[dataType]);
}

int64_t CpuSqlDispatcher::execute(int32_t index)
{
    blockIndex = index;

    int32_t err = 0;
    while (err == 0)
    {
        err = (this->*cpuDispatcherFunctions[instructionPointer++])();
    }
    instructionPointer = 0;
    arguments.reset();

    for (auto& pointer : allocatedPointers)
    {
        operator delete(reinterpret_cast<void*>(std::get<0>(pointer.second)));
    }
    allocatedPointers.clear();

    return whereResult;
}

template <>
int32_t CpuSqlDispatcher::loadCol<std::string>(std::string& colName)
{
    if (allocatedPointers.find(colName) == allocatedPointers.end() && !colName.empty() && colName.front() != '$')
    {
        std::string tableName;
        std::string columnName;

        std::tie(tableName, columnName) = splitColumnName(colName);
        if (blockIndex >= database->GetTables().at(tableName).GetColumns().at(columnName).get()->GetBlockCount())
        {
            return 1;
        }

        std::string reg_min = colName + "_min";
        std::string reg_max = colName + "_max";

        std::string blockMin = getBlockMin<std::string>(tableName, columnName);
        std::string blockMax = getBlockMax<std::string>(tableName, columnName);

        char* mask_min = allocateRegister<char>(reg_min, blockMin.size() + 1, false);
        char* mask_max = allocateRegister<char>(reg_max, blockMax.size() + 1, false);

        std::copy(blockMin.begin(), blockMin.end(), mask_min);
        mask_min[blockMin.size()] = '\0';
        std::copy(blockMax.begin(), blockMax.end(), mask_max);
        mask_max[blockMax.size()] = '\0';
    }
    return 0;
}

void CpuSqlDispatcher::copyExecutionDataTo(CpuSqlDispatcher& other)
{
    other.cpuDispatcherFunctions = cpuDispatcherFunctions;
    other.arguments = arguments;
}


std::pair<std::string, std::string> CpuSqlDispatcher::getPointerNames(const std::string& colName)
{
    return {colName + "_min", colName + "_max"};
}
