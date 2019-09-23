//
// Created by Martin Staňo on 2019-01-15.
//

#include "GpuSqlListener.h"
#include "../Table.h"
#include "../Database.h"
#include "../PointFactory.h"
#include "../ComplexPolygonFactory.h"
#include "ParserExceptions.h"
#include "JoinType.h"
#include "GroupByType.h"
#include "GpuSqlDispatcher.h"
#include "GpuSqlJoinDispatcher.h"
#include <ctime>
#include <iostream>
#include <sstream>
#include <locale>
#include <iomanip>
#include <boost/algorithm/string.hpp>

/// <summary>
/// Definition of PI constant
/// </summary>
constexpr float pi()
{
    return 3.1415926f;
}

/// <summary>
/// GpuListner Constructor
/// Initializes AST walk flags (insideAgg, insideGroupBy, etc.)
/// Takes reference to database and dispatcher instances
/// </summary>
/// <param name="database">Database instance reference</param>
/// <param name="dispatcher">Dispatcher instance reference</param>
GpuSqlListener::GpuSqlListener(const std::shared_ptr<Database>& database,
                               GpuSqlDispatcher& dispatcher,
                               GpuSqlJoinDispatcher& joinDispatcher)
: database_(database), dispatcher_(dispatcher), joinDispatcher_(joinDispatcher), linkTableIndex_(0),
  orderByColumnIndex_(0), usingGroupBy_(false), usingAgg_(false), insideAgg_(false),
  insideWhere_(false), insideGroupBy_(false), insideOrderBy_(false), insideAlias_(false),
  insideSelectColumn_(false), isAggSelectColumn_(false), isSelectColumnValid_(false),
  ResultLimit(std::numeric_limits<int64_t>::max()), ResultOffset(0), CurrentSelectColumnIndex(0),
  currentExpressionAlias_("")
{
    GpuSqlDispatcher::linkTable.clear();
}

/// <summary>
/// Method that executes on exit of binary operation node in the AST
/// Pops the two operands from stack, reads operation from context and add dispatcher
/// operation and operands to respective dispatcher queues. Pushes result back to parser stack
/// </summary>
/// <param name="ctx">Binary operation context</param>
void GpuSqlListener::exitBinaryOperation(GpuSqlParser::BinaryOperationContext* ctx)
{
    std::pair<std::string, DataType> right = StackTopAndPop();
    std::pair<std::string, DataType> left = StackTopAndPop();

    std::string op = ctx->op->getText();
    StringToUpper(op);

    DataType rightOperandType = std::get<1>(right);
    DataType leftOperandType = std::get<1>(left);

    std::string rightOperand = std::get<0>(right);
    std::string leftOperand = std::get<0>(left);

    PushArgument(rightOperand.c_str(), rightOperandType);
    PushArgument(leftOperand.c_str(), leftOperandType);

    DataType returnDataType = DataType::CONST_ERROR;

    std::string reg;
    TrimReg(rightOperand);
    TrimReg(leftOperand);
    switch (ctx->op->getType())
    {
    case GpuSqlLexer::GREATER:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddGreaterFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::LESS:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddLessFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::GREATEREQ:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddGreaterEqualFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::LESSEQ:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddLessEqualFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::EQUALS:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddEqualFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::NOTEQUALS:
    case GpuSqlLexer::NOTEQUALS_GT_LT:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddNotEqualFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::AND:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddLogicalAndFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::OR:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddLogicalOrFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::ASTERISK:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddMulFunction(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::DIVISION:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddDivFunction(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::PLUS:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddAddFunction(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::MINUS:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddSubFunction(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::MODULO:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddModFunction(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::BIT_OR:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddBitwiseOrFunction(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::BIT_AND:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddBitwiseAndFunction(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::XOR:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddBitwiseXorFunction(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::L_SHIFT:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddBitwiseLeftShiftFunction(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::R_SHIFT:
        reg = "$" + leftOperand + op + rightOperand;
        dispatcher_.AddBitwiseRightShiftFunction(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::POINT:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        dispatcher_.AddPointFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_POINT;
        break;
    case GpuSqlLexer::GEO_CONTAINS:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        dispatcher_.AddContainsFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::GEO_INTERSECT:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        dispatcher_.AddIntersectFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_POLYGON;
        break;
    case GpuSqlLexer::GEO_UNION:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        dispatcher_.AddUnionFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_POLYGON;
        break;
    case GpuSqlLexer::LOG:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        dispatcher_.AddLogarithmFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::POW:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        dispatcher_.AddPowerFunction(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::ROOT:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        dispatcher_.AddRootFunction(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::ATAN2:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        dispatcher_.AddArctangent2Function(leftOperandType, rightOperandType);
        returnDataType = GetReturnDataType(DataType::COLUMN_FLOAT);
        break;
    case GpuSqlLexer::CONCAT:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        dispatcher_.AddConcatFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::LEFT:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        dispatcher_.AddLeftFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::RIGHT:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        dispatcher_.AddRightFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_STRING;
        break;
    default:
        break;
    }

    if (groupByColumns_.find({reg, returnDataType}) != groupByColumns_.end() && insideSelectColumn_)
    {
        isSelectColumnValid_ = true;
    }

    PushArgument(reg.c_str(), returnDataType);
    PushTempResult(reg, returnDataType);
}

/// <summary>
/// Method that executes on exit of ternary operation node in the AST
/// Pops the three operands from stack, reads operation from context and add dispatcher
/// operation and operands to respective dispatcher queues. Pushes result back to parser stack
/// </summary>
/// <param name="ctx">Ternary operation context</param>
void GpuSqlListener::exitTernaryOperation(GpuSqlParser::TernaryOperationContext* ctx)
{
    std::pair<std::string, DataType> op1 = StackTopAndPop();
    std::pair<std::string, DataType> op2 = StackTopAndPop();
    std::pair<std::string, DataType> op3 = StackTopAndPop();

    std::string op = ctx->op->getText();
    StringToUpper(op);

    DataType op1Type = std::get<1>(op1);
    DataType op2Type = std::get<1>(op2);
    DataType op3Type = std::get<1>(op3);

    std::string op1Str = std::get<0>(op1);
    std::string op2Str = std::get<0>(op2);
    std::string op3Str = std::get<0>(op3);

    PushArgument(op1Str.c_str(), op1Type);
    PushArgument(op2Str.c_str(), op2Type);
    PushArgument(op3Str.c_str(), op3Type);

    std::string reg;
    TrimReg(op3Str);
    TrimReg(op2Str);
    TrimReg(op1Str);

    switch (ctx->op->getType())
    {
    case GpuSqlLexer::BETWEEN:
        reg = "$" + op + "(" + op3Str + "," + op2Str + "," + op1Str + ")";
        dispatcher_.AddBetweenFunction(op1Type, op2Type, op3Type);
        break;
    default:
        break;
    }

    PushArgument(reg.c_str(), ::COLUMN_INT8_T);
    PushTempResult(reg, DataType::COLUMN_INT8_T);
}

/// <summary>
/// Method that executes on exit of unary operation node in the AST
/// Pops the one operand from stack, reads operation from context and add dispatcher
/// operation and operand to respective dispatcher queues. Pushes result back to parser stack
/// </summary>
/// <param name="ctx">Unary operation context</param>
void GpuSqlListener::exitUnaryOperation(GpuSqlParser::UnaryOperationContext* ctx)
{
    std::pair<std::string, DataType> arg = StackTopAndPop();

    std::string op = ctx->op->getText();
    StringToUpper(op);

    std::string operand = std::get<0>(arg);
    DataType operandType = std::get<1>(arg);

    PushArgument(operand.c_str(), operandType);

    DataType returnDataType = DataType::CONST_ERROR;

    std::string reg;
    TrimReg(operand);

    switch (ctx->op->getType())
    {
    case GpuSqlLexer::LOGICAL_NOT:
        reg = "$" + op + operand;
        dispatcher_.AddLogicalNotFunction(operandType);
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::ISNULL:
        reg = "$" + op + operand;
        if (operandType < DataType::COLUMN_INT)
        {
            throw NullMaskOperationInvalidOperandException();
        }
        dispatcher_.AddIsNullFunction();
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::ISNOTNULL:
        reg = "$" + op + operand;
        if (operandType < DataType::COLUMN_INT)
        {
            throw NullMaskOperationInvalidOperandException();
        }
        dispatcher_.AddIsNotNullFunction();
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::MINUS:
        reg = "$" + op + operand;
        dispatcher_.AddMinusFunction(operandType);
        returnDataType = GetReturnDataType(operandType);
        break;
    case GpuSqlLexer::YEAR:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddYearFunction(operandType);
        returnDataType = COLUMN_INT;
        break;
    case GpuSqlLexer::MONTH:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddMonthFunction(operandType);
        returnDataType = COLUMN_INT;
        break;
    case GpuSqlLexer::DAY:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddDayFunction(operandType);
        returnDataType = COLUMN_INT;
        break;
    case GpuSqlLexer::HOUR:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddHourFunction(operandType);
        returnDataType = COLUMN_INT;
        break;
    case GpuSqlLexer::MINUTE:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddMinuteFunction(operandType);
        returnDataType = COLUMN_INT;
        break;
    case GpuSqlLexer::SECOND:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddSecondFunction(operandType);
        returnDataType = COLUMN_INT;
        break;
    case GpuSqlLexer::ABS:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddAbsoluteFunction(operandType);
        returnDataType = GetReturnDataType(operandType);
        break;
    case GpuSqlLexer::SIN:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddSineFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::COS:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddCosineFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::TAN:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddTangentFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::COT:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddCotangentFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::ASIN:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddArcsineFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::ACOS:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddArccosineFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::ATAN:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddArctangentFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::LOG10:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddLogarithm10Function(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::LOG:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddLogarithmNaturalFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::EXP:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddExponentialFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::SQRT:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddSquareRootFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::SQUARE:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddSquareFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::SIGN:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddSignFunction(operandType);
        returnDataType = DataType::COLUMN_INT;
        break;
    case GpuSqlLexer::ROUND:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddRoundFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::FLOOR:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddFloorFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::CEIL:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddCeilFunction(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::LTRIM:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddLtrimFunction(operandType);
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::RTRIM:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddRtrimFunction(operandType);
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::LOWER:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddLowerFunction(operandType);
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::UPPER:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddUpperFunction(operandType);
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::REVERSE:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddReverseFunction(operandType);
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::LEN:
        reg = "$" + op + "(" + operand + ")";
        dispatcher_.AddLenFunction(operandType);
        returnDataType = DataType::COLUMN_INT;
        break;
    default:
        break;
    }

    if (groupByColumns_.find({reg, returnDataType}) != groupByColumns_.end() && insideSelectColumn_)
    {
        isSelectColumnValid_ = true;
    }

    PushArgument(reg.c_str(), returnDataType);
    PushTempResult(reg, returnDataType);
}

void GpuSqlListener::exitCastOperation(GpuSqlParser::CastOperationContext* ctx)
{
    std::pair<std::string, DataType> arg = StackTopAndPop();

    std::string operand = std::get<0>(arg);
    DataType operandType = std::get<1>(arg);

    PushArgument(operand.c_str(), operandType);
    std::string castTypeStr = ctx->DATATYPE()->getText();
    StringToUpper(castTypeStr);
    DataType castType = GetDataTypeFromString(castTypeStr);

    switch (castType)
    {
    case COLUMN_INT:
        dispatcher_.AddCastToIntFunction(operandType);
        break;
    case COLUMN_LONG:
        if (castTypeStr == "DATE")
        {
            dispatcher_.AddCastToDateFunction(operandType);
        }
        else
        {
            dispatcher_.AddCastToLongFunction(operandType);
        }
        break;
    case COLUMN_FLOAT:
        dispatcher_.AddCastToFloatFunction(operandType);
        break;
    case COLUMN_DOUBLE:
        dispatcher_.AddCastToDoubleFunction(operandType);
        break;
    case COLUMN_STRING:
        dispatcher_.AddCastToStringFunction(operandType);
        break;
    case COLUMN_POINT:
        dispatcher_.AddCastToPointFunction(operandType);
        break;
    case COLUMN_POLYGON:
        dispatcher_.AddCastToPolygonFunction(operandType);
        break;
    case COLUMN_INT8_T:
        dispatcher_.AddCastToInt8TFunction(operandType);
        break;
    default:
        break;
    }

    TrimReg(operand);
    std::string reg = "$CAST(" + operand + "AS" + castTypeStr + ")";

    if (groupByColumns_.find({reg, castType}) != groupByColumns_.end() && insideSelectColumn_)
    {
        isSelectColumnValid_ = true;
    }

    PushArgument(reg.c_str(), castType);
    PushTempResult(reg, castType);
}

/// <summary>
/// Method that executes on enter of aggregation operation node in the AST
/// Sets insideAgg, isAggSelectColumn parser flag
/// Throws NestedAggregationException in case e.g SUM(SUM(colA))
/// </summary>
/// <param name="ctx">Aggregation context</param>
void GpuSqlListener::enterAggregation(GpuSqlParser::AggregationContext* ctx)
{
    if (insideAgg_)
    {
        throw NestedAggregationException();
    }
    if (insideWhere_)
    {
        throw AggregationWhereException();
    }
    if (insideGroupBy_)
    {
        throw AggregationGroupByException();
    }
    insideAgg_ = true;
    usingAgg_ = true;
    isAggSelectColumn_ = insideSelectColumn_;
    dispatcher_.AddAggregationBeginFunction();
}

/// <summary>
/// Method that executes on exit of aggregation operation node in the AST
/// Pops one operand from stack, adds aggregation operation and argument to respective Dispatcher
/// queues. Pushes result back to parser stack.
/// </summary>
/// <param name="ctx">Aggregation context</param>
void GpuSqlListener::exitAggregation(GpuSqlParser::AggregationContext* ctx)
{
    bool aggAsterisk = false;

    if (ctx->COUNT_AGG() && ctx->ASTERISK())
    {
        aggAsterisk = true;
        // JOIN case handled in dispatcher
        std::string tableName = *(loadedTables_.begin());
        PushTempResult(tableName, COLUMN_INT);
    }
    else if (ctx->COUNT_AGG() && ctx->ASTERISK() == nullptr && usingGroupBy_)
    {
        std::pair<std::string, DataType> arg = StackTopAndPop();
        std::string value = std::get<0>(arg);
        PushTempResult(value, COLUMN_INT);
    }

    std::pair<std::string, DataType> arg = StackTopAndPop();

    std::string op = ctx->op->getText();
    StringToUpper(op);

    std::string value = std::get<0>(arg);
    DataType valueType = std::get<1>(arg);

    PushArgument(value.c_str(), valueType);
    DataType returnDataType = DataType::CONST_ERROR;

    GroupByType groupByType = GroupByType::NO_GROUP_BY;
    DataType keyType = static_cast<DataType>(0);

    if (usingGroupBy_)
    {
        groupByType = GroupByType::SINGLE_KEY_GROUP_BY;
        keyType = std::get<1>(*(groupByColumns_.begin()));

        if (groupByColumns_.size() > 1)
        {
            groupByType = GroupByType::MULTI_KEY_GROUP_BY;
        }
    }

    std::string reg;
    TrimReg(value);
    switch (ctx->op->getType())
    {
    case GpuSqlLexer::MIN_AGG:
        reg = "$" + op + "(" + value + ")";
        dispatcher_.AddMinFunction(keyType, valueType, groupByType);
        returnDataType = GetReturnDataType(valueType);
        break;
    case GpuSqlLexer::MAX_AGG:
        reg = "$" + op + "(" + value + ")";
        dispatcher_.AddMaxFunction(keyType, valueType, groupByType);
        returnDataType = GetReturnDataType(valueType);
        break;
    case GpuSqlLexer::SUM_AGG:
        reg = "$" + op + "(" + value + ")";
        dispatcher_.AddSumFunction(keyType, valueType, groupByType);
        returnDataType = GetReturnDataType(valueType);
        break;
    case GpuSqlLexer::COUNT_AGG:
        reg = "$" + op + "(" + (aggAsterisk ? "*" : value) + ")";
        dispatcher_.AddCountFunction(keyType, valueType, groupByType);
        returnDataType = DataType::COLUMN_LONG;
        break;
    case GpuSqlLexer::AVG_AGG:
        reg = "$" + op + "(" + value + ")";
        dispatcher_.AddAvgFunction(keyType, valueType, groupByType);
        returnDataType = GetReturnDataType(valueType);
        break;
    default:
        break;
    }

    if (insideSelectColumn_)
    {
        isSelectColumnValid_ = true;
    }

    PushArgument(reg.c_str(), returnDataType);
    PushTempResult(reg, returnDataType);

    dispatcher_.AddArgument<bool>(aggAsterisk);

    dispatcher_.AddAggregationDoneFunction();
    insideAgg_ = false;
}

/// Method that executes on exit of SELECT clause (return columns)
/// Generates jump operation (used to iterate blocks) and done operation (marking end of exucution)
/// <param name="ctx">Select Columns context</param>
void GpuSqlListener::exitSelectColumns(GpuSqlParser::SelectColumnsContext* ctx)
{
    for (auto& retCol : returnColumns_)
    {
        std::string colName = retCol.first;
        DataType retType = std::get<0>(retCol.second);
        std::string alias = std::get<1>(retCol.second);
        dispatcher_.AddRetFunction(retType);
        PushArgument(colName.c_str(), retType);
        dispatcher_.AddArgument<const std::string&>(alias);
    }

    dispatcher_.AddJmpInstruction();
    dispatcher_.AddDoneFunction();
}


/// Method that executes on enter of SELECT clause (return columns)
/// Sets insideSelectColumn parser flag
/// <param name="ctx">Select Columns context</param>
void GpuSqlListener::enterSelectColumn(GpuSqlParser::SelectColumnContext* ctx)
{
    insideSelectColumn_ = true;
    if (ctx->alias())
    {
        currentExpressionAlias_ = ctx->alias()->getText();
        TrimDelimitedIdentifier(currentExpressionAlias_);
    }
    else
    {
        currentExpressionAlias_ = "";
    }
    isSelectColumnValid_ = !usingGroupBy_;
}


/// Method that executes on exit of single SELECT column (return column)
/// Checks if a column is either aggregation or group by column
/// Pops from parser stack and generates return operation
/// Sets column alias if present and checks its potential redefinition
/// Resets insideAggregation and insideSelectColumn parser flags
/// <param name="ctx">Select Column context</param>
void GpuSqlListener::exitSelectColumn(GpuSqlParser::SelectColumnContext* ctx)
{
    std::pair<std::string, DataType> arg = StackTopAndPop();
    std::string colName = std::get<0>(arg);
    DataType retType = std::get<1>(arg);

    isSelectColumnValid_ = isSelectColumnValid_ && !(!usingGroupBy_ && !isAggSelectColumn_ && usingAgg_);

    if (!isSelectColumnValid_)
    {
        throw ColumnGroupByException(colName);
    }

    std::string alias;

    if (ctx->alias())
    {
        alias = ctx->alias()->getText();
        TrimDelimitedIdentifier(alias);

        if (columnAliases_.find(alias) != columnAliases_.end())
        {
            throw AliasRedefinitionException(alias);
        }
        columnAliases_.insert(alias);
    }
    else
    {
        alias = colName;
    }

    if (returnColumns_.find(colName) == returnColumns_.end())
    {
        returnColumns_.insert({colName, {retType, alias}});
        ColumnOrder.insert({CurrentSelectColumnIndex, alias});

        dispatcher_.AddArgument<const std::string&>(colName);
        dispatcher_.AddLockRegisterFunction();
    }
    currentExpressionAlias_ = "";
    insideSelectColumn_ = false;
    isAggSelectColumn_ = false;
}

void GpuSqlListener::exitSelectAllColumns(GpuSqlParser::SelectAllColumnsContext* ctx)
{
    int32_t columnOrderNumber = 0;
    for (auto& tableName : loadedTables_)
    {
        const Table& table = database_->GetTables().at(tableName);
        for (auto& columnPair : table.GetColumns())
        {
            std::string colName = tableName + "." + columnPair.first;
            DataType retType = columnPair.second->GetColumnType();

            if (returnColumns_.find(colName) == returnColumns_.end())
            {
                returnColumns_.insert({colName, {retType, colName}});
                ColumnOrder.insert({columnOrderNumber++, colName});

                dispatcher_.AddArgument<const std::string&>(colName);
                dispatcher_.AddLockRegisterFunction();
            }
        }
    }
}


/// Method that executes on exit of FROM clause (tables)
/// Checks for table existance
/// Sets table alias if present and checks its potential redefinition
/// <param name="ctx">From Tables context</param>
void GpuSqlListener::exitFromTables(GpuSqlParser::FromTablesContext* ctx)
{
    for (auto fromTable : ctx->fromTable())
    {
        std::string table = fromTable->table()->getText();
        TrimDelimitedIdentifier(table);
        if (database_->GetTables().find(table) == database_->GetTables().end())
        {
            throw TableNotFoundFromException(table);
        }
        loadedTables_.insert(table);
        dispatcher_.SetLoadedTableName(table);
        if (fromTable->alias())
        {
            std::string alias = fromTable->alias()->getText();
            TrimDelimitedIdentifier(alias);

            if (tableAliases_.find(alias) != tableAliases_.end())
            {
                throw AliasRedefinitionException(alias);
            }
            tableAliases_.insert({alias, table});
        }
    }
}

void GpuSqlListener::exitJoinClause(GpuSqlParser::JoinClauseContext* ctx)
{
    std::string joinTable = ctx->joinTable()->table()->getText();
    TrimDelimitedIdentifier(joinTable);

    if (database_->GetTables().find(joinTable) == database_->GetTables().end())
    {
        throw TableNotFoundFromException(joinTable);
    }

    loadedTables_.insert(joinTable);

    if (ctx->joinTable()->alias())
    {
        std::string alias = ctx->joinTable()->alias()->getText();
        TrimDelimitedIdentifier(alias);

        if (tableAliases_.find(alias) != tableAliases_.end())
        {
            throw AliasRedefinitionException(alias);
        }
        tableAliases_.insert({alias, joinTable});
    }

    std::string leftColName;
    DataType leftColType;
    std::tie(leftColName, leftColType) = GenerateAndValidateColumnName(ctx->joinColumnLeft()->columnId());

    std::string rightColName;
    DataType rightColType;
    std::tie(rightColName, rightColType) = GenerateAndValidateColumnName(ctx->joinColumnRight()->columnId());

    if (leftColType != rightColType)
    {
        throw JoinColumnTypeException(leftColName, rightColName);
    }

    JoinType joinType = JoinType::INNER_JOIN;
    if (ctx->joinType())
    {
        std::string joinTypeName = ctx->joinType()->getText();
        StringToUpper(joinTypeName);

        if (joinTypeName == "INNER")
        {
            joinType = JoinType::INNER_JOIN;
        }
        else if (joinTypeName == "LEFT")
        {
            joinType = JoinType::LEFT_JOIN;
        }
        else if (joinTypeName == "RIGHT")
        {
            joinType = JoinType::RIGHT_JOIN;
        }
        else if (joinTypeName == "FULL OUTER")
        {
            joinType = JoinType::FULL_OUTER_JOIN;
        }
    }

    std::string joinOperator = ctx->joinOperator()->getText();

    joinDispatcher_.AddJoinFunction(leftColType, joinOperator);
    joinDispatcher_.AddArgument<const std::string&>(leftColName);
    joinDispatcher_.AddArgument<const std::string&>(rightColName);
    joinDispatcher_.AddArgument<int32_t>(joinType);
}

void GpuSqlListener::exitJoinClauses(GpuSqlParser::JoinClausesContext* ctx)
{
    joinDispatcher_.AddJoinDoneFunction();
}


/// Method that executes on exit of WHERE clause
/// Pops from parser stack, generates fil operation which marks register
/// used as final filtration mask in recostruct operations
/// <param name="ctx">Where Clause context</param>
void GpuSqlListener::exitWhereClause(GpuSqlParser::WhereClauseContext* ctx)
{
    std::pair<std::string, DataType> arg = StackTopAndPop();
    dispatcher_.AddArgument<const std::string&>(std::get<0>(arg));
    dispatcher_.AddFilFunction();
    insideWhere_ = false;
}

void GpuSqlListener::enterWhereClause(GpuSqlParser::WhereClauseContext* ctx)
{
    insideWhere_ = true;
    dispatcher_.AddWhereEvaluationFunction();
}


/// Method that executes on enter of GROUP BY clause
/// Sets insideGroupBy parser flag.
/// <param name="ctx">Group By Columns context</param>
void GpuSqlListener::enterGroupByColumns(GpuSqlParser::GroupByColumnsContext* ctx)
{
    dispatcher_.AddGroupByBeginFunction();
    insideGroupBy_ = true;
}


/// Method that executes on exit of GROUP BY clause
/// Sets usingGroupBy and resets insideGoupBy parser flags
/// <param name="ctx">Group By Columns context</param>
void GpuSqlListener::exitGroupByColumns(GpuSqlParser::GroupByColumnsContext* ctx)
{
    dispatcher_.AddGroupByDoneFunction();
    usingGroupBy_ = true;
    insideGroupBy_ = false;
}


/// Method that executes on exit of a single GROUP BY column
/// Pops from parser stack and generates group by operation
/// Appends to list of group by columns
/// <param name="ctx">Group By Column context</param>
void GpuSqlListener::exitGroupByColumn(GpuSqlParser::GroupByColumnContext* ctx)
{
    std::pair<std::string, DataType> operand = StackTopAndPop();
    std::string groupByColName = std::get<0>(operand);
    DataType groupByDataType = std::get<1>(operand);


    if (groupByDataType < DataType::COLUMN_INT)
    {
        if (groupByDataType != DataType::CONST_INT && groupByDataType != DataType::CONST_LONG)
        {
            throw GroupByInvalidColumnException(groupByColName);
        }
        else
        {
            int64_t value = std::stoll(groupByColName);

            if (columnNumericAliasContexts_.find(value) != columnNumericAliasContexts_.end() && !insideAlias_)
            {
                WalkAliasExpression(value);
                operand = StackTopAndPop();
                groupByColName = std::get<0>(operand);
                groupByDataType = std::get<1>(operand);
            }
            else
            {
                throw GroupByInvalidColumnException(groupByColName);
            }
        }
    }

    if (groupByColumns_.find(operand) == groupByColumns_.end())
    {
        dispatcher_.AddGroupByFunction(groupByDataType);
        dispatcher_.AddArgument<const std::string&>(groupByColName);
        groupByColumns_.insert(operand);
    }
}


/// Method that executes on exit of SHOW DATABASES command
/// Generates show databases operation
/// <param name="ctx">Show Databases context</param>
void GpuSqlListener::exitShowDatabases(GpuSqlParser::ShowDatabasesContext* ctx)
{
    dispatcher_.AddShowDatabasesFunction();
    ColumnOrder.insert({0, "Databases"});
}

/// Method that executes on exit of SHOW TABLES command
/// Generates show tables operation
/// Checks if database with given name exists
/// If no database name is provided uses currently bound database
/// <param name="ctx">Show Tables context</param>
void GpuSqlListener::exitShowTables(GpuSqlParser::ShowTablesContext* ctx)
{
    dispatcher_.AddShowTablesFunction();
    std::string db;

    if (ctx->database())
    {
        db = ctx->database()->getText();
        TrimDelimitedIdentifier(db);

        if (!Database::Exists(db))
        {
            throw DatabaseNotFoundException(db);
        }
    }
    else
    {
        if (database_)
        {
            db = database_->GetName();
        }
        else
        {
            throw DatabaseNotFoundException(db);
        }
    }

    dispatcher_.AddArgument<const std::string&>(db);
    ColumnOrder.insert({0, db});
}

/// Method that executes on exit of SHOW COLUMNS command
/// Generates show tables operation
/// Checks if database with given name exists
/// If no database name is provided uses currently bound database
/// Checks if table with given name exists
/// <param name="ctx">Show Columns context</param>
void GpuSqlListener::exitShowColumns(GpuSqlParser::ShowColumnsContext* ctx)
{
    dispatcher_.AddShowColumnsFunction();
    std::string db;
    std::string table;

    if (ctx->database())
    {
        db = ctx->database()->getText();
        TrimDelimitedIdentifier(db);

        if (!Database::Exists(db))
        {
            throw DatabaseNotFoundException(db);
        }
    }
    else
    {
        if (database_)
        {

            db = database_->GetName();
        }
        else
        {
            throw DatabaseNotFoundException(db);
        }
    }

    std::shared_ptr<Database> databaseObject = Database::GetDatabaseByName(db);
    table = ctx->table()->getText();
    TrimDelimitedIdentifier(table);

    if (databaseObject->GetTables().find(table) == databaseObject->GetTables().end())
    {
        throw TableNotFoundFromException(table);
    }

    dispatcher_.AddArgument<const std::string&>(db);
    dispatcher_.AddArgument<const std::string&>(table);

    ColumnOrder.insert({0, table + "_columns"});
    ColumnOrder.insert({1, table + "_types"});
}

void GpuSqlListener::exitSqlCreateDb(GpuSqlParser::SqlCreateDbContext* ctx)
{
    std::string newDbName = ctx->database()->getText();
    TrimDelimitedIdentifier(newDbName);

    if (Database::Exists(newDbName))
    {
        throw DatabaseAlreadyExistsException(newDbName);
    }

    int32_t newDbBlockSize;

    if (ctx->blockSize())
    {
        newDbBlockSize = std::stoi(ctx->blockSize()->getText());
    }
    else
    {
        newDbBlockSize = Configuration::GetInstance().GetBlockSize();
    }

    dispatcher_.AddCreateDatabaseFunction();
    dispatcher_.AddArgument<const std::string&>(newDbName);
    dispatcher_.AddArgument<int32_t>(newDbBlockSize);
}

void GpuSqlListener::exitSqlDropDb(GpuSqlParser::SqlDropDbContext* ctx)
{
    std::string dbName = ctx->database()->getText();
    TrimDelimitedIdentifier(dbName);

    if (!Database::Exists(dbName))
    {
        throw DatabaseNotFoundException(dbName);
    }

    dispatcher_.AddDropDatabaseFunction();
    dispatcher_.AddArgument<const std::string&>(dbName);
}

void GpuSqlListener::exitSqlCreateTable(GpuSqlParser::SqlCreateTableContext* ctx)
{
    std::string newTableName = ctx->table()->getText();
    TrimDelimitedIdentifier(newTableName);

    if (database_->GetTables().find(newTableName) != database_->GetTables().end())
    {
        throw TableAlreadyExistsException(newTableName);
    }

    std::unordered_map<std::string, DataType> newColumns;
    std::unordered_map<std::string, std::vector<std::string>> newIndices;

    for (auto& entry : ctx->newTableEntries()->newTableEntry())
    {
        if (entry->newTableColumn())
        {
            auto newColumnContext = entry->newTableColumn();
            DataType newColumnDataType = GetDataTypeFromString(newColumnContext->DATATYPE()->getText());
            std::string newColumnName = newColumnContext->column()->getText();
            TrimDelimitedIdentifier(newColumnName);

            if (newColumns.find(newColumnName) != newColumns.end())
            {
                throw ColumnAlreadyExistsException(newColumnName);
            }

            newColumns.insert({newColumnName, newColumnDataType});
        }
        if (entry->newTableIndex())
        {
            auto newColumnContext = entry->newTableIndex();
            std::string indexName = newColumnContext->indexName()->getText();
            TrimDelimitedIdentifier(indexName);

            if (newIndices.find(indexName) != newIndices.end())
            {
                throw IndexAlreadyExistsException(indexName);
            }

            std::vector<std::string> indexColumns;
            for (auto& column : newColumnContext->indexColumns()->column())
            {
                std::string indexColumnName = column->getText();
                TrimDelimitedIdentifier(indexColumnName);

                if (newColumns.find(indexColumnName) == newColumns.end())
                {
                    throw ColumnNotFoundException(indexColumnName);
                }

                DataType indexColumnDataType = newColumns.at(indexColumnName);
                if (indexColumnDataType == DataType::COLUMN_POINT || indexColumnDataType == DataType::COLUMN_POLYGON)
                {
                    throw IndexColumnDataTypeException(indexColumnName, indexColumnDataType);
                }

                if (std::find(indexColumns.begin(), indexColumns.end(), indexColumnName) !=
                    indexColumns.end())
                {
                    throw ColumnAlreadyExistsInIndexException(indexColumnName);
                }
                indexColumns.push_back(indexColumnName);
            }
            newIndices.insert({indexName, indexColumns});
        }
    }

    dispatcher_.AddCreateTableFunction();

    dispatcher_.AddArgument<const std::string&>(newTableName);
    dispatcher_.AddArgument<int32_t>(newColumns.size());
    for (auto& newColumn : newColumns)
    {
        dispatcher_.AddArgument<const std::string&>(newColumn.first);
        dispatcher_.AddArgument<int32_t>(static_cast<int32_t>(newColumn.second));
    }

    dispatcher_.AddArgument<int32_t>(newIndices.size());
    for (auto& newIndex : newIndices)
    {
        dispatcher_.AddArgument<const std::string&>(newIndex.first);
        dispatcher_.AddArgument<int32_t>(newIndex.second.size());
        for (auto& indexColumn : newIndex.second)
        {
            dispatcher_.AddArgument<const std::string&>(indexColumn);
        }
    }
}

void GpuSqlListener::exitSqlDropTable(GpuSqlParser::SqlDropTableContext* ctx)
{
    std::string tableName = ctx->table()->getText();
    TrimDelimitedIdentifier(tableName);

    if (database_->GetTables().find(tableName) == database_->GetTables().end())
    {
        throw TableNotFoundFromException(tableName);
    }

    dispatcher_.AddDropTableFunction();
    dispatcher_.AddArgument<const std::string&>(tableName);
}

void GpuSqlListener::exitSqlAlterTable(GpuSqlParser::SqlAlterTableContext* ctx)
{
    std::string tableName = ctx->table()->getText();
    TrimDelimitedIdentifier(tableName);

    if (database_->GetTables().find(tableName) == database_->GetTables().end())
    {
        throw TableNotFoundFromException(tableName);
    }

    std::unordered_map<std::string, DataType> addColumns;
    std::unordered_set<std::string> dropColumns;
    std::unordered_map<std::string, DataType> alterColumns;

    for (auto& entry : ctx->alterTableEntries()->alterTableEntry())
    {
        if (entry->addColumn())
        {
            auto addColumnContext = entry->addColumn();
            DataType addColumnDataType = GetDataTypeFromString(addColumnContext->DATATYPE()->getText());
            std::string addColumnName = addColumnContext->column()->getText();
            TrimDelimitedIdentifier(addColumnName);

            if (database_->GetTables().at(tableName).GetColumns().find(addColumnName) !=
                    database_->GetTables().at(tableName).GetColumns().end() ||
                addColumns.find(addColumnName) != addColumns.end())
            {
                throw ColumnAlreadyExistsException(addColumnName);
            }

            addColumns.insert({addColumnName, addColumnDataType});
        }
        else if (entry->dropColumn())
        {
            auto dropColumnContext = entry->dropColumn();
            std::string dropColumnName = dropColumnContext->column()->getText();
            TrimDelimitedIdentifier(dropColumnName);

            if (database_->GetTables().at(tableName).GetColumns().find(dropColumnName) ==
                    database_->GetTables().at(tableName).GetColumns().end() ||
                dropColumns.find(dropColumnName) != dropColumns.end())
            {
                throw ColumnNotFoundException(dropColumnName);
            }

            dropColumns.insert({dropColumnName});
        }
        else if (entry->alterColumn())
        {
            auto alterColumnContext = entry->alterColumn();
            DataType alterColumnDataType = GetDataTypeFromString(alterColumnContext->DATATYPE()->getText());
            std::string alterColumnName = alterColumnContext->column()->getText();
            TrimDelimitedIdentifier(alterColumnName);

            if (database_->GetTables().at(tableName).GetColumns().find(alterColumnName) ==
                database_->GetTables().at(tableName).GetColumns().end())
            {
                throw ColumnNotFoundException(alterColumnName);
            }

            if (alterColumns.find(alterColumnName) != alterColumns.end())
            {
                throw AlreadyModifiedColumnException();
            }

            alterColumns.insert({alterColumnName, alterColumnDataType});
        }
    }

    dispatcher_.AddAlterTableFunction();
    dispatcher_.AddArgument<const std::string&>(tableName);

    dispatcher_.AddArgument<int32_t>(addColumns.size());
    for (auto& addColumn : addColumns)
    {
        dispatcher_.AddArgument<const std::string&>(addColumn.first);
        dispatcher_.AddArgument<int32_t>(static_cast<int32_t>(addColumn.second));
    }

    dispatcher_.AddArgument<int32_t>(dropColumns.size());
    for (auto& dropColumn : dropColumns)
    {
        dispatcher_.AddArgument<const std::string&>(dropColumn);
    }

    dispatcher_.AddArgument<int32_t>(alterColumns.size());
    for (auto& alterColumn : alterColumns)
    {
        dispatcher_.AddArgument<const std::string&>(alterColumn.first);
        dispatcher_.AddArgument<int32_t>(static_cast<int32_t>(alterColumn.second));
    }
}

void GpuSqlListener::exitSqlCreateIndex(GpuSqlParser::SqlCreateIndexContext* ctx)
{
    std::string indexName = ctx->indexName()->getText();
    TrimDelimitedIdentifier(indexName);

    std::string tableName = ctx->table()->getText();
    TrimDelimitedIdentifier(tableName);

    if (database_->GetTables().find(tableName) == database_->GetTables().end())
    {
        throw TableNotFoundFromException(tableName);
    }

    if (database_->GetTables().at(tableName).GetSize() > 0)
    {
        throw TableIsFilledException();
    }

    // check if index already exists

    std::vector<std::string> indexColumns;

    for (auto& column : ctx->indexColumns()->column())
    {
        std::string indexColumnName = column->getText();
        TrimDelimitedIdentifier(indexColumnName);

        if (database_->GetTables().at(tableName).GetColumns().find(indexColumnName) ==
            database_->GetTables().at(tableName).GetColumns().end())
        {
            throw ColumnNotFoundException(indexColumnName);
        }

        DataType indexColumnDataType =
            database_->GetTables().at(tableName).GetColumns().at(indexColumnName)->GetColumnType();
        if (indexColumnDataType == DataType::COLUMN_POINT || indexColumnDataType == DataType::COLUMN_POLYGON)
        {
            throw IndexColumnDataTypeException(indexColumnName, indexColumnDataType);
        }

        if (std::find(indexColumns.begin(), indexColumns.end(), indexColumnName) != indexColumns.end())
        {
            throw ColumnAlreadyExistsInIndexException(indexColumnName);
        }
        indexColumns.push_back(indexColumnName);
    }

    dispatcher_.AddCreateIndexFunction();

    dispatcher_.AddArgument<const std::string&>(indexName);
    dispatcher_.AddArgument<const std::string&>(tableName);

    dispatcher_.AddArgument<int32_t>(indexColumns.size());
    for (auto& indexColumn : indexColumns)
    {
        dispatcher_.AddArgument<const std::string&>(indexColumn);
    }
}

void GpuSqlListener::enterOrderByColumns(GpuSqlParser::OrderByColumnsContext* ctx)
{
    insideOrderBy_ = true;
}

void GpuSqlListener::exitOrderByColumns(GpuSqlParser::OrderByColumnsContext* ctx)
{
    for (auto& orderByColumn : orderByColumns_)
    {
        std::string orderByColName = orderByColumn.first;
        DataType dataType = std::get<0>(orderByColumn.second);

        dispatcher_.AddArgument<const std::string&>(orderByColName);
        dispatcher_.AddArgument<bool>(false);
        dispatcher_.AddOrderByReconstructFunction(dataType);
    }

    for (auto& returnColumn : returnColumns_)
    {
        std::string returnColName = returnColumn.first;
        DataType dataType = std::get<0>(returnColumn.second);

        dispatcher_.AddArgument<const std::string&>(returnColName);
        dispatcher_.AddArgument<bool>(true);
        dispatcher_.AddOrderByReconstructFunction(dataType);
    }

    insideOrderBy_ = false;
    dispatcher_.AddFreeOrderByTableFunction();
    dispatcher_.AddOrderByReconstructRetAllBlocksFunction();
}


void GpuSqlListener::exitOrderByColumn(GpuSqlParser::OrderByColumnContext* ctx)
{
    std::pair<std::string, DataType> arg = StackTopAndPop();
    std::string orderByColName = std::get<0>(arg);
    DataType orderByDataType = std::get<1>(arg);

    if (orderByDataType < DataType::COLUMN_INT)
    {
        if (orderByDataType != DataType::CONST_INT && orderByDataType != DataType::CONST_LONG)
        {
            throw OrderByInvalidColumnException(orderByColName);
        }
        else
        {
            int64_t value = std::stoll(orderByColName);

            if (columnNumericAliasContexts_.find(value) != columnNumericAliasContexts_.end() && !insideAlias_)
            {
                WalkAliasExpression(value);
                arg = StackTopAndPop();
                orderByColName = std::get<0>(arg);
                orderByDataType = std::get<1>(arg);
            }
            else
            {
                throw OrderByInvalidColumnException(orderByColName);
            }
        }
    }

    if (orderByColumns_.find(orderByColName) != orderByColumns_.end())
    {
        throw OrderByColumnAlreadyReferencedException(orderByColName);
    }

    DataType dataType = std::get<1>(arg);
    OrderBy::Order order = OrderBy::Order::ASC;

    if (ctx->DIR())
    {
        std::string dir = ctx->DIR()->getText();
        StringToUpper(dir);
        if (dir == "DESC")
        {
            order = OrderBy::Order::DESC;
        }
    }

    dispatcher_.AddArgument<const std::string&>(orderByColName);
    dispatcher_.AddArgument<int32_t>(static_cast<int32_t>(order));
    dispatcher_.AddArgument<int32_t>(orderByColumnIndex_++);
    dispatcher_.AddOrderByFunction(dataType);

    orderByColumns_.insert({orderByColName, {dataType, order}});
}

/// Method that executes on exit of INSERT INTO command
/// Generates insert into operation
/// Checks if table with given name exists
/// Checks if table.column (dot notation) is not used
/// Checks if column with given name exists
/// Checks if the same column is not referenced multiple times
/// Checks if same number of values and columns is provided
/// <param name="ctx">Sql Insert Into context</param>
void GpuSqlListener::exitSqlInsertInto(GpuSqlParser::SqlInsertIntoContext* ctx)
{
    std::string table = ctx->table()->getText();
    TrimDelimitedIdentifier(table);

    if (database_->GetTables().find(table) == database_->GetTables().end())
    {
        throw TableNotFoundFromException(table);
    }
    auto& tab = database_->GetTables().at(table);

    std::vector<std::pair<std::string, DataType>> columns;
    std::vector<std::string> values;
    std::vector<bool> isValueNull;
    for (auto& insertIntoColumn : ctx->insertIntoColumns()->columnId())
    {
        if (insertIntoColumn->table())
        {
            throw ColumnNotFoundException(insertIntoColumn->getText());
        }

        std::string column = insertIntoColumn->column()->getText();
        TrimDelimitedIdentifier(column);

        if (tab.GetColumns().find(column) == tab.GetColumns().end())
        {
            throw ColumnNotFoundException(column);
        }
        DataType columnDataType = tab.GetColumns().at(column).get()->GetColumnType();
        std::pair<std::string, DataType> columnPair = std::make_pair(column, columnDataType);

        if (std::find(columns.begin(), columns.end(), columnPair) != columns.end())
        {
            throw InsertIntoException(column);
        }
        columns.push_back(columnPair);
    }

    for (auto& value : ctx->insertIntoValues()->columnValue())
    {
        auto start = value->start->getStartIndex();
        auto stop = value->stop->getStopIndex();
        antlr4::misc::Interval interval(start, stop);
        std::string valueText = value->start->getInputStream()->getText(interval);
        values.push_back(valueText);
        isValueNull.push_back(value->NULLLIT() != nullptr);
    }

    if (columns.size() != values.size())
    {
        throw NotSameAmoutOfValuesException();
    }

    for (auto& column : tab.GetColumns())
    {
        std::string columnName = column.first;
        DataType columnDataType = column.second.get()->GetColumnType();
        std::pair<std::string, DataType> columnPair = std::make_pair(columnName, columnDataType);

        dispatcher_.AddInsertIntoFunction(columnDataType);


        bool hasValue = std::find(columns.begin(), columns.end(), columnPair) != columns.end();
        if (hasValue)
        {
            int valueIndex = std::find(columns.begin(), columns.end(), columnPair) - columns.begin();
            hasValue &= !isValueNull[valueIndex];
        }
        dispatcher_.AddArgument<const std::string&>(columnName);
        dispatcher_.AddArgument<bool>(hasValue);

        if (hasValue)
        {
            int valueIndex = std::find(columns.begin(), columns.end(), columnPair) - columns.begin();
            CudaLogBoost::getInstance(CudaLogBoost::info)
                << values[valueIndex].c_str() << " " << columnName << '\n';
            PushArgument(values[valueIndex].c_str(),
                         static_cast<DataType>(static_cast<int>(columnDataType) - DataType::COLUMN_INT));
        }
    }
    dispatcher_.AddArgument<const std::string&>(table);
    dispatcher_.AddInsertIntoDoneFunction();
}

/// Method that executes on exit of LIMIT clause
/// Sets the row limit count
/// <param name="ctx">Limit context</param>
void GpuSqlListener::exitLimit(GpuSqlParser::LimitContext* ctx)
{
    ResultLimit = std::stoi(ctx->getText());
}

/// Method that executes on exit of OFFSET clause
/// Sets the row offset count
/// <param name="ctx">Offset context</param>
void GpuSqlListener::exitOffset(GpuSqlParser::OffsetContext* ctx)
{
    ResultOffset = std::stoi(ctx->getText());
}

void GpuSqlListener::ExtractColumnAliasContexts(GpuSqlParser::SelectColumnsContext* ctx)
{
    for (int32_t i = 0; i < ctx->selectColumn().size(); i++)
    {
        auto selectColumn = ctx->selectColumn()[i];
        if (selectColumn->alias())
        {
            std::string alias = selectColumn->alias()->getText();
            if (columnAliasContexts_.find(alias) != columnAliasContexts_.end())
            {
                throw AliasRedefinitionException(alias);
            }
            columnAliasContexts_.insert({alias, selectColumn->expression()});
        }

        columnNumericAliasContexts_.insert({i + 1, selectColumn->expression()});
    }
}

/// Method that executes on exit of integer literal (10, 20, 5, ...)
/// Infers token data type (int or long)
/// Pushes the literal token to parser stack along with its inferred data type (int or long)
/// <param name="ctx">Int Literal context</param>
void GpuSqlListener::exitIntLiteral(GpuSqlParser::IntLiteralContext* ctx)
{
    std::string token = ctx->getText();

    if (IsLong(token))
    {
        parserStack_.push(std::make_pair(token, DataType::CONST_LONG));
    }
    else
    {
        parserStack_.push(std::make_pair(token, DataType::CONST_INT));
    }
}

/// Method that executes on exit of decimal literal (10.5, 20.6, 5.2, ...)
/// Infers token data type (float or double)
/// Pushes the literal token to parser stack along with its inferred data type (float or double)
/// <param name="ctx">Decimal Literal context</param>
void GpuSqlListener::exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext* ctx)
{
    std::string token = ctx->getText();
    if (IsDouble(token))
    {
        parserStack_.push(std::make_pair(token, DataType::CONST_DOUBLE));
    }
    else
    {
        parserStack_.push(std::make_pair(token, DataType::CONST_FLOAT));
    }
}

/// Method that executes on exit of string literal ("Hello", ...)
/// Pushes the literal token to parser stack along with its data type
/// <param name="ctx">String Literal context</param>
void GpuSqlListener::exitStringLiteral(GpuSqlParser::StringLiteralContext* ctx)
{
    parserStack_.push(std::make_pair(ctx->getText(), DataType::CONST_STRING));
}

/// Method that executes on exit of boolean literal (True, False)
/// Pushes the literal token to parser stack along with its data type
/// <param name="ctx">Boolean Literal context</param>
void GpuSqlListener::exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext* ctx)
{
    parserStack_.push(std::make_pair(ctx->getText(), DataType::CONST_INT8_T));
}

/// Method that executes on exit of column name reference (colInt1, tableA.colInt, ...)
/// Validates column existance and generates its full name in dot notation (table.column)
/// Pushes the token to the parser stack
/// Fills link table (used for gpu where dispatch) with column name and ordinal number of
/// its first appearance in the AST traversal
/// <param name="ctx">Var Reference context</param>
void GpuSqlListener::exitVarReference(GpuSqlParser::VarReferenceContext* ctx)
{
    std::string colName = ctx->columnId()->getText();

    if (columnAliasContexts_.find(colName) != columnAliasContexts_.end() && !insideAlias_ && colName != currentExpressionAlias_)
    {
        WalkAliasExpression(colName);
        return;
    }

    std::pair<std::string, DataType> tableColumnData = GenerateAndValidateColumnName(ctx->columnId());
    const DataType columnType = std::get<1>(tableColumnData);
    const std::string tableColumn = std::get<0>(tableColumnData);

    parserStack_.push(std::make_pair(tableColumn, columnType));

    if (GpuSqlDispatcher::linkTable.find(tableColumn) == GpuSqlDispatcher::linkTable.end())
    {
        GpuSqlDispatcher::linkTable.insert({tableColumn, linkTableIndex_++});
    }

    if (groupByColumns_.find(tableColumnData) != groupByColumns_.end() && insideSelectColumn_)
    {
        isSelectColumnValid_ = true;
    }
}

/// Method that executes on exit of date time literal in format of yyyy-mm-dd hh:mm:ss
/// Converts the literal to epoch time and pushes is to parser stack as LONG data type literal
/// <param name="ctx">Date Time Literal context</param>
void GpuSqlListener::exitDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext* ctx)
{
    auto start = ctx->start->getStartIndex();
    auto stop = ctx->stop->getStopIndex();
    antlr4::misc::Interval interval(start, stop);
    std::string dateValue = ctx->start->getInputStream()->getText(interval);

    std::time_t epochTime = DateToLong(dateValue);

    parserStack_.push(std::make_pair(std::to_string(epochTime), DataType::CONST_LONG));
}

time_t GpuSqlListener::DateToLong(std::string dateString)
{
    if (dateString.front() == '\'' && dateString.back() == '\'')
    {
        dateString = dateString.substr(1, dateString.size() - 2);
    }
    if (dateString.size() <= 10)
    {
        dateString += " 00:00:00";
    }

    std::tm t;
    std::istringstream ss(dateString);
    ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");
    return std::mktime(&t);
}

/// Method that executes on exit of PI() literal (3.1415926)
/// Pushes pi literal to stack as float data type
/// <param name="ctx">Pi Literal context</param>
void GpuSqlListener::exitPiLiteral(GpuSqlParser::PiLiteralContext* ctx)
{
    parserStack_.push(std::make_pair(std::to_string(pi()), DataType::CONST_FLOAT));
    shortColumnNames_.insert({std::to_string(pi()), ctx->PI()->getText()});
}

/// Method that executes on exit of NOW() literal (current date time)
/// Converts the current date time to epoch time and pushes it to parser stack as long data type
/// literal <param name="ctx">Now Literal context</param>
void GpuSqlListener::exitNowLiteral(GpuSqlParser::NowLiteralContext* ctx)
{
    std::time_t epochTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    parserStack_.push(std::make_pair(std::to_string(epochTime), DataType::CONST_LONG));
    // Bug case if column exists with the same name as long reprsentation of NOW()
    shortColumnNames_.insert({std::to_string(epochTime), ctx->NOW()->getText()});
}

/// Method that executes on exit of polygon and point literals
/// Infers the type of literal (polygon or point) and pushes it to parser stack
/// <param name="ctx">Geo Reference context</param>
void GpuSqlListener::exitGeoReference(GpuSqlParser::GeoReferenceContext* ctx)
{
    auto start = ctx->start->getStartIndex();
    auto stop = ctx->stop->getStopIndex();
    antlr4::misc::Interval interval(start, stop);
    std::string geoValue = ctx->geometry()->start->getInputStream()->getText(interval);

    if (IsPolygon(geoValue))
    {
        parserStack_.push(std::make_pair(geoValue, DataType::CONST_POLYGON));
    }
    else if (IsPoint(geoValue))
    {
        parserStack_.push(std::make_pair(geoValue, DataType::CONST_POINT));
    }
}

/// Method used to generate column name in dot notation (table.column) and validation of its name
/// Checks for table existance if provided
/// Infers table name if not provided and checks for column ambiguity between tables
/// Retrieves the column data type
/// Checks if column is used in aggregation or group by clause if its present
/// <param name="ctx">Column Id context</param>
/// <returns="tableColumnPair">Column name in dot notation (table.column)</returns>
std::pair<std::string, DataType> GpuSqlListener::GenerateAndValidateColumnName(GpuSqlParser::ColumnIdContext* ctx)
{
    std::string table;
    std::string column;

    std::string col = ctx->column()->getText();
    TrimDelimitedIdentifier(col);

    if (ctx->table())
    {
        table = ctx->table()->getText();
        TrimDelimitedIdentifier(table);
        column = ctx->column()->getText();
        TrimDelimitedIdentifier(column);

        if (tableAliases_.find(table) != tableAliases_.end())
        {
            table = tableAliases_.at(table);
        }

        if (loadedTables_.find(table) == loadedTables_.end())
        {
            throw TableNotFoundFromException(table);
        }
        if (database_->GetTables().at(table).GetColumns().find(column) ==
            database_->GetTables().at(table).GetColumns().end())
        {
            throw ColumnNotFoundException(column);
        }
        shortColumnNames_.insert({table + "." + column, table + "." + column});
    }
    else
    {
        int uses = 0;
        for (auto& tab : loadedTables_)
        {
            if (database_->GetTables().at(tab).GetColumns().find(col) !=
                database_->GetTables().at(tab).GetColumns().end())
            {
                table = tab;
                column = col;
                uses++;
            }
            if (uses > 1)
            {
                throw ColumnAmbiguityException(col);
            }
        }
        if (column.empty())
        {
            throw ColumnNotFoundException(col);
        }

        shortColumnNames_.insert({table + "." + column, column});
    }

    std::string tableColumn = table + "." + column;
    DataType columnType = database_->GetTables().at(table).GetColumns().at(column)->GetColumnType();

    std::pair<std::string, DataType> tableColumnPair = std::make_pair(tableColumn, columnType);

    if (insideGroupBy_ && originalGroupByColumns_.find(tableColumnPair) == originalGroupByColumns_.end())
    {
        originalGroupByColumns_.insert(tableColumnPair);
    }

    if (usingGroupBy_ && !insideAgg_ &&
        originalGroupByColumns_.find(tableColumnPair) == originalGroupByColumns_.end())
    {
        throw ColumnGroupByException(column);
    }

    return tableColumnPair;
}

void GpuSqlListener::WalkAliasExpression(const std::string& alias)
{
    antlr4::tree::ParseTreeWalker walker;
    insideAlias_ = true;
    walker.walk(this, columnAliasContexts_.at(alias));
    insideAlias_ = false;
}

void GpuSqlListener::WalkAliasExpression(const int64_t alias)
{
    antlr4::tree::ParseTreeWalker walker;
    insideAlias_ = true;
    walker.walk(this, columnNumericAliasContexts_.at(alias));
    insideAlias_ = false;
}

void GpuSqlListener::LockAliasRegisters()
{
    for (auto& aliasContext : columnAliasContexts_)
    {
        std::string reg = "$" + aliasContext.second->getText();
        dispatcher_.AddArgument<const std::string&>(reg);
        dispatcher_.AddLockRegisterFunction();
    }
}

/// Method used to pop contnt from parser stack
/// <returns="value">Pair of content string and contnet data type</returns>
std::pair<std::string, DataType> GpuSqlListener::StackTopAndPop()
{
    std::pair<std::string, DataType> value = parserStack_.top();
    parserStack_.pop();
    return value;
}

/// Method used to push temp results to parser stack
/// <param name="reg">String representing the literal</param>
/// <param name="type">Data type of the literal</param>
void GpuSqlListener::PushTempResult(std::string reg, DataType type)
{
    parserStack_.push(std::make_pair(reg, type));
}

/// Method used to push argument to dispatcher argument queue
/// Converts the string token to actual numeric data if its numeric literal
/// Keeps it as a string in case of polygons, points and column names
/// <param name="token">String representing the literal</param>
/// <param name="type">Data type of the literal</param>
void GpuSqlListener::PushArgument(const char* token, DataType dataType)
{
    switch (dataType)
    {
    case DataType::CONST_INT:
        dispatcher_.AddArgument<int32_t>(std::stoi(token));
        break;
    case DataType::CONST_LONG:
        try
        {
            dispatcher_.AddArgument<int64_t>(std::stoll(token));
        }
        catch(const std::invalid_argument& e)
        {
            dispatcher_.AddArgument<int64_t>(DateToLong(token));
        }
        break;
    case DataType::CONST_FLOAT:
        dispatcher_.AddArgument<float>(std::stof(token));
        break;
    case DataType::CONST_DOUBLE:
        dispatcher_.AddArgument<double>(std::stod(token));
        break;
    case DataType::CONST_STRING:
    {
        std::string str(token);
        std::string strTrimmed = str.substr(1, str.length() - 2);
        dispatcher_.AddArgument<const std::string&>(strTrimmed);
    }
    break;
    case DataType::CONST_POINT:
    case DataType::CONST_POLYGON:
    case DataType::COLUMN_INT:
    case DataType::COLUMN_LONG:
    case DataType::COLUMN_FLOAT:
    case DataType::COLUMN_DOUBLE:
    case DataType::COLUMN_POINT:
    case DataType::COLUMN_POLYGON:
    case DataType::COLUMN_STRING:
    case DataType::COLUMN_INT8_T:
        dispatcher_.AddArgument<const std::string&>(token);
        break;
    case DataType::CONST_INT8_T:
    {
        std::string booleanToken(token);
        StringToUpper(booleanToken);
        dispatcher_.AddArgument<int8_t>(booleanToken == "TRUE" ? 1 : 0);
    }
    break;
    case DataType::DATA_TYPE_SIZE:
    case DataType::CONST_ERROR:
        break;
    }
}

/// Checks if given integer literal is long
/// <param name="value">String representing the literal</param>
/// <returns="isLong">True if literal is long</returns>
bool GpuSqlListener::IsLong(const std::string& value)
{
    try
    {
        std::stoi(value);
    }
    catch (std::out_of_range&)
    {
        std::stoll(value);
        return true;
    }
    return false;
}

/// Checks if given decimal literal is double
/// <param nameDouble">True if literal is double</returns>
bool GpuSqlListener::IsDouble(const std::string& value)
{
    try
    {
        std::stof(value);
    }
    catch (std::out_of_range&)
    {
        std::stod(value);
        return true;
    }
    return false;
}


/// Checks if given geo literal is point
/// <param name="value">String representing the literal</param>
/// <returns="isPoints">True if literal is point</returns>
bool GpuSqlListener::IsPoint(const std::string& value)
{
    return (value.find("POINT") == 0);
}

/// Checks if given geo literal is polygon
/// <param name="value">String representing the literal</param>
/// <returns="isPolygon">True if literal is polygon</returns>
bool GpuSqlListener::IsPolygon(const std::string& value)
{
    return (value.find("POLYGON") == 0);
}

/// Converts string to uppercase
/// <param name="value">String to convert</param>
void GpuSqlListener::StringToUpper(std::string& str)
{
    for (auto& c : str)
    {
        c = toupper(c);
    }
}

void GpuSqlListener::TrimDelimitedIdentifier(std::string& str)
{
    if (str.front() == '[' && str.back() == ']' && str.size() > 2)
    {
        str.erase(0, 1);
        str.erase(str.size() - 1);
    }
}

/// Defines return data type for binary operation
/// If operand type is a constant data type its converted to column data type
/// Data type with higher ordinal number (the ordering is designed with this feature in mind) is
/// chosen <param name="left">Left operand data type</param> <param name="right">Right operand data
/// type</param> <returns="result">Operation result data type</returns>
DataType GpuSqlListener::GetReturnDataType(DataType left, DataType right)
{
    if (right < DataType::COLUMN_INT)
    {
        right = static_cast<DataType>(right + DataType::COLUMN_INT);
    }
    if (left < DataType::COLUMN_INT)
    {
        left = static_cast<DataType>(left + DataType::COLUMN_INT);
    }
    DataType result = std::max<DataType>(left, right);

    return result;
}

/// Defines return data type for unary operation
/// If operand type is a constant data type its converted to column data type
/// <param name="operand">Operand data type</param>
/// <returns="result">Operation result data type</returns>
DataType GpuSqlListener::GetReturnDataType(DataType operand)
{
    if (operand < DataType::COLUMN_INT)
    {
        return static_cast<DataType>(operand + DataType::COLUMN_INT);
    }
    return operand;
}

DataType GpuSqlListener::GetDataTypeFromString(const std::string& dataType)
{
    return ::GetColumnDataTypeFromString(dataType);
}

void GpuSqlListener::TrimReg(std::string& reg)
{
    if (reg.front() == '$')
    {
        reg.erase(reg.begin());
    }
    else if (shortColumnNames_.find(reg) != shortColumnNames_.end())
    {
        reg = shortColumnNames_.at(reg);
    }
}
