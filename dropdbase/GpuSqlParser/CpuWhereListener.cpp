#include "CpuWhereListener.h"
#include <boost/algorithm/string.hpp>
#include "../ColumnBase.h"

constexpr float pi()
{
    return 3.1415926f;
}

CpuWhereListener::CpuWhereListener(const std::shared_ptr<Database>& database, CpuSqlDispatcher& dispatcher)
: database_(database), dispatcher_(dispatcher), insideAlias_(false)
{
}

void CpuWhereListener::exitBinaryOperation(GpuSqlParser::BinaryOperationContext* ctx)
{
    std::pair<std::string, DataType> right = StackTopAndPop();
    std::pair<std::string, DataType> left = StackTopAndPop();

    std::string op = ctx->op->getText();
    StringToUpper(op);

    DataType rightOperandType = std::get<1>(right);
    DataType leftOperandType = std::get<1>(left);

    std::string rightOperand = std::get<0>(right);
    std::string leftOperand = std::get<0>(left);

    PushArgument(leftOperand.c_str(), leftOperandType);
    PushArgument(rightOperand.c_str(), rightOperandType);

    std::string reg;
    TrimReg(rightOperand);
    TrimReg(leftOperand);

    DataType returnDataType;

    switch (ctx->op->getType())
    {
    case GpuSqlLexer::GREATER:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::LESS:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::GREATEREQ:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::LESSEQ:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::EQUALS:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::NOTEQUALS:
    case GpuSqlLexer::NOTEQUALS_GT_LT:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::AND:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::OR:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::ASTERISK:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::DIVISION:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::PLUS:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::MINUS:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::MODULO:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::BIT_OR:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::BIT_AND:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::XOR:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::L_SHIFT:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::R_SHIFT:
        reg = "$" + leftOperand + op + rightOperand;
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::POINT:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        returnDataType = DataType::COLUMN_POINT;
        break;
    case GpuSqlLexer::GEO_CONTAINS:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::GEO_INTERSECT:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        returnDataType = DataType::COLUMN_POLYGON;
        break;
    case GpuSqlLexer::GEO_UNION:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        returnDataType = DataType::COLUMN_POLYGON;
        break;
    case GpuSqlLexer::LOG:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::POW:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::ROOT:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        returnDataType = GetReturnDataType(leftOperandType, rightOperandType);
        break;
    case GpuSqlLexer::ATAN2:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        returnDataType = GetReturnDataType(DataType::COLUMN_FLOAT);
        break;
    case GpuSqlLexer::CONCAT:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::LEFT:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::RIGHT:
        reg = "$" + op + "(" + leftOperand + "," + rightOperand + ")";
        returnDataType = DataType::COLUMN_STRING;
        break;
    default:
        break;
    }
    dispatcher_.AddBinaryOperation(leftOperandType, rightOperandType, ctx->op->getType());

    PushArgument(reg.c_str(), returnDataType);
    PushTempResult(reg, returnDataType);
}

void CpuWhereListener::exitTernaryOperation(GpuSqlParser::TernaryOperationContext* ctx)
{
}

void CpuWhereListener::exitUnaryOperation(GpuSqlParser::UnaryOperationContext* ctx)
{
    std::pair<std::string, DataType> arg = StackTopAndPop();

    std::string op = ctx->op->getText();
    StringToUpper(op);

    std::string operand = std::get<0>(arg);
    DataType operandType = std::get<1>(arg);

    PushArgument(operand.c_str(), operandType);

    DataType returnDataType;

    std::string reg;
    TrimReg(operand);

    switch (ctx->op->getType())
    {
    case GpuSqlLexer::LOGICAL_NOT:
        reg = "$" + op + operand;
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::ISNULL:
        reg = "$" + op + operand;
        if (operandType < DataType::COLUMN_INT)
        {
            throw NullMaskOperationInvalidOperandException();
        }
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::ISNOTNULL:
        reg = "$" + op + operand;
        if (operandType < DataType::COLUMN_INT)
        {
            throw NullMaskOperationInvalidOperandException();
        }
        returnDataType = DataType::COLUMN_INT8_T;
        break;
    case GpuSqlLexer::MINUS:
        reg = "$" + op + operand;
        returnDataType = GetReturnDataType(operandType);
        break;
    case GpuSqlLexer::YEAR:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = COLUMN_INT;
        break;
    case GpuSqlLexer::MONTH:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = COLUMN_INT;
        break;
    case GpuSqlLexer::DAY:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = COLUMN_INT;
        break;
    case GpuSqlLexer::HOUR:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = COLUMN_INT;
        break;
    case GpuSqlLexer::MINUTE:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = COLUMN_INT;
        break;
    case GpuSqlLexer::SECOND:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = COLUMN_INT;
        break;
    case GpuSqlLexer::ABS:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = GetReturnDataType(operandType);
        break;
    case GpuSqlLexer::SIN:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::COS:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::TAN:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::COT:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::ASIN:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::ACOS:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::ATAN:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::LOG10:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::LOG:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::EXP:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::SQRT:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::SQUARE:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::SIGN:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_INT;
        break;
    case GpuSqlLexer::ROUND:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::FLOOR:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::CEIL:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_FLOAT;
        break;
    case GpuSqlLexer::LTRIM:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::RTRIM:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::LOWER:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::UPPER:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::REVERSE:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_STRING;
        break;
    case GpuSqlLexer::LEN:
        reg = "$" + op + "(" + operand + ")";
        returnDataType = DataType::COLUMN_INT;
        break;
    default:
        break;
    }
    dispatcher_.AddUnaryOperation(operandType, ctx->op->getType());

    PushArgument(reg.c_str(), returnDataType);
    PushTempResult(reg, returnDataType);
}

void CpuWhereListener::exitCastOperation(GpuSqlParser::CastOperationContext* ctx)
{
    std::pair<std::string, DataType> arg = StackTopAndPop();

    std::string operand = std::get<0>(arg);
    DataType operandType = std::get<1>(arg);

    PushArgument(operand.c_str(), operandType);
    std::string castTypeStr = ctx->DATATYPE()->getText();
    StringToUpper(castTypeStr);
    DataType castType = GetDataTypeFromString(castTypeStr);

    dispatcher_.AddCastOperation(operandType, castType, castTypeStr);

    TrimReg(operand);
    std::string reg = "$CAST(" + operand + "AS" + castTypeStr + ")";

    PushArgument(reg.c_str(), castType);
    PushTempResult(reg, castType);
}

void CpuWhereListener::exitIntLiteral(GpuSqlParser::IntLiteralContext* ctx)
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

void CpuWhereListener::exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext* ctx)
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

void CpuWhereListener::exitStringLiteral(GpuSqlParser::StringLiteralContext* ctx)
{
    parserStack_.push(std::make_pair(ctx->getText(), DataType::CONST_STRING));
}

void CpuWhereListener::exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext* ctx)
{
    parserStack_.push(std::make_pair(ctx->getText(), DataType::CONST_INT8_T));
}

void CpuWhereListener::exitGeoReference(GpuSqlParser::GeoReferenceContext* ctx)
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

void CpuWhereListener::exitVarReference(GpuSqlParser::VarReferenceContext* ctx)
{
    std::string colName = ctx->columnId()->getText();

    if (columnAliasContexts_.find(colName) != columnAliasContexts_.end() && !insideAlias_)
    {
        WalkAliasExpression(colName);
        return;
    }

    std::pair<std::string, DataType> tableColumnData = GenerateAndValidateColumnName(ctx->columnId());
    const DataType columnType = std::get<1>(tableColumnData);
    const std::string tableColumn = std::get<0>(tableColumnData);

    parserStack_.push(std::make_pair(tableColumn, columnType));
}

void CpuWhereListener::exitDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext* ctx)
{
    auto start = ctx->start->getStartIndex();
    auto stop = ctx->stop->getStopIndex();
    antlr4::misc::Interval interval(start, stop);
    std::string dateValue = ctx->start->getInputStream()->getText(interval);
    dateValue = dateValue.substr(1, dateValue.size() - 2);

    if (dateValue.size() <= 10)
    {
        dateValue += " 00:00:00";
    }

    std::tm t;
    std::istringstream ss(dateValue);
    ss >> std::get_time(&t, "%Y-%m-%d %H:%M:%S");
    std::time_t epochTime = std::mktime(&t);

    parserStack_.push(std::make_pair(std::to_string(epochTime), DataType::CONST_LONG));
}

void CpuWhereListener::exitPiLiteral(GpuSqlParser::PiLiteralContext* ctx)
{
    parserStack_.push(std::make_pair(std::to_string(pi()), DataType::CONST_FLOAT));
    shortColumnNames_.insert({std::to_string(pi()), ctx->PI()->getText()});
}

void CpuWhereListener::exitNowLiteral(GpuSqlParser::NowLiteralContext* ctx)
{
    std::time_t epochTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    parserStack_.push(std::make_pair(std::to_string(epochTime), DataType::CONST_LONG));
    shortColumnNames_.insert({std::to_string(epochTime), ctx->NOW()->getText()});
}

void CpuWhereListener::exitWhereClause(GpuSqlParser::WhereClauseContext* ctx)
{
    std::pair<std::string, DataType> arg = StackTopAndPop();
    dispatcher_.AddArgument<const std::string&>(std::get<0>(arg));
    dispatcher_.AddWhereResultFunction(std::get<1>(arg));
}

void CpuWhereListener::exitFromTables(GpuSqlParser::FromTablesContext* ctx)
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

        if (fromTable->alias())
        {
            std::string alias = fromTable->alias()->getText();
            TrimDelimitedIdentifier(alias);
            if (tableAliases_.find(alias) != tableAliases_.end())
            {
                throw AliasRedefinitionException();
            }
            tableAliases_.insert({alias, table});
        }
    }
}

void CpuWhereListener::ExtractColumnAliasContexts(GpuSqlParser::SelectColumnsContext* ctx)
{
    for (auto& selectColumn : ctx->selectColumn())
    {
        if (selectColumn->alias())
        {
            std::string alias = selectColumn->alias()->getText();
            if (columnAliasContexts_.find(alias) != columnAliasContexts_.end())
            {
                throw AliasRedefinitionException();
            }
            columnAliasContexts_.insert({alias, selectColumn->expression()});
        }
    }
}

void CpuWhereListener::PushArgument(const char* token, DataType dataType)
{
    switch (dataType)
    {
    case DataType::CONST_INT:
        dispatcher_.AddArgument<int32_t>(std::stoi(token));
        break;
    case DataType::CONST_LONG:
        dispatcher_.AddArgument<int64_t>(std::stoll(token));
        break;
    case DataType::CONST_FLOAT:
        dispatcher_.AddArgument<float>(std::stof(token));
        break;
    case DataType::CONST_DOUBLE:
        dispatcher_.AddArgument<double>(std::stod(token));
        break;
    case DataType::CONST_POINT:
    case DataType::CONST_POLYGON:
    case DataType::CONST_STRING:
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
    case DataType::DATA_TYPE_SIZE:
    case DataType::CONST_ERROR:
        break;
    }
}

std::pair<std::string, DataType> CpuWhereListener::StackTopAndPop()
{
    std::pair<std::string, DataType> value = parserStack_.top();
    parserStack_.pop();
    return value;
}

void CpuWhereListener::StringToUpper(std::string& str)
{
    for (auto& c : str)
    {
        c = toupper(c);
    }
}

void CpuWhereListener::PushTempResult(std::string reg, DataType type)
{
    parserStack_.push(std::make_pair(reg, type));
}

bool CpuWhereListener::IsLong(const std::string& value)
{
    try
    {
        std::stoi(value);
    }
    catch (std::out_of_range& e)
    {
        std::stoll(value);
        return true;
    }
    return false;
}

bool CpuWhereListener::IsDouble(const std::string& value)
{
    try
    {
        std::stof(value);
    }
    catch (std::out_of_range& e)
    {
        std::stod(value);
        return true;
    }
    return false;
}

bool CpuWhereListener::IsPoint(const std::string& value)
{
    return (value.find("POINT") == 0);
}

bool CpuWhereListener::IsPolygon(const std::string& value)
{
    return (value.find("POLYGON") == 0);
}

void CpuWhereListener::TrimDelimitedIdentifier(std::string& str)
{
    if (str.front() == '[' && str.back() == ']' && str.size() > 2)
    {
        str.erase(0, 1);
        str.erase(str.size() - 1);
    }
}

DataType CpuWhereListener::GetReturnDataType(DataType left, DataType right)
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

DataType CpuWhereListener::GetReturnDataType(DataType operand)
{
    if (operand < DataType::COLUMN_INT)
    {
        return static_cast<DataType>(operand + DataType::COLUMN_INT);
    }
    return operand;
}

DataType CpuWhereListener::GetDataTypeFromString(const std::string& dataType)
{
    return ::GetColumnDataTypeFromString(dataType);
}

std::pair<std::string, DataType> CpuWhereListener::GenerateAndValidateColumnName(GpuSqlParser::ColumnIdContext* ctx)
{
    std::string table;
    std::string column;

    std::string col = ctx->column()->getText();

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

    return tableColumnPair;
}

void CpuWhereListener::WalkAliasExpression(const std::string& alias)
{
    antlr4::tree::ParseTreeWalker walker;
    insideAlias_ = true;
    walker.walk(this, columnAliasContexts_.at(alias));
    insideAlias_ = false;
}

void CpuWhereListener::TrimReg(std::string& reg)
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