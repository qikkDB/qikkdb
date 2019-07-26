#include "CpuWhereListener.h"
#include "../ColumnBase.h"

constexpr float pi() { return 3.1415926f; }

CpuWhereListener::CpuWhereListener(const std::shared_ptr<Database>& database, CpuSqlDispatcher& dispatcher) :
	database(database),
	dispatcher(dispatcher)
{
}

void CpuWhereListener::exitBinaryOperation(GpuSqlParser::BinaryOperationContext * ctx)
{
	std::pair<std::string, DataType> right = stackTopAndPop();
	std::pair<std::string, DataType> left = stackTopAndPop();

	std::string op = ctx->op->getText();
	stringToUpper(op);

	DataType rightOperandType = std::get<1>(right);
	DataType leftOperandType = std::get<1>(left);

	pushArgument(std::get<0>(left).c_str(), leftOperandType);
	pushArgument(std::get<0>(right).c_str(), rightOperandType);

	DataType returnDataType;

	if (op == ">")
	{
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == "<")
	{
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == ">=")
	{
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == "<=")
	{
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == "=")
	{
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == "!=" || op == "<>")
	{
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == "AND")
	{
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == "OR")
	{
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == "*")
	{
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "/")
	{
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "+")
	{
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "-")
	{
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "%")
	{
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "|")
	{
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "&")
	{
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "^")
	{
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "<<")
	{
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == ">>")
	{
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "POINT")
	{
		returnDataType = DataType::COLUMN_POINT;
	}
	else if (op == "GEO_CONTAINS")
	{
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == "GEO_INTERSECT")
	{
		returnDataType = DataType::COLUMN_POLYGON;
	}
	else if (op == "GEO_UNION")
	{
		returnDataType = DataType::COLUMN_POLYGON;
	}
	else if (op == "LOG")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "POW")
	{
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "ROOT")
	{
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "ATAN2")
	{
		returnDataType = getReturnDataType(DataType::COLUMN_FLOAT);
	}
	else if (op == "LEFT")
	{
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "RIGHT")
	{
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "CONCAT")
	{
		returnDataType = DataType::COLUMN_STRING;
	}
	dispatcher.addBinaryOperation(leftOperandType, rightOperandType, op);

	std::string reg = getRegString(ctx);
	pushArgument(reg.c_str(), returnDataType);
	pushTempResult(reg, returnDataType);
}

void CpuWhereListener::exitTernaryOperation(GpuSqlParser::TernaryOperationContext * ctx)
{
}

void CpuWhereListener::exitUnaryOperation(GpuSqlParser::UnaryOperationContext * ctx)
{
	std::pair<std::string, DataType> arg = stackTopAndPop();

	std::string op = ctx->op->getText();
	stringToUpper(op);
	DataType operandType = std::get<1>(arg);
	pushArgument(std::get<0>(arg).c_str(), operandType);

	DataType returnDataType;

	if (op == "!")
	{
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == "IS NULL")
	{
		if (operandType < DataType::COLUMN_INT)
		{
			throw NullMaskOperationInvalidOperandException();
		}
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == "IS NOT NULL")
	{
		if (operandType < DataType::COLUMN_INT)
		{
			throw NullMaskOperationInvalidOperandException();
		}
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == "-")
	{
		returnDataType = getReturnDataType(operandType);
	}
	else if (op == "YEAR")
	{
		returnDataType = COLUMN_INT;
	}
	else if (op == "MONTH")
	{
		returnDataType = COLUMN_INT;
	}
	else if (op == "DAY")
	{
		returnDataType = COLUMN_INT;
	}
	else if (op == "HOUR")
	{
		returnDataType = COLUMN_INT;
	}
	else if (op == "MINUTE")
	{
		returnDataType = COLUMN_INT;
	}
	else if (op == "SECOND")
	{
		returnDataType = COLUMN_INT;
	}
	else if (op == "ABS")
	{
		returnDataType = getReturnDataType(operandType);
	}
	else if (op == "SIN")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "COS")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "TAN")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "COT")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "ASIN")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "ACOS")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "ATAN")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "LOG10")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "LOG")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "EXP")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "SQRT")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "SQUARE")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "SIGN")
	{
		returnDataType = DataType::COLUMN_INT;
	}
	else if (op == "ROUND")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "FLOOR")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "CEIL")
	{
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "LTRIM")
	{
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "RTRIM")
	{
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "LOWER")
	{
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "UPPER")
	{
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "LEN")
	{
		returnDataType = DataType::COLUMN_INT;
	}
	dispatcher.addUnaryOperation(operandType, op);

	std::string reg = getRegString(ctx);
	pushArgument(reg.c_str(), returnDataType);
	pushTempResult(reg, returnDataType);
}

void CpuWhereListener::exitIntLiteral(GpuSqlParser::IntLiteralContext * ctx)
{
	std::string token = ctx->getText();
	if (isLong(token))
	{
		parserStack.push(std::make_pair(token, DataType::CONST_LONG));
	}
	else
	{
		parserStack.push(std::make_pair(token, DataType::CONST_INT));
	}
}

void CpuWhereListener::exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext * ctx)
{
	std::string token = ctx->getText();
	if (isDouble(token))
	{
		parserStack.push(std::make_pair(token, DataType::CONST_DOUBLE));
	}
	else
	{
		parserStack.push(std::make_pair(token, DataType::CONST_FLOAT));
	}
}

void CpuWhereListener::exitStringLiteral(GpuSqlParser::StringLiteralContext * ctx)
{
	parserStack.push(std::make_pair(ctx->getText(), DataType::CONST_STRING));
}

void CpuWhereListener::exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext * ctx)
{
	parserStack.push(std::make_pair(ctx->getText(), DataType::CONST_INT8_T));
}

void CpuWhereListener::exitGeoReference(GpuSqlParser::GeoReferenceContext * ctx)
{
	auto start = ctx->start->getStartIndex();
	auto stop = ctx->stop->getStopIndex();
	antlr4::misc::Interval interval(start, stop);
	std::string geoValue = ctx->geometry()->start->getInputStream()->getText(interval);

	if (isPolygon(geoValue))
	{
		parserStack.push(std::make_pair(geoValue, DataType::CONST_POLYGON));
	}
	else if (isPoint(geoValue))
	{
		parserStack.push(std::make_pair(geoValue, DataType::CONST_POINT));
	}
}

void CpuWhereListener::exitVarReference(GpuSqlParser::VarReferenceContext * ctx)
{
	std::pair<std::string, DataType> tableColumnData = generateAndValidateColumnName(ctx->columnId());
	const DataType columnType = std::get<1>(tableColumnData);
	const std::string tableColumn = std::get<0>(tableColumnData);

	parserStack.push(std::make_pair(tableColumn, columnType));
}

void CpuWhereListener::exitDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext * ctx)
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

	parserStack.push(std::make_pair(std::to_string(epochTime), DataType::CONST_LONG));
}

void CpuWhereListener::exitPiLiteral(GpuSqlParser::PiLiteralContext * ctx)
{
	parserStack.push(std::make_pair(std::to_string(pi()), DataType::CONST_FLOAT));
}

void CpuWhereListener::exitNowLiteral(GpuSqlParser::NowLiteralContext * ctx)
{
	std::time_t epochTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	parserStack.push(std::make_pair(std::to_string(epochTime), DataType::CONST_LONG));
}

void CpuWhereListener::exitWhereClause(GpuSqlParser::WhereClauseContext * ctx)
{
	std::pair<std::string, DataType> arg = stackTopAndPop();
	dispatcher.addArgument<const std::string&>(std::get<0>(arg));
	dispatcher.addWhereResultFunction(std::get<1>(arg));
}

void CpuWhereListener::exitFromTables(GpuSqlParser::FromTablesContext *ctx)
{
	for (auto fromTable : ctx->fromTable())
	{
		std::string table = fromTable->table()->getText();
		trimDelimitedIdentifier(table);
		if (database->GetTables().find(table) == database->GetTables().end())
		{
			throw TableNotFoundFromException();
		}
		loadedTables.insert(table);

		if (fromTable->alias())
		{
			std::string alias = fromTable->alias()->getText();
			trimDelimitedIdentifier(alias);
			if (tableAliases.find(alias) != tableAliases.end())
			{
				throw AliasRedefinitionException();
			}
			tableAliases.insert({ alias, table });
		}
	}
}

void CpuWhereListener::pushArgument(const char * token, DataType dataType)
{
	switch (dataType)
	{
	case DataType::CONST_INT:
		dispatcher.addArgument<int32_t>(std::stoi(token));
		break;
	case DataType::CONST_LONG:
		dispatcher.addArgument<int64_t>(std::stoll(token));
		break;
	case DataType::CONST_FLOAT:
		dispatcher.addArgument<float>(std::stof(token));
		break;
	case DataType::CONST_DOUBLE:
		dispatcher.addArgument<double>(std::stod(token));
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
		dispatcher.addArgument<const std::string&>(token);
		break;
	case DataType::DATA_TYPE_SIZE:
	case DataType::CONST_ERROR:
		break;
	}
}

std::pair<std::string, DataType> CpuWhereListener::stackTopAndPop()
{
	std::pair<std::string, DataType> value = parserStack.top();
	parserStack.pop();
	return value;
}

void CpuWhereListener::stringToUpper(std::string & str)
{
	for (auto &c : str)
	{
		c = toupper(c);
	}
}

void CpuWhereListener::pushTempResult(std::string reg, DataType type)
{
	parserStack.push(std::make_pair(reg, type));
}

bool CpuWhereListener::isLong(const std::string & value)
{
	try
	{
		std::stoi(value);
	}
	catch (std::out_of_range &e)
	{
		std::stoll(value);
		return true;
	}
	return false;
}

bool CpuWhereListener::isDouble(const std::string & value)
{
	try
    {
        std::stof(value);
    }
    catch (std::out_of_range &e)
    {
        std::stod(value);
        return true;
    }
    return false;
}

bool CpuWhereListener::isPoint(const std::string & value)
{
	return (value.find("POINT") == 0);
}

bool CpuWhereListener::isPolygon(const std::string & value)
{
	return (value.find("POLYGON") == 0);
}

void CpuWhereListener::trimDelimitedIdentifier(std::string& str)
{
	if (str.front() == '[' && str.back() == ']' && str.size() > 2)
	{
		str.erase(0, 1);
		str.erase(str.size() - 1);
	}
}

std::string CpuWhereListener::getRegString(antlr4::ParserRuleContext * ctx)
{
	return std::string("$") + ctx->getText();
}

DataType CpuWhereListener::getReturnDataType(DataType left, DataType right)
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

DataType CpuWhereListener::getReturnDataType(DataType operand)
{
	if (operand < DataType::COLUMN_INT)
	{
		return static_cast<DataType>(operand + DataType::COLUMN_INT);
	}
	return operand;
}

std::pair<std::string, DataType> CpuWhereListener::generateAndValidateColumnName(GpuSqlParser::ColumnIdContext * ctx)
{
	std::string table;
	std::string column;
	
	std::string col = ctx->column()->getText();
	
	if (ctx->table())
	{
		table = ctx->table()->getText();
		trimDelimitedIdentifier(table);
		column = ctx->column()->getText();
		trimDelimitedIdentifier(column);
	
		if (tableAliases.find(table) != tableAliases.end())
		{
			table = tableAliases.at(table);
		}
	
		if (loadedTables.find(table) == loadedTables.end())
		{
			throw TableNotFoundFromException();
		}
		if (database->GetTables().at(table).GetColumns().find(column) == database->GetTables().at(table).GetColumns().end())
		{
			throw ColumnNotFoundException();
		}
	}
	else
	{
		int uses = 0;
		for (auto &tab : loadedTables)
		{
			if (database->GetTables().at(tab).GetColumns().find(col) != database->GetTables().at(tab).GetColumns().end())
			{
				table = tab;
				column = col;
				uses++;
			}
			if (uses > 1)
			{
				throw ColumnAmbiguityException();
			}
		}
		if (column.empty())
		{
			throw ColumnNotFoundException();
		}
	}
	
	std::string tableColumn = table + "." + column;
	DataType columnType = database->GetTables().at(table).GetColumns().at(column)->GetColumnType();
	
	std::pair<std::string, DataType> tableColumnPair = std::make_pair(tableColumn, columnType);
	
	return tableColumnPair;
}
