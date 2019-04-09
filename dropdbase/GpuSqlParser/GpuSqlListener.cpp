//
// Created by Martin Staňo on 2019-01-15.
//

#include "GpuSqlListener.h"
#include "../Table.h"
#include "GpuSqlDispatcher.h"
#include <ctime>
#include <iostream>
#include <sstream>
#include <locale>
#include <iomanip>


GpuSqlListener::GpuSqlListener(const std::shared_ptr<Database>& database, GpuSqlDispatcher& dispatcher): 
	database(database), 
	dispatcher(dispatcher), 
	resultLimit(std::numeric_limits<int64_t>::max()), 
	resultOffset(0),
	usingGroupBy(false), 
	insideAgg(false), 
	insideGroupBy(false), 
	insideSelectColumn(false), 
	isAggSelectColumn(false)
{
}


void GpuSqlListener::exitBinaryOperation(GpuSqlParser::BinaryOperationContext *ctx)
{
    std::pair<std::string, DataType> right = stackTopAndPop();
    std::pair<std::string, DataType> left = stackTopAndPop();

    std::string op = ctx->op->getText();
    stringToUpper(op);

    DataType rightOperandType = std::get<1>(right);
    DataType leftOperandType = std::get<1>(left);
	
    pushArgument(std::get<0>(right).c_str(), rightOperandType);
    pushArgument(std::get<0>(left).c_str(), leftOperandType);

	DataType returnDataType;

    if (op == ">")
    {
        dispatcher.addGreaterFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_INT8_T;
    } 
	else if (op == "<")
    {
        dispatcher.addLessFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_INT8_T;
    } 
	else if (op == ">=")
    {
        dispatcher.addGreaterEqualFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_INT8_T;
    } 
	else if (op == "<=")
    {
        dispatcher.addLessEqualFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_INT8_T;
    } 
	else if (op == "=")
    {
        dispatcher.addEqualFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_INT8_T;
    } 
	else if (op == "!=")
    {
        dispatcher.addNotEqualFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_INT8_T;
    } 
	else if (op == "AND")
    {
        dispatcher.addLogicalAndFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_INT8_T;
    } 
	else if (op == "OR")
    {
        dispatcher.addLogicalOrFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_INT8_T;
    } 
	else if (op == "*")
    {
        dispatcher.addMulFunction(leftOperandType, rightOperandType);
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
    } 
	else if (op == "/")
    {
        dispatcher.addDivFunction(leftOperandType, rightOperandType);
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
    } 
	else if (op == "+")
    {
        dispatcher.addAddFunction(leftOperandType, rightOperandType);
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
    } 
	else if (op == "-")
    {
        dispatcher.addSubFunction(leftOperandType, rightOperandType);
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
    } 
	else if (op == "%")
    {
        dispatcher.addModFunction(leftOperandType, rightOperandType);
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
    }
	else if (op == "POINT")
	{
		dispatcher.addPointFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_POINT;
	}
	else if (op == "GEO_CONTAINS")
    {
        dispatcher.addContainsFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_INT8_T;
    }
    else if (op == "GEO_INTERSECT")
    {
        dispatcher.addIntersectFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_POLYGON;
    }
    else if (op == "GEO_UNION")
    {
        dispatcher.addUnionFunction(leftOperandType, rightOperandType);
        returnDataType = DataType::COLUMN_POLYGON;
    }

	std::string reg = getRegString(ctx);
	pushArgument(reg.c_str(), returnDataType);
    pushTempResult(reg, returnDataType);
}


void GpuSqlListener::exitTernaryOperation(GpuSqlParser::TernaryOperationContext *ctx)
{
    std::pair<std::string, DataType> op1 = stackTopAndPop();
    std::pair<std::string, DataType> op2 = stackTopAndPop();
    std::pair<std::string, DataType> op3 = stackTopAndPop();

    std::string op = ctx->op->getText();
    stringToUpper(op);

    DataType op1Type = std::get<1>(op1);
    DataType op2Type = std::get<1>(op2);
    DataType op3Type = std::get<1>(op3);

    pushArgument(std::get<0>(op1).c_str(), op1Type);
    pushArgument(std::get<0>(op2).c_str(), op2Type);
    pushArgument(std::get<0>(op3).c_str(), op3Type);

    if (op == "BETWEEN")
    {
        dispatcher.addBetweenFunction(op1Type, op2Type, op3Type);
    }

	std::string reg = getRegString(ctx);
	pushArgument(reg.c_str(), DataType::COLUMN_INT8_T);
    pushTempResult(reg, DataType::COLUMN_INT8_T);
}

void GpuSqlListener::exitUnaryOperation(GpuSqlParser::UnaryOperationContext *ctx)
{
    std::pair<std::string, DataType> arg = stackTopAndPop();

    std::string op = ctx->op->getText();
    stringToUpper(op);
    DataType operandType = std::get<1>(arg);
    pushArgument(std::get<0>(arg).c_str(), operandType);

	DataType returnDataType;

    if (op == "!")
    {
        dispatcher.addLogicalNotFunction(operandType);
		returnDataType = DataType::COLUMN_INT8_T;
    } 
	else if (op == "-")
    {
        dispatcher.addMinusFunction(operandType);
		returnDataType = getReturnDataType(operandType);
    }
	else if (op == "YEAR")
	{
		dispatcher.addYearFunction(operandType);
		returnDataType = COLUMN_INT;
	}
	else if (op == "MONTH")
	{
		dispatcher.addMonthFunction(operandType);
		returnDataType = COLUMN_INT;
	}
	else if (op == "DAY")
	{
		dispatcher.addDayFunction(operandType);
		returnDataType = COLUMN_INT;
	}
	else if (op == "HOUR")
	{
		dispatcher.addHourFunction(operandType);
		returnDataType = COLUMN_INT;
	}
	else if (op == "MINUTE")
	{
		dispatcher.addMinuteFunction(operandType);
		returnDataType = COLUMN_INT;
	}
	else if (op == "SECOND")
	{
		dispatcher.addSecondFunction(operandType);
		returnDataType = COLUMN_INT;
	}

	std::string reg = getRegString(ctx);
	pushArgument(reg.c_str(), returnDataType);
    pushTempResult(reg, returnDataType);
}

void GpuSqlListener::enterAggregation(GpuSqlParser::AggregationContext * ctx)
{
	if (insideAgg)
	{
		throw NestedAggregationException();
	}
	insideAgg = true;
	isAggSelectColumn = insideSelectColumn;
}

void GpuSqlListener::exitAggregation(GpuSqlParser::AggregationContext *ctx)
{
    std::pair<std::string, DataType> arg = stackTopAndPop();

    std::string op = ctx->AGG()->getText();
    stringToUpper(op);

    DataType operandType = std::get<1>(arg);
    pushArgument(std::get<0>(arg).c_str(), operandType);
	DataType returnDataType;

	DataType groupByType;
	if (usingGroupBy)
	{
		groupByType = std::get<1>(*(groupByColumns.begin()));
	}
	else
	{
		// TODO
		groupByType = static_cast<DataType>(0);
	}

    if (op == "MIN")
    {
        dispatcher.addMinFunction(groupByType, operandType, groupByType);
		returnDataType = getReturnDataType(operandType);
    } 
	else if (op == "MAX")
    {
        dispatcher.addMaxFunction(groupByType, operandType, groupByType);
		returnDataType = getReturnDataType(operandType);
    } 
	else if (op == "SUM")
    {
        dispatcher.addSumFunction(groupByType, operandType, groupByType);
		returnDataType = getReturnDataType(operandType);
    } 
	else if (op == "COUNT")
    {
        dispatcher.addCountFunction(groupByType, operandType, groupByType);
		returnDataType = DataType::COLUMN_LONG;
    } 
	else if (op == "AVG")
    {
        dispatcher.addAvgFunction(groupByType, operandType, groupByType);
		returnDataType = getReturnDataType(operandType);
    }

	insideAgg = false;
	std::string reg = getRegString(ctx);

	pushArgument(reg.c_str(), returnDataType);
    pushTempResult(reg, returnDataType);
}

void GpuSqlListener::exitSelectColumns(GpuSqlParser::SelectColumnsContext *ctx)
{
	dispatcher.addJmpInstruction();
	dispatcher.addDoneFunction();
}

void GpuSqlListener::enterSelectColumn(GpuSqlParser::SelectColumnContext * ctx)
{
	insideSelectColumn = true;
}

void GpuSqlListener::exitSelectColumn(GpuSqlParser::SelectColumnContext *ctx)
{
	std::pair<std::string, DataType> arg = stackTopAndPop();

	if (!isAggSelectColumn && groupByColumns.find(arg) == groupByColumns.end() && usingGroupBy)
	{
		throw ColumnGroupByException();
	}

	std::string colName = std::get<0>(arg);
	DataType retType = std::get<1>(arg);
	dispatcher.addRetFunction(retType);
	dispatcher.addArgument<const std::string&>(colName);
	
	if (ctx->alias())
	{
		std::string alias = ctx->alias()->getText();
		if (columnAliases.find(alias) != columnAliases.end())
		{
			throw AliasRedefinitionException();
		}
		columnAliases.insert(alias);
		dispatcher.addArgument<const std::string&>(alias);
	}
	else
	{
		dispatcher.addArgument<const std::string&>(colName);
	}

	insideSelectColumn = false;
	isAggSelectColumn = false;
}

void GpuSqlListener::exitFromTables(GpuSqlParser::FromTablesContext *ctx)
{
    for (auto fromTable : ctx->fromTable())
    {
		std::string table = fromTable->table()->getText();
        if (database->GetTables().find(table) == database->GetTables().end())
        {
            throw TableNotFoundFromException();
        }
        loadedTables.insert(table);

		if (fromTable->alias())
		{
			std::string alias = fromTable->alias()->getText();
			if (tableAliases.find(alias) != tableAliases.end())
			{
				throw AliasRedefinitionException();
			}
			tableAliases.insert({ alias, table });
		}
    }
}

void GpuSqlListener::exitWhereClause(GpuSqlParser::WhereClauseContext *ctx)
{
    std::pair<std::string, DataType> arg = stackTopAndPop();
    dispatcher.addArgument<const std::string&>(std::get<0>(arg));
    dispatcher.addFilFunction();
}

void GpuSqlListener::enterGroupByColumns(GpuSqlParser::GroupByColumnsContext * ctx)
{
	insideGroupBy = true;
}

void GpuSqlListener::exitGroupByColumns(GpuSqlParser::GroupByColumnsContext *ctx)
{
    usingGroupBy = true;
	insideGroupBy = false;
}

void GpuSqlListener::exitGroupByColumn(GpuSqlParser::GroupByColumnContext * ctx)
{
	std::pair<std::string, DataType> operand = stackTopAndPop();

	if (groupByColumns.find(operand) == groupByColumns.end())
	{
		dispatcher.addGroupByFunction(std::get<1>(operand));
		dispatcher.addArgument<const std::string&>(std::get<0>(operand));
		groupByColumns.insert(operand);
	}
}

void GpuSqlListener::exitShowDatabases(GpuSqlParser::ShowDatabasesContext * ctx)
{
	dispatcher.addShowDatabasesFunction();
}

void GpuSqlListener::exitShowTables(GpuSqlParser::ShowTablesContext * ctx)
{
	dispatcher.addShowTablesFunction();
	std::string db;

	if(ctx->database())
	{
		db = ctx->database()->getText();

		if(!Database::Exists(db))
		{
			throw DatabaseNotFoundException();
		}
	}
	else
	{
		if (database)
		{
			db = database->GetName();
		}
		else
		{
			throw DatabaseNotFoundException();
		}
	}

	dispatcher.addArgument<const std::string&>(db);
}

void GpuSqlListener::exitShowColumns(GpuSqlParser::ShowColumnsContext * ctx)
{
	dispatcher.addShowColumnsFunction();
	std::string db;
	std::string table;

	if (ctx->database())
	{
		db = ctx->database()->getText();

		if (!Database::Exists(db))
		{
			throw DatabaseNotFoundException();
		}
	}
	else
	{
		if (database)
		{

			db = database->GetName();
		}
		else
		{
			throw DatabaseNotFoundException();
		}
	}

	std::shared_ptr<Database> databaseObject = Database::GetDatabaseByName(db);
	table = ctx->table()->getText();
	
	if (databaseObject->GetTables().find(table) == databaseObject->GetTables().end())
	{
		throw TableNotFoundFromException();
	}

	dispatcher.addArgument<const std::string&>(db);
	dispatcher.addArgument<const std::string&>(table);
}

void GpuSqlListener::exitSqlInsertInto(GpuSqlParser::SqlInsertIntoContext * ctx)
{
	std::string table = ctx->table()->getText();
	if (database->GetTables().find(table) == database->GetTables().end())
	{
		throw TableNotFoundFromException();
	}
	auto& tab = database->GetTables().at(table);
	
	std::vector<std::pair<std::string, DataType>> columns;
	std::vector<std::string> values;

	for (auto& insertIntoColumn : ctx->insertIntoColumns()->columnId())
	{
		if (insertIntoColumn->table())
		{
			throw ColumnNotFoundException();
		}

		std::string column = insertIntoColumn->column()->getText();
		if (tab.GetColumns().find(column) == tab.GetColumns().end())
		{
			throw ColumnNotFoundException();
		}
		DataType columnDataType = tab.GetColumns().at(column).get()->GetColumnType();
		std::pair<std::string, DataType> columnPair = std::make_pair(column, columnDataType);
		
		if (std::find(columns.begin(), columns.end(), columnPair) != columns.end())
		{
			throw InsertIntoException();
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

		dispatcher.addInsertIntoFunction(columnDataType);

		bool isReferencedColumn = std::find(columns.begin(), columns.end(), columnPair) != columns.end();

		dispatcher.addArgument<const std::string&>(table);
		dispatcher.addArgument<const std::string&>(columnName);
		dispatcher.addArgument<bool>(isReferencedColumn);

		if (isReferencedColumn)
		{
			int valueIndex = std::find(columns.begin(), columns.end(), columnPair) - columns.begin();
			std::cout << values[valueIndex].c_str() << " " <<  columnName << std::endl;
			pushArgument(values[valueIndex].c_str(), static_cast<DataType>(static_cast<int>(columnDataType) - DataType::COLUMN_INT));
		}
	}
	dispatcher.addInsertIntoDoneFunction();
}

void GpuSqlListener::exitLimit(GpuSqlParser::LimitContext* ctx)
{
    resultLimit = std::stoi(ctx->getText());
}

void GpuSqlListener::exitOffset(GpuSqlParser::OffsetContext* ctx)
{
    resultOffset = std::stoi(ctx->getText());
}

void GpuSqlListener::exitIntLiteral(GpuSqlParser::IntLiteralContext *ctx)
{
    std::string token = ctx->getText();
    if (isLong(token))
    {
        parserStack.push(std::make_pair(token, DataType::CONST_LONG));
    } else
    {
        parserStack.push(std::make_pair(token, DataType::CONST_INT));
    }
}

void GpuSqlListener::exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext *ctx)
{
    std::string token = ctx->getText();
    if (isDouble(token))
    {
        parserStack.push(std::make_pair(token, DataType::CONST_DOUBLE));
    } else
    {
        parserStack.push(std::make_pair(token, DataType::CONST_FLOAT));
    }
}

void GpuSqlListener::exitStringLiteral(GpuSqlParser::StringLiteralContext *ctx)
{
    parserStack.push(std::make_pair(ctx->getText(), DataType::CONST_STRING));
}


void GpuSqlListener::exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext *ctx)
{
    parserStack.push(std::make_pair(ctx->getText(), DataType::CONST_INT8_T));
}

void GpuSqlListener::exitVarReference(GpuSqlParser::VarReferenceContext *ctx)
{
    std::pair<std::string, DataType> tableColumnData = generateAndValidateColumnName(ctx->columnId());
    const DataType columnType = std::get<1>(tableColumnData);
	const std::string tableColumn = std::get<0>(tableColumnData);
	
    parserStack.push(std::make_pair(tableColumn, columnType));
}

void GpuSqlListener::exitDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext * ctx)
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

void GpuSqlListener::exitGeoReference(GpuSqlParser::GeoReferenceContext *ctx)
{

    auto start = ctx->start->getStartIndex();
    auto stop = ctx->stop->getStopIndex();
    antlr4::misc::Interval interval(start, stop);
    std::string geoValue = ctx->geometry()->start->getInputStream()->getText(interval);

    if (isPolygon(geoValue))
    {
        parserStack.push(std::make_pair(geoValue, DataType::CONST_POLYGON));
    } else if (isPoint(geoValue))
    {
        parserStack.push(std::make_pair(geoValue, DataType::CONST_POINT));
    }
}


std::pair<std::string, DataType> GpuSqlListener::generateAndValidateColumnName(GpuSqlParser::ColumnIdContext *ctx)
{
    std::string table;
    std::string column;

    std::string col = ctx->column()->getText();

    if (ctx->table())
    {
        table = ctx->table()->getText();
        column = ctx->column()->getText();

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

	if (insideGroupBy && originalGroupByColumns.find(tableColumnPair) == originalGroupByColumns.end())
	{
		originalGroupByColumns.insert(tableColumnPair);
	}

    if (usingGroupBy && !insideAgg && originalGroupByColumns.find(tableColumnPair) == originalGroupByColumns.end())
    {
        throw ColumnGroupByException();
    }

    return tableColumnPair;
}

std::pair<std::string, DataType> GpuSqlListener::stackTopAndPop()
{
    std::pair<std::string, DataType> value = parserStack.top();
    parserStack.pop();
    return value;
}

void GpuSqlListener::pushTempResult(std::string reg, DataType type)
{
    parserStack.push(std::make_pair(reg, type));
}

void GpuSqlListener::pushArgument(const char *token, DataType dataType)
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

bool GpuSqlListener::isLong(const std::string &value)
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

bool GpuSqlListener::isDouble(const std::string &value)
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

bool GpuSqlListener::isPoint(const std::string &value)
{
	return (value.find("POINT") == 0);
}

bool GpuSqlListener::isPolygon(const std::string &value)
{
	return (value.find("POLYGON") == 0);
}

void GpuSqlListener::stringToUpper(std::string &str)
{
    for (auto &c : str)
    {
        c = toupper(c);
    }
}

std::string GpuSqlListener::getRegString(antlr4::ParserRuleContext* ctx)
{
	return std::string("$") + ctx->getText();
}

DataType GpuSqlListener::getReturnDataType(DataType left, DataType right)
{
	if (right < DataType::COLUMN_INT)
	{
		right = static_cast<DataType>(right + DataType::COLUMN_INT);
	}
	if (left < DataType::COLUMN_INT)
	{
		left = static_cast<DataType>(left + DataType::COLUMN_INT);
	}
	DataType result = std::max<DataType>(left,right);
	
	return result;
}

DataType GpuSqlListener::getReturnDataType(DataType operand)
{
	if (operand < DataType::COLUMN_INT)
	{
		return static_cast<DataType>(operand + DataType::COLUMN_INT);
	}
	return operand;
}
