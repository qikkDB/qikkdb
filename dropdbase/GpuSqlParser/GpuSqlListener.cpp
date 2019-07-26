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

/// <summary>
/// Definition of PI constant
/// </summary>
constexpr float pi() { return 3.1415926f; }

/// <summary>
/// GpuListner Constructor
/// Initializes AST walk flags (insideAgg, insideGroupBy, etc.)
/// Takes reference to database and dispatcher instances
/// </summary>
/// <param name="database">Database instance reference</param>
/// <param name="dispatcher">Dispatcher instance reference</param>
GpuSqlListener::GpuSqlListener(const std::shared_ptr<Database>& database, GpuSqlDispatcher& dispatcher, GpuSqlJoinDispatcher& joinDispatcher) :
	database(database),
	dispatcher(dispatcher),
	joinDispatcher(joinDispatcher),
	linkTableIndex(0),
	orderByColumnIndex(0),
	usingLoad(false),
	usingWhere(false),
	usingGroupBy(false), 
	insideAgg(false), 
	insideGroupBy(false),
	insideOrderBy(false),
	insideSelectColumn(false), 
	isAggSelectColumn(false),
	resultLimit(std::numeric_limits<int64_t>::max()),
	resultOffset(0)
{
	GpuSqlDispatcher::linkTable.clear();
}

/// <summary>
/// Method that executes on exit of binary operation node in the AST
/// Pops the two operands from stack, reads operation from context and add dispatcher
/// operation and operands to respective dispatcher queues. Pushes result back to parser stack
/// </summary>
/// <param name="ctx">Binary operation context</param>
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

	DataType returnDataType = DataType::CONST_ERROR;

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
	else if (op == "!=" || op == "<>")
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
	else if (op == "|")
	{
		dispatcher.addBitwiseOrFunction(leftOperandType, rightOperandType);
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "&")
	{
		dispatcher.addBitwiseAndFunction(leftOperandType, rightOperandType);
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "^")
	{
		dispatcher.addBitwiseXorFunction(leftOperandType, rightOperandType);
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "<<")
	{
		dispatcher.addBitwiseLeftShiftFunction(leftOperandType, rightOperandType);
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == ">>")
	{
		dispatcher.addBitwiseRightShiftFunction(leftOperandType, rightOperandType);
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
	else if (op == "LOG")
	{
		dispatcher.addLogarithmFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "POW")
	{
		dispatcher.addPowerFunction(leftOperandType, rightOperandType);
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "ROOT")
	{
		dispatcher.addRootFunction(leftOperandType, rightOperandType);
		returnDataType = getReturnDataType(leftOperandType, rightOperandType);
	}
	else if (op == "ATAN2")
	{
		dispatcher.addArctangent2Function(leftOperandType, rightOperandType);
		returnDataType = getReturnDataType(DataType::COLUMN_FLOAT);
	}
	else if (op == "CONCAT")
	{
		dispatcher.addConcatFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "LEFT")
	{
		dispatcher.addLeftFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "RIGHT")
	{
		dispatcher.addRightFunction(leftOperandType, rightOperandType);
		returnDataType = DataType::COLUMN_STRING;
	}

	std::string reg = getRegString(ctx);
	pushArgument(reg.c_str(), returnDataType);
    pushTempResult(reg, returnDataType);
}

/// <summary>
/// Method that executes on exit of ternary operation node in the AST
/// Pops the three operands from stack, reads operation from context and add dispatcher
/// operation and operands to respective dispatcher queues. Pushes result back to parser stack
/// </summary>
/// <param name="ctx">Ternary operation context</param>
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

/// <summary>
/// Method that executes on exit of unary operation node in the AST
/// Pops the one operand from stack, reads operation from context and add dispatcher
/// operation and operand to respective dispatcher queues. Pushes result back to parser stack
/// </summary>
/// <param name="ctx">Unary operation context</param>
void GpuSqlListener::exitUnaryOperation(GpuSqlParser::UnaryOperationContext *ctx)
{
    std::pair<std::string, DataType> arg = stackTopAndPop();

    std::string op = ctx->op->getText();
    stringToUpper(op);
    DataType operandType = std::get<1>(arg);
    pushArgument(std::get<0>(arg).c_str(), operandType);

	DataType returnDataType = DataType::CONST_ERROR;

    if (op == "!")
    {
        dispatcher.addLogicalNotFunction(operandType);
		returnDataType = DataType::COLUMN_INT8_T;
    }
	else if (op == "IS NULL")
	{
		if (operandType < DataType::COLUMN_INT)
		{
			throw NullMaskOperationInvalidOperandException();
		}
		dispatcher.addIsNullFunction();
		returnDataType = DataType::COLUMN_INT8_T;
	}
	else if (op == "IS NOT NULL")
	{
		if (operandType < DataType::COLUMN_INT)
		{
			throw NullMaskOperationInvalidOperandException();
		}
		dispatcher.addIsNotNullFunction();
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
	else if (op == "ABS")
	{
		dispatcher.addAbsoluteFunction(operandType);
		returnDataType = getReturnDataType(operandType);
	}
	else if (op == "SIN")
	{
		dispatcher.addSineFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "COS")
	{
		dispatcher.addCosineFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "TAN")
	{
		dispatcher.addTangentFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "COT")
	{
		dispatcher.addCotangentFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "ASIN")
	{
		dispatcher.addArcsineFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "ACOS")
	{
		dispatcher.addArccosineFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "ATAN")
	{
		dispatcher.addArctangentFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
    else if (op == "LOG10")
    {
        dispatcher.addLogarithm10Function(operandType);
        returnDataType = DataType::COLUMN_FLOAT;
    }
	else if (op == "LOG")
	{
		dispatcher.addLogarithmNaturalFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "EXP")
	{
		dispatcher.addExponentialFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "SQRT")
	{
		dispatcher.addSquareRootFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "SQUARE")
	{
		dispatcher.addSquareFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "SIGN")
	{
		dispatcher.addSignFunction(operandType);
		returnDataType = DataType::COLUMN_INT;
	}
	else if (op == "ROUND")
	{
		dispatcher.addRoundFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "FLOOR")
	{
		dispatcher.addFloorFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "CEIL")
	{
		dispatcher.addCeilFunction(operandType);
		returnDataType = DataType::COLUMN_FLOAT;
	}
	else if (op == "LTRIM")
	{
		dispatcher.addLtrimFunction(operandType);
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "RTRIM")
	{
		dispatcher.addRtrimFunction(operandType);
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "LOWER")
	{
		dispatcher.addLowerFunction(operandType);
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "UPPER")
	{
		dispatcher.addUpperFunction(operandType);
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "REVERSE")
	{
		dispatcher.addReverseFunction(operandType);
		returnDataType = DataType::COLUMN_STRING;
	}
	else if (op == "LEN")
	{
		dispatcher.addLenFunction(operandType);
		returnDataType = DataType::COLUMN_INT;
	}

	std::string reg = getRegString(ctx);
	pushArgument(reg.c_str(), returnDataType);
    pushTempResult(reg, returnDataType);
}

/// <summary>
/// Method that executes on enter of aggregation operation node in the AST
/// Sets insideAgg, isAggSelectColumn parser flag
/// Throws NestedAggregationException in case e.g SUM(SUM(colA))
/// </summary>
/// <param name="ctx">Aggregation context</param>
void GpuSqlListener::enterAggregation(GpuSqlParser::AggregationContext * ctx)
{
	if (insideAgg)
	{
		throw NestedAggregationException();
	}
	insideAgg = true;
	isAggSelectColumn = insideSelectColumn;
}

/// <summary>
/// Method that executes on exit of aggregation operation node in the AST
/// Pops one operand from stack, adds aggregation operation and argument to respective Dispatcher queues.
/// Pushes result back to parser stack.
/// </summary>
/// <param name="ctx">Aggregation context</param>
void GpuSqlListener::exitAggregation(GpuSqlParser::AggregationContext *ctx)
{
    std::pair<std::string, DataType> arg = stackTopAndPop();

    std::string op = ctx->AGG()->getText();
    stringToUpper(op);

    DataType valueType = std::get<1>(arg);
    pushArgument(std::get<0>(arg).c_str(), valueType);
	DataType returnDataType = DataType::CONST_ERROR;

	GroupByType groupByType = GroupByType::NO_GROUP_BY;
	DataType keyType = static_cast<DataType>(0);

	if (usingGroupBy)
	{
		groupByType = GroupByType::SINGLE_KEY_GROUP_BY;
		keyType = std::get<1>(*(groupByColumns.begin()));

		if (groupByColumns.size() > 1)
		{
			groupByType = GroupByType::MULTI_KEY_GROUP_BY;
		}
	}

    if (op == "MIN")
    {
        dispatcher.addMinFunction(keyType, valueType, groupByType);
		returnDataType = getReturnDataType(valueType);
    } 
	else if (op == "MAX")
    {
        dispatcher.addMaxFunction(keyType, valueType, groupByType);
		returnDataType = getReturnDataType(valueType);
    } 
	else if (op == "SUM")
    {
        dispatcher.addSumFunction(keyType, valueType, groupByType);
		returnDataType = getReturnDataType(valueType);
    } 
	else if (op == "COUNT")
    {
        dispatcher.addCountFunction(keyType, valueType, groupByType);
		returnDataType = DataType::COLUMN_LONG;
    } 
	else if (op == "AVG")
    {
        dispatcher.addAvgFunction(keyType, valueType, groupByType);
		returnDataType = getReturnDataType(valueType);
    }

	insideAgg = false;
	std::string reg = getRegString(ctx);

	pushArgument(reg.c_str(), returnDataType);
    pushTempResult(reg, returnDataType);
}

/// Method that executes on exit of SELECT clause (return columns)
/// Generates jump operation (used to iterate blocks) and done operation (marking end of exucution)
/// <param name="ctx">Select Columns context</param>
void GpuSqlListener::exitSelectColumns(GpuSqlParser::SelectColumnsContext *ctx)
{
	for (auto& retCol : returnColumns)
	{
		std::string colName = retCol.first;
		DataType retType = std::get<0>(retCol.second);
		std::string alias = std::get<1>(retCol.second);
		dispatcher.addRetFunction(retType);
		dispatcher.addArgument<const std::string&>(colName);
		dispatcher.addArgument<const std::string&>(alias);
	}

	dispatcher.addJmpInstruction();
	dispatcher.addDoneFunction();
}


/// Method that executes on enter of SELECT clause (return columns)
/// Sets insideSelectColumn parser flag
/// <param name="ctx">Select Columns context</param>
void GpuSqlListener::enterSelectColumn(GpuSqlParser::SelectColumnContext * ctx)
{
	insideSelectColumn = true;
}


/// Method that executes on exit of single SELECT column (return column)
/// Checks if a column is either aggregation or group by column
/// Pops from parser stack and generates return operation
/// Sets column alias if present and checks its potential redefinition
/// Resets insideAggregation and insideSelectColumn parser flags
/// <param name="ctx">Select Column context</param>
void GpuSqlListener::exitSelectColumn(GpuSqlParser::SelectColumnContext *ctx)
{
	std::pair<std::string, DataType> arg = stackTopAndPop();
	std::string colName = std::get<0>(arg);
	DataType retType = std::get<1>(arg);
	std::string alias;
	
	if (ctx->alias())
	{
		alias = ctx->alias()->getText();
		trimDelimitedIdentifier(alias);

		if (columnAliases.find(alias) != columnAliases.end())
		{
			throw AliasRedefinitionException();
		}
		columnAliases.insert(alias);
	}
	else
	{
		alias = colName;
	}

	returnColumns.insert({ colName, {retType, alias } });

	dispatcher.addArgument<const std::string&>(colName);
	dispatcher.addLockRegisterFunction();

	insideSelectColumn = false;
	isAggSelectColumn = false;
}


/// Method that executes on exit of FROM clause (tables)
/// Checks for table existance
/// Sets table alias if present and checks its potential redefinition
/// <param name="ctx">From Tables context</param>
void GpuSqlListener::exitFromTables(GpuSqlParser::FromTablesContext *ctx)
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

void GpuSqlListener::exitJoinClause(GpuSqlParser::JoinClauseContext * ctx)
{
	std::string joinTable = ctx->joinTable()->getText();

	if (database->GetTables().find(joinTable) == database->GetTables().end())
	{
		throw TableNotFoundFromException();
	}

	loadedTables.insert(joinTable);

	std::string leftColName;
	DataType leftColType;
	std::tie(leftColName, leftColType) = generateAndValidateColumnName(ctx->joinColumnLeft()->columnId());

	std::string rightColName;
	DataType rightColType;
	std::tie(rightColName, rightColType) = generateAndValidateColumnName(ctx->joinColumnRight()->columnId());

	if (leftColType != rightColType)
	{
		throw JoinColumnTypeException();
	}

	JoinType joinType = JoinType::INNER_JOIN;
	if (ctx->joinType())
	{
		std::string joinTypeName = ctx->joinType()->getText();
		stringToUpper(joinTypeName);

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

	joinDispatcher.addJoinFunction(leftColType, joinOperator);
	joinDispatcher.addArgument<const std::string&>(leftColName);
	joinDispatcher.addArgument<const std::string&>(rightColName);
	joinDispatcher.addArgument<int32_t>(joinType);
}

void GpuSqlListener::exitJoinClauses(GpuSqlParser::JoinClausesContext * ctx)
{
	joinDispatcher.addJoinDoneFunction();
}


/// Method that executes on exit of WHERE clause
/// Pops from parser stack, generates fil operation which marks register
/// used as final filtration mask in recostruct operations
/// <param name="ctx">Where Clause context</param>
void GpuSqlListener::exitWhereClause(GpuSqlParser::WhereClauseContext *ctx)
{
	usingWhere = true;
    std::pair<std::string, DataType> arg = stackTopAndPop();
    dispatcher.addArgument<const std::string&>(std::get<0>(arg));
    dispatcher.addFilFunction();
}

void GpuSqlListener::enterWhereClause(GpuSqlParser::WhereClauseContext * ctx)
{
	dispatcher.addWhereEvaluationFunction();
}


/// Method that executes on enter of GROUP BY clause
/// Sets insideGroupBy parser flag.
/// <param name="ctx">Group By Columns context</param>
void GpuSqlListener::enterGroupByColumns(GpuSqlParser::GroupByColumnsContext * ctx)
{
	insideGroupBy = true;
}


/// Method that executes on exit of GROUP BY clause
/// Sets usingGroupBy and resets insideGoupBy parser flags
/// <param name="ctx">Group By Columns context</param>
void GpuSqlListener::exitGroupByColumns(GpuSqlParser::GroupByColumnsContext *ctx)
{
    usingGroupBy = true;
	insideGroupBy = false;
}


/// Method that executes on exit of a single GROUP BY column
/// Pops from parser stack and generates group by operation
/// Appends to list of group by columns
/// <param name="ctx">Group By Column context</param>
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


/// Method that executes on exit of SHOW DATABASES command
/// Generates show databases operation
/// <param name="ctx">Show Databases context</param>
void GpuSqlListener::exitShowDatabases(GpuSqlParser::ShowDatabasesContext * ctx)
{
	dispatcher.addShowDatabasesFunction();
}

/// Method that executes on exit of SHOW TABLES command
/// Generates show tables operation
/// Checks if database with given name exists
/// If no database name is provided uses currently bound database
/// <param name="ctx">Show Tables context</param>
void GpuSqlListener::exitShowTables(GpuSqlParser::ShowTablesContext * ctx)
{
	dispatcher.addShowTablesFunction();
	std::string db;

	if(ctx->database())
	{
		db = ctx->database()->getText();
		trimDelimitedIdentifier(db);

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

/// Method that executes on exit of SHOW COLUMNS command
/// Generates show tables operation
/// Checks if database with given name exists
/// If no database name is provided uses currently bound database
/// Checks if table with given name exists
/// <param name="ctx">Show Columns context</param>
void GpuSqlListener::exitShowColumns(GpuSqlParser::ShowColumnsContext * ctx)
{
	dispatcher.addShowColumnsFunction();
	std::string db;
	std::string table;

	if (ctx->database())
	{
		db = ctx->database()->getText();
		trimDelimitedIdentifier(db);

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
	trimDelimitedIdentifier(table);
	
	if (databaseObject->GetTables().find(table) == databaseObject->GetTables().end())
	{
		throw TableNotFoundFromException();
	}

	dispatcher.addArgument<const std::string&>(db);
	dispatcher.addArgument<const std::string&>(table);
}

void GpuSqlListener::exitSqlCreateDb(GpuSqlParser::SqlCreateDbContext * ctx)
{
	std::string newDbName = ctx->database()->getText();
	trimDelimitedIdentifier(newDbName);

	if (Database::Exists(newDbName))
	{
		throw DatabaseAlreadyExistsException();
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

	dispatcher.addCreateDatabaseFunction();
	dispatcher.addArgument<const std::string&>(newDbName);
	dispatcher.addArgument<int32_t>(newDbBlockSize);
}

void GpuSqlListener::exitSqlDropDb(GpuSqlParser::SqlDropDbContext * ctx)
{
	std::string dbName = ctx->database()->getText();
	trimDelimitedIdentifier(dbName);

	if (!Database::Exists(dbName))
	{
		throw DatabaseNotFoundException();
	}

	dispatcher.addDropDatabaseFunction();
	dispatcher.addArgument<const std::string&>(dbName);
}

void GpuSqlListener::exitSqlCreateTable(GpuSqlParser::SqlCreateTableContext * ctx)
{
	std::string newTableName = ctx->table()->getText();
	trimDelimitedIdentifier(newTableName);

	if (database->GetTables().find(newTableName) != database->GetTables().end())
	{
		throw TableAlreadyExistsException();
	}

	std::unordered_map<std::string, DataType> newColumns;
	std::unordered_map<std::string, std::vector<std::string>> newIndices;

	for (auto& entry : ctx->newTableEntries()->newTableEntry())
	{
		if (entry->newTableColumn())
		{
			auto newColumnContext = entry->newTableColumn();
			DataType newColumnDataType = getDataTypeFromString(newColumnContext->DATATYPE()->getText());
			std::string newColumnName = newColumnContext->column()->getText();
			trimDelimitedIdentifier(newColumnName);
			
			if (newColumns.find(newColumnName) != newColumns.end())
			{
				throw ColumnAlreadyExistsException();
			}

			newColumns.insert({ newColumnName, newColumnDataType });
		}
		if (entry->newTableIndex())
		{
			auto newColumnContext = entry->newTableIndex();
			std::string indexName = newColumnContext->indexName()->getText();
			trimDelimitedIdentifier(indexName);

			if (newIndices.find(indexName) != newIndices.end())
			{
				throw IndexAlreadyExistsException();
			}

			std::vector<std::string> indexColumns;
			for (auto& column : newColumnContext->indexColumns()->column())
			{
				std::string indexColumnName = column->getText();
				trimDelimitedIdentifier(indexColumnName);

				if (newColumns.find(indexColumnName) == newColumns.end())
				{
					throw ColumnNotFoundException();
				}
				if (std::find(indexColumns.begin(), indexColumns.end(), indexColumnName) != indexColumns.end())
				{
					throw ColumnAlreadyExistsInIndexException();
				}
				indexColumns.push_back(indexColumnName);
			}
			newIndices.insert({indexName, indexColumns});
		}
	}

	dispatcher.addCreateTableFunction();

	dispatcher.addArgument<const std::string&>(newTableName);
	dispatcher.addArgument<int32_t>(newColumns.size());
	for (auto& newColumn : newColumns)
	{
		dispatcher.addArgument<const std::string&>(newColumn.first);
		dispatcher.addArgument<int32_t>(static_cast<int32_t>(newColumn.second));
	}

	dispatcher.addArgument<int32_t>(newIndices.size());
	for (auto& newIndex : newIndices)
	{
		dispatcher.addArgument<const std::string&>(newIndex.first);
		dispatcher.addArgument<int32_t>(newIndex.second.size());
		for (auto& indexColumn : newIndex.second)
		{
			dispatcher.addArgument<const std::string&>(indexColumn);
		}
	}
}

void GpuSqlListener::exitSqlDropTable(GpuSqlParser::SqlDropTableContext * ctx)
{
	std::string tableName = ctx->table()->getText();
	trimDelimitedIdentifier(tableName);

	if (database->GetTables().find(tableName) == database->GetTables().end())
	{
		throw TableNotFoundFromException();
	}

	dispatcher.addDropTableFunction();
	dispatcher.addArgument<const std::string&>(tableName);
}

void GpuSqlListener::exitSqlAlterTable(GpuSqlParser::SqlAlterTableContext * ctx)
{
	std::string tableName = ctx->table()->getText();
	trimDelimitedIdentifier(tableName);

	if (database->GetTables().find(tableName) == database->GetTables().end())
	{
		throw TableNotFoundFromException();
	}

	std::unordered_map<std::string, DataType> addColumns;
	std::unordered_set<std::string> dropColumns;
	 
	for (auto &entry : ctx->alterTableEntries()->alterTableEntry())
	{
		if (entry->addColumn())
		{
			auto addColumnContext = entry->addColumn();
			DataType addColumnDataType = getDataTypeFromString(addColumnContext->DATATYPE()->getText());
			std::string addColumnName = addColumnContext->column()->getText();
			trimDelimitedIdentifier(addColumnName);

			if (database->GetTables().at(tableName).GetColumns().find(addColumnName) != database->GetTables().at(tableName).GetColumns().end() ||
				addColumns.find(addColumnName) != addColumns.end())
			{
				throw ColumnAlreadyExistsException();
			}

			addColumns.insert({ addColumnName, addColumnDataType });
		}
		else if (entry->dropColumn())
		{
			auto dropColumnContext = entry->dropColumn();
			std::string dropColumnName = dropColumnContext->column()->getText();
			trimDelimitedIdentifier(dropColumnName);

			if (database->GetTables().at(tableName).GetColumns().find(dropColumnName) == database->GetTables().at(tableName).GetColumns().end() ||
				dropColumns.find(dropColumnName) != dropColumns.end())
			{
				throw ColumnNotFoundException();
			}

			dropColumns.insert({ dropColumnName });
		}
		// Alter Column - type casting
	}

	dispatcher.addAlterTableFunction();
	dispatcher.addArgument<const std::string&>(tableName);

	dispatcher.addArgument<int32_t>(addColumns.size());
	for (auto& addColumn : addColumns)
	{
		dispatcher.addArgument<const std::string&>(addColumn.first);
		dispatcher.addArgument<int32_t>(static_cast<int32_t>(addColumn.second));
	}

	dispatcher.addArgument<int32_t>(dropColumns.size());
	for (auto& dropColumn : dropColumns)
	{
		dispatcher.addArgument<const std::string&>(dropColumn);
	}
}

void GpuSqlListener::exitSqlCreateIndex(GpuSqlParser::SqlCreateIndexContext * ctx)
{
	std::string indexName = ctx->indexName()->getText();
	trimDelimitedIdentifier(indexName);

	std::string tableName = ctx->table()->getText();
	trimDelimitedIdentifier(tableName);

	if (database->GetTables().find(tableName) == database->GetTables().end())
	{
		throw TableNotFoundFromException();
	}

	if (database->GetTables().at(tableName).GetSize() > 0)
	{
		throw TableIsFilledException();
	}

	//check if index already exists

	std::vector<std::string> indexColumns;

	for (auto& column : ctx->indexColumns()->column())
	{
		std::string indexColumnName = column->getText();
		trimDelimitedIdentifier(indexColumnName);

		if (database->GetTables().at(tableName).GetColumns().find(indexColumnName) ==
			database->GetTables().at(tableName).GetColumns().end())
		{
			throw ColumnNotFoundException();
		}
		if (std::find(indexColumns.begin(), indexColumns.end(), indexColumnName) != indexColumns.end())
		{
			throw ColumnAlreadyExistsInIndexException();
		}
		indexColumns.push_back(indexColumnName);
	}

	dispatcher.addCreateIndexFunction();

	dispatcher.addArgument<const std::string&>(indexName);
	dispatcher.addArgument<const std::string&>(tableName);

	dispatcher.addArgument<int32_t>(indexColumns.size());
	for (auto& indexColumn : indexColumns)
	{
		dispatcher.addArgument<const std::string&>(indexColumn);
	}
}

void GpuSqlListener::enterOrderByColumns(GpuSqlParser::OrderByColumnsContext * ctx)
{
	insideOrderBy = true;
}

void GpuSqlListener::exitOrderByColumns(GpuSqlParser::OrderByColumnsContext * ctx)
{
	for (auto& orderByColumn : orderByColumns)
	{
		std::string orderByColName = orderByColumn.first;
		DataType dataType = std::get<0>(orderByColumn.second);

		dispatcher.addArgument<const std::string&>(orderByColName);
		dispatcher.addOrderByReconstructOrderFunction(dataType);
	}

	for (auto& returnColumn : returnColumns)
	{
		std::string returnColName = returnColumn.first;
		DataType dataType = std::get<0>(returnColumn.second);

		dispatcher.addArgument<const std::string&>(returnColName);
		dispatcher.addOrderByReconstructRetFunction(dataType);
	}

	insideOrderBy = false;
	dispatcher.addFreeOrderByTableFunction();
	dispatcher.addOrderByReconstructRetAllBlocksFunction();
}


void GpuSqlListener::exitOrderByColumn(GpuSqlParser::OrderByColumnContext * ctx)
{
	std::pair<std::string, DataType> arg = stackTopAndPop();
	std::string orderByColName = std::get<0>(arg);

	if (orderByColumns.find(orderByColName) != orderByColumns.end())
	{
		throw OrderByColumnAlreadyReferencedException();
	}

	DataType dataType = std::get<1>(arg);
	OrderBy::Order order = OrderBy::Order::ASC;

	if (ctx->DIR())
	{
		std::string dir = ctx->DIR()->getText();
		stringToUpper(dir);
		if (dir == "DESC")
		{
			order = OrderBy::Order::DESC;
		}
	}

	dispatcher.addArgument<const std::string&>(orderByColName);
	dispatcher.addArgument<int32_t>(static_cast<int32_t>(order));
	dispatcher.addArgument<int32_t>(orderByColumnIndex++);
	dispatcher.addOrderByFunction(dataType);

	orderByColumns.insert({ orderByColName, { dataType, order } });
}

/// Method that executes on exit of INSERT INTO command
/// Generates insert into operation
/// Checks if table with given name exists
/// Checks if table.column (dot notation) is not used
/// Checks if column with given name exists
/// Checks if the same column is not referenced multiple times
/// Checks if same number of values and columns is provided
/// <param name="ctx">Sql Insert Into context</param>
void GpuSqlListener::exitSqlInsertInto(GpuSqlParser::SqlInsertIntoContext * ctx)
{
	std::string table = ctx->table()->getText();
	trimDelimitedIdentifier(table);

	if (database->GetTables().find(table) == database->GetTables().end())
	{
		throw TableNotFoundFromException();
	}
	auto& tab = database->GetTables().at(table);
	
	std::vector<std::pair<std::string, DataType>> columns;
	std::vector<std::string> values;
	std::vector<bool> isValueNull;
	for (auto& insertIntoColumn : ctx->insertIntoColumns()->columnId())
	{
		if (insertIntoColumn->table())
		{
			throw ColumnNotFoundException();
		}

		std::string column = insertIntoColumn->column()->getText();
		trimDelimitedIdentifier(column);

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

		dispatcher.addInsertIntoFunction(columnDataType);


		bool hasValue = std::find(columns.begin(), columns.end(), columnPair) != columns.end();
		if(hasValue)
		{
			int valueIndex = std::find(columns.begin(), columns.end(), columnPair) - columns.begin();
			hasValue &= !isValueNull[valueIndex];
		}
		dispatcher.addArgument<const std::string&>(columnName);
		dispatcher.addArgument<bool>(hasValue);
		
		if (hasValue)
		{
			int valueIndex = std::find(columns.begin(), columns.end(), columnPair) - columns.begin();
			std::cout << values[valueIndex].c_str() << " " <<  columnName << std::endl;
			pushArgument(values[valueIndex].c_str(), static_cast<DataType>(static_cast<int>(columnDataType) - DataType::COLUMN_INT));
		}

	}
	dispatcher.addArgument<const std::string&>(table);
	dispatcher.addInsertIntoDoneFunction();
}

/// Method that executes on exit of LIMIT clause
/// Sets the row limit count
/// <param name="ctx">Limit context</param>
void GpuSqlListener::exitLimit(GpuSqlParser::LimitContext* ctx)
{
    resultLimit = std::stoi(ctx->getText());
}

/// Method that executes on exit of OFFSET clause
/// Sets the row offset count
/// <param name="ctx">Offset context</param>
void GpuSqlListener::exitOffset(GpuSqlParser::OffsetContext* ctx)
{
    resultOffset = std::stoi(ctx->getText());
}

bool GpuSqlListener::GetUsingLoad()
{
	return usingLoad;
}

bool GpuSqlListener::GetUsingWhere()
{
	return usingWhere;
}

/// Method that executes on exit of integer literal (10, 20, 5, ...)
/// Infers token data type (int or long)
/// Pushes the literal token to parser stack along with its inferred data type (int or long)
/// <param name="ctx">Int Literal context</param>
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

/// Method that executes on exit of decimal literal (10.5, 20.6, 5.2, ...)
/// Infers token data type (float or double)
/// Pushes the literal token to parser stack along with its inferred data type (float or double)
/// <param name="ctx">Decimal Literal context</param>
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

/// Method that executes on exit of string literal ("Hello", ...)
/// Pushes the literal token to parser stack along with its data type
/// <param name="ctx">String Literal context</param>
void GpuSqlListener::exitStringLiteral(GpuSqlParser::StringLiteralContext *ctx)
{
	std::string strLit = ctx->getText().substr(1, ctx->getText().length() - 2);
    parserStack.push(std::make_pair(strLit, DataType::CONST_STRING));
}

/// Method that executes on exit of boolean literal (True, False)
/// Pushes the literal token to parser stack along with its data type
/// <param name="ctx">Boolean Literal context</param>
void GpuSqlListener::exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext *ctx)
{
    parserStack.push(std::make_pair(ctx->getText(), DataType::CONST_INT8_T));
}

/// Method that executes on exit of column name reference (colInt1, tableA.colInt, ...)
/// Validates column existance and generates its full name in dot notation (table.column)
/// Pushes the token to the parser stack
/// Fills link table (used for gpu where dispatch) with column name and ordinal number of
/// its first appearance in the AST traversal
/// <param name="ctx">Var Reference context</param>
void GpuSqlListener::exitVarReference(GpuSqlParser::VarReferenceContext *ctx)
{
    std::pair<std::string, DataType> tableColumnData = generateAndValidateColumnName(ctx->columnId());
    const DataType columnType = std::get<1>(tableColumnData);
	const std::string tableColumn = std::get<0>(tableColumnData);

	parserStack.push(std::make_pair(tableColumn, columnType));
	usingLoad = true;

	if (GpuSqlDispatcher::linkTable.find(tableColumn) == GpuSqlDispatcher::linkTable.end())
	{
		GpuSqlDispatcher::linkTable.insert({ tableColumn, linkTableIndex++ });
	}
}

/// Method that executes on exit of date time literal in format of yyyy-mm-dd hh:mm:ss
/// Converts the literal to epoch time and pushes is to parser stack as LONG data type literal
/// <param name="ctx">Date Time Literal context</param>
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

/// Method that executes on exit of PI() literal (3.1415926)
/// Pushes pi literal to stack as float data type
/// <param name="ctx">Pi Literal context</param>
void GpuSqlListener::exitPiLiteral(GpuSqlParser::PiLiteralContext * ctx)
{
	parserStack.push(std::make_pair(std::to_string(pi()), DataType::CONST_FLOAT));
}

/// Method that executes on exit of NOW() literal (current date time)
/// Converts the current date time to epoch time and pushes it to parser stack as long data type literal
/// <param name="ctx">Now Literal context</param>
void GpuSqlListener::exitNowLiteral(GpuSqlParser::NowLiteralContext * ctx)
{
	std::time_t epochTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	parserStack.push(std::make_pair(std::to_string(epochTime), DataType::CONST_LONG));
}

/// Method that executes on exit of polygon and point literals
/// Infers the type of literal (polygon or point) and pushes it to parser stack
/// <param name="ctx">Geo Reference context</param>
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

/// Method used to generate column name in dot notation (table.column) and validation of its name
/// Checks for table existance if provided
/// Infers table name if not provided and checks for column ambiguity between tables
/// Retrieves the column data type
/// Checks if column is used in aggregation or group by clause if its present 
/// <param name="ctx">Column Id context</param>
/// <returns="tableColumnPair">Column name in dot notation (table.column)</returns>
std::pair<std::string, DataType> GpuSqlListener::generateAndValidateColumnName(GpuSqlParser::ColumnIdContext *ctx)
{
    std::string table;
    std::string column;

    std::string col = ctx->column()->getText();
	trimDelimitedIdentifier(col);

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

/// Method used to pop contnt from parser stack
/// <returns="value">Pair of content string and contnet data type</returns>
std::pair<std::string, DataType> GpuSqlListener::stackTopAndPop()
{
    std::pair<std::string, DataType> value = parserStack.top();
    parserStack.pop();
    return value;
}

/// Method used to push temp results to parser stack 
/// <param name="reg">String representing the literal</param>
/// <param name="type">Data type of the literal</param>
void GpuSqlListener::pushTempResult(std::string reg, DataType type)
{
    parserStack.push(std::make_pair(reg, type));
}

/// Method used to push argument to dispatcher argument queue
/// Converts the string token to actual numeric data if its numeric literal
/// Keeps it as a string in case of polygons, points and column names
/// <param name="token">String representing the literal</param>
/// <param name="type">Data type of the literal</param>
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
		case DataType::CONST_INT8_T:
		{
			std::string booleanToken(token);
			stringToUpper(booleanToken);
			dispatcher.addArgument<int8_t>(booleanToken == "TRUE");
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
bool GpuSqlListener::isLong(const std::string &value)
{
    try
    {
        std::stoi(value);
    }
    catch (std::out_of_range &)
    {
        std::stoll(value);
        return true;
    }
    return false;
}

/// Checks if given decimal literal is double
/// <param nameDouble">True if literal is double</returns>
bool GpuSqlListener::isDouble(const std::string &value)
{
    try
    {
        std::stof(value);
    }
    catch (std::out_of_range &)
    {
        std::stod(value);
        return true;
    }
    return false;
}


/// Checks if given geo literal is point
/// <param name="value">String representing the literal</param>
/// <returns="isPoints">True if literal is point</returns>
bool GpuSqlListener::isPoint(const std::string &value)
{
	return (value.find("POINT") == 0);
}

/// Checks if given geo literal is polygon
/// <param name="value">String representing the literal</param>
/// <returns="isPolygon">True if literal is polygon</returns>
bool GpuSqlListener::isPolygon(const std::string &value)
{
	return (value.find("POLYGON") == 0);
}

/// Converts string to uppercase
/// <param name="value">String to convert</param>
void GpuSqlListener::stringToUpper(std::string &str)
{
    for (auto &c : str)
    {
        c = toupper(c);
    }
}

void GpuSqlListener::trimDelimitedIdentifier(std::string & str)
{
	if (str.front() == '[' && str.back() == ']' && str.size() > 2)
	{
		str.erase(0, 1);
		str.erase(str.size() - 1);
	}
}

/// Prefixes temporary result key (register) with an $
/// Content of arbitrary parser rule context is used temporar result key
/// <param name="ctx">Parser Rule Context</param>
/// <returns="reg">Prefixed register name</returns>
std::string GpuSqlListener::getRegString(antlr4::ParserRuleContext* ctx)
{
	return std::string("$") + ctx->getText();
}

/// Defines return data type for binary operation
/// If operand type is a constant data type its converted to column data type
/// Data type with higher ordinal number (the ordering is designed with this feature in mind) is chosen
/// <param name="left">Left operand data type</param>
/// <param name="right">Right operand data type</param>
/// <returns="result">Operation result data type</returns>
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

/// Defines return data type for unary operation
/// If operand type is a constant data type its converted to column data type
/// <param name="operand">Operand data type</param>
/// <returns="result">Operation result data type</returns>
DataType GpuSqlListener::getReturnDataType(DataType operand)
{
	if (operand < DataType::COLUMN_INT)
	{
		return static_cast<DataType>(operand + DataType::COLUMN_INT);
	}
	return operand;
}

DataType GpuSqlListener::getDataTypeFromString(std::string dataType)
{
	std::string type = dataType;
	stringToUpper(type);

	if (type == "INT")
	{
		return DataType::COLUMN_INT;
	}
	else if (type == "LONG")
	{
		return DataType::COLUMN_LONG;
	}
	else if (type == "FLOAT")
	{
		return DataType::COLUMN_FLOAT;
	}
	else if (type == "DOUBLE")
	{
		return DataType::COLUMN_DOUBLE;
	}
	else if (type == "POINT")
	{
		return DataType::COLUMN_POINT;
	}
	else if (type == "POLYGON")
	{
		return DataType::COLUMN_POLYGON;
	}
	else if (type == "STRING")
	{
		return DataType::COLUMN_STRING;
	}
	else if (type == "BOOLEAN")
	{
		return DataType::COLUMN_INT8_T;
	}
	else
	{
		return DataType::CONST_ERROR;
	}
}
