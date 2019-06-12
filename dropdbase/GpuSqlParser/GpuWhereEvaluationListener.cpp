#include "GpuWhereEvaluationListener.h"
#include "../ColumnBase.h"
#include "../QueryEngine/GPUCore/GPUFilter.cuh"
#include "../QueryEngine/GPUCore/GPULogic.cuh"
#include "../QueryEngine/GPUCore/GPUArithmetic.cuh"
#include "../QueryEngine/GPUCore/GPUArithmeticUnary.cuh"

void GpuWhereEvaluationListener::exitBinaryOperation(GpuSqlParser::BinaryOperationContext * ctx)
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
	dispatcher.addBinaryOperation(leftOperandType, rightOperandType, op);

	std::string reg = getRegString(ctx);
	pushArgument(reg.c_str(), returnDataType);
	pushTempResult(reg, returnDataType);
}

void GpuWhereEvaluationListener::exitIntLiteral(GpuSqlParser::IntLiteralContext * ctx)
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

void GpuWhereEvaluationListener::exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext * ctx)
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

void GpuWhereEvaluationListener::exitStringLiteral(GpuSqlParser::StringLiteralContext * ctx)
{
	parserStack.push(std::make_pair(ctx->getText(), DataType::CONST_STRING));
}

void GpuWhereEvaluationListener::exitGeoReference(GpuSqlParser::GeoReferenceContext * ctx)
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

void GpuWhereEvaluationListener::exitVarReference(GpuSqlParser::VarReferenceContext * ctx)
{
	std::pair<std::string, DataType> tableColumnData = generateAndValidateColumnName(ctx->columnId());
	const DataType columnType = std::get<1>(tableColumnData);
	const std::string tableColumn = std::get<0>(tableColumnData);

	parserStack.push(std::make_pair(tableColumn, columnType));
}
