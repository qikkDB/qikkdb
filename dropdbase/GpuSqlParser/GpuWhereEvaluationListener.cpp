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

	DataType returnDataType;
	bool ignoreFlag;

	if (op == ">")
	{
		
	}
	else if (op == "<")
	{
		
	}
	else if (op == ">=")
	{
		
	}
	else if (op == "<=")
	{
		
	}
	else if (op == "=")
	{
		
	}
	else if (op == "!=" || op == "<>")
	{
		
	}
	else if (op == "AND")
	{
		
	}
	else if (op == "OR")
	{
		
	}
	else if (op == "*")
	{
		
	}
	else if (op == "/")
	{
		
	}
	else if (op == "+")
	{
		
	}
	else if (op == "-")
	{
		
	}
	else if (op == "%")
	{
		
	}
	else if (op == "|")
	{
	}
	else if (op == "&")
	{
	}
	else if (op == "^")
	{
	}
	else if (op == "<<")
	{
	}
	else if (op == ">>")
	{
	}
	else if (op == "POINT")
	{
	}
	else if (op == "GEO_CONTAINS")
	{
	}
	else if (op == "GEO_INTERSECT")
	{
	}
	else if (op == "GEO_UNION")
	{
	}
	else if (op == "LOG")
	{
	}
	else if (op == "POW")
	{
	}
	else if (op == "ROOT")
	{
	}
	else if (op == "ATAN2")
	{
	}
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

std::pair<std::string, DataType> GpuWhereEvaluationListener::stackTopAndPop()
{
	std::pair<std::string, DataType> value = parserStack.top();
	parserStack.pop();
	return value;
}