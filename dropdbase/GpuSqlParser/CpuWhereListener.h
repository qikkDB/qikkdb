#pragma once
#include "GpuSqlListener.h"
#include "CpuSqlDispatcher.h"

#include "../DataType.h"
#include "../Database.h"
#include <any>
#include <stack>

class CpuWhereListener : public GpuSqlParserBaseListener
{
private:
	const std::shared_ptr<Database> &database;
	int32_t blockIndex;
	CpuSqlDispatcher &dispatcher;
	std::unordered_map<std::string, GpuSqlParser::ExpressionContext*> columnAliasContexts;
	std::unordered_map<std::string, std::string> tableAliases;
	std::unordered_set<std::string> loadedTables;
	std::unordered_map<std::string, std::string> shortColumnNames;
	std::stack<std::pair<std::string, DataType>> parserStack;

	void pushArgument(const char *token, DataType dataType);
	std::pair<std::string, DataType> stackTopAndPop();
	void stringToUpper(std::string &str);

	void pushTempResult(std::string reg, DataType type);

	bool isLong(const std::string &value);

	bool isDouble(const std::string &value);

	bool isPoint(const std::string &value);

	bool isPolygon(const std::string &value);

	void trimDelimitedIdentifier(std::string& str);

	DataType getReturnDataType(DataType left, DataType right);
	DataType getReturnDataType(DataType operand);
	DataType getDataTypeFromString(const std::string& dataType);

	void trimReg(std::string& reg);

	std::pair<std::string, DataType> generateAndValidateColumnName(GpuSqlParser::ColumnIdContext *ctx);
	void walkAliasExpression(const std::string & alias);
	bool insideAlias;

public:
	CpuWhereListener(const std::shared_ptr<Database> &database, CpuSqlDispatcher &dispatcher);

	void exitBinaryOperation(GpuSqlParser::BinaryOperationContext *ctx) override;

	void exitTernaryOperation(GpuSqlParser::TernaryOperationContext *ctx) override;

	void exitUnaryOperation(GpuSqlParser::UnaryOperationContext *ctx) override;

	void exitCastOperation(GpuSqlParser::CastOperationContext *ctx) override;

	void exitIntLiteral(GpuSqlParser::IntLiteralContext *ctx) override;

	void exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext *ctx) override;

	void exitStringLiteral(GpuSqlParser::StringLiteralContext *ctx) override;

	void exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext *ctx) override;

	void exitGeoReference(GpuSqlParser::GeoReferenceContext *ctx) override;

	void exitVarReference(GpuSqlParser::VarReferenceContext *ctx) override;

	void exitDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext *ctx) override;

	void exitPiLiteral(GpuSqlParser::PiLiteralContext *ctx) override;

	void exitNowLiteral(GpuSqlParser::NowLiteralContext *ctx) override;

	void exitWhereClause(GpuSqlParser::WhereClauseContext *ctx) override;

	void exitFromTables(GpuSqlParser::FromTablesContext *ctx) override;

	void ExtractColumnAliasContexts(GpuSqlParser::SelectColumnsContext * ctx);
	
};