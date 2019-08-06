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
    const std::shared_ptr<Database>& database_;
    int32_t blockIndex_;
    CpuSqlDispatcher& dispatcher_;
    std::unordered_map<std::string, GpuSqlParser::ExpressionContext*> columnAliasContexts_;
    std::unordered_map<std::string, std::string> tableAliases_;
    std::unordered_set<std::string> loadedTables_;
    std::unordered_map<std::string, std::string> shortColumnNames_;
    std::stack<std::pair<std::string, DataType>> parserStack_;

    void PushArgument(const char* token, DataType dataType);
    std::pair<std::string, DataType> StackTopAndPop();
    void StringToUpper(std::string& str);

    void PushTempResult(std::string reg, DataType type);

    bool IsLong(const std::string& value);

    bool IsDouble(const std::string& value);

    bool IsPoint(const std::string& value);

    bool IsPolygon(const std::string& value);

    void TrimDelimitedIdentifier(std::string& str);

    DataType GetReturnDataType(DataType left, DataType right);
    DataType GetReturnDataType(DataType operand);
    DataType GetDataTypeFromString(const std::string& dataType);

    void TrimReg(std::string& reg);

    std::pair<std::string, DataType> GenerateAndValidateColumnName(GpuSqlParser::ColumnIdContext* ctx);
    void WalkAliasExpression(const std::string& alias);
    bool insideAlias_;

public:
    CpuWhereListener(const std::shared_ptr<Database>& database, CpuSqlDispatcher& dispatcher);

    void exitBinaryOperation(GpuSqlParser::BinaryOperationContext* ctx) override;

    void exitTernaryOperation(GpuSqlParser::TernaryOperationContext* ctx) override;

    void exitUnaryOperation(GpuSqlParser::UnaryOperationContext* ctx) override;

    void exitCastOperation(GpuSqlParser::CastOperationContext* ctx) override;

    void exitIntLiteral(GpuSqlParser::IntLiteralContext* ctx) override;

    void exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext* ctx) override;

    void exitStringLiteral(GpuSqlParser::StringLiteralContext* ctx) override;

    void exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext* ctx) override;

    void exitGeoReference(GpuSqlParser::GeoReferenceContext* ctx) override;

    void exitVarReference(GpuSqlParser::VarReferenceContext* ctx) override;

    void exitDateTimeLiteral(GpuSqlParser::DateTimeLiteralContext* ctx) override;

    void exitPiLiteral(GpuSqlParser::PiLiteralContext* ctx) override;

    void exitNowLiteral(GpuSqlParser::NowLiteralContext* ctx) override;

    void exitWhereClause(GpuSqlParser::WhereClauseContext* ctx) override;

    void exitFromTables(GpuSqlParser::FromTablesContext* ctx) override;

    void ExtractColumnAliasContexts(GpuSqlParser::SelectColumnsContext* ctx);
};