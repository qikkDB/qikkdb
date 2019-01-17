//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLLISTENER_H
#define DROPDBASE_INSTAREA_GPUSQLLISTENER_H

#include "GpuSqlParserBaseListener.h"
#include "Database.h"
#include "GpuSqlDispatcher.h"
#include "DataType.h"
#include "ParserExceptions.h"
#include "PointFactory.h"
#include "ComplexPolygonFactory.h"
#include <unordered_set>
#include <functional>
#include <string>
#include <memory>
#include <stack>
#include <regex>

class GpuSqlListener : public GpuSqlParserBaseListener
{
private:
    const std::shared_ptr<Database> &database;
    GpuSqlDispatcher &dispatcher;
    std::stack<std::tuple<std::string, DataType>> parserStack;
    std::unordered_set<std::string> loadedTables;
    std::unordered_set<std::string> loadedColumns;
    std::unordered_set<std::string> groupByColumns;

    bool usingGroupBy;
    bool insideAgg;

    int tempCounter;

    std::tuple<std::string, DataType> stackTopAndPop();

    std::string generateAndValidateColumnName(GpuSqlParser::ColumnIdContext *ctx);

    void pushTempResult();

    bool isLong(const std::string &value);

    bool isDouble(const std::string &value);

    bool isPoint(const std::string &value);

    bool isPolygon(const std::string &value);

    void stringToUpper(std::string &str);

public:
    GpuSqlListener(std::shared_ptr<Database> &database, GpuSqlDispatcher &dispatcher);

    void exitBinaryOperation(GpuSqlParser::BinaryOperationContext *ctx) override;

    void exitTernaryOperation(GpuSqlParser::TernaryOperationContext *ctx) override;

    void exitUnaryOperation(GpuSqlParser::UnaryOperationContext *ctx) override;

    void exitIntLiteral(GpuSqlParser::IntLiteralContext *ctx) override;

    void exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext *ctx) override;

    void exitStringLiteral(GpuSqlParser::StringLiteralContext *ctx) override;

    void exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext *ctx) override;

    void exitGeoReference(GpuSqlParser::GeoReferenceContext *ctx) override;

    void exitVarReference(GpuSqlParser::VarReferenceContext *ctx) override;

    void exitAggregation(GpuSqlParser::AggregationContext *ctx) override;

    void exitSelectColumns(GpuSqlParser::SelectColumnsContext *ctx) override;

    void exitSelectColumn(GpuSqlParser::SelectColumnContext *ctx) override;

    void exitFromTables(GpuSqlParser::FromTablesContext *ctx) override;

    void exitWhereClause(GpuSqlParser::WhereClauseContext *ctx) override;

    void exitGroupByColumns(GpuSqlParser::GroupByColumnsContext *ctx) override;


};


#endif //DROPDBASE_INSTAREA_GPUSQLLISTENER_H
