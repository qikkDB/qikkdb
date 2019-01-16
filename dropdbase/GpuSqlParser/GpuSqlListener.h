//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLLISTENER_H
#define DROPDBASE_INSTAREA_GPUSQLLISTENER_H

#include "GpuSqlParserListener.h"
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

class GpuSqlListener : public GpuSqlParserListener {
private:
    const std::shared_ptr<Database> database;
    const std::shared_ptr<GpuSqlDispatcher> dispatcher;
    std::stack<std::tuple<std::string,DataType>> parserStack;
    std::unordered_set<std::string> loadedTables;
    std::unordered_set<std::string> loadedColumns;
    std::unordered_set<std::string> groupByColumns;

    bool usingGroupBy;
    bool insideAgg;

    int tempCounter;

    std::tuple<std::string,DataType> stackTopAndPop();
    std::string generateAndValidateColumnName(GpuSqlParser::VarReferenceContext *ctx);
    void pushTempResult();
    bool isLong(const std::string &value);
    bool isDouble(const std::string &value);
    bool isPoint(const std::string &value);
    bool isPolygon(const std::string &value);
    void stringToUpper(std::string &str);

public:
    GpuSqlListener(const std::shared_ptr<Database> &database, const std::shared_ptr<GpuSqlDispatcher> &dispatcher);

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

    void exitFromTables(GpuSqlParser::FromTablesContext *ctx) override;


};


#endif //DROPDBASE_INSTAREA_GPUSQLLISTENER_H
