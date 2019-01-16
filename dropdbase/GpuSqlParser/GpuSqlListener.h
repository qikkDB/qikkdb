//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLLISTENER_H
#define DROPDBASE_INSTAREA_GPUSQLLISTENER_H

#include "GpuSqlParserListener.h"
#include "Database.h"
#include "GpuSqlDispatcher.h"
#include "DataTypes.h"
#include "ParserExceptions.h"
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
    std::stack<std::string> parserStack;
    std::unordered_set<std::string> loadedTables;
    std::unordered_set<std::string> loadedColumns;
    std::unordered_set<std::string> groupByColumns;

    bool usingGroupBy;
    bool insideAgg;

    std::string stackTopAndPop();
    std::string generateAndValidateColumnName(GpuSqlParser::VarReferenceContext *ctx);

public:
    GpuSqlListener(const std::shared_ptr<Database> &database, const std::shared_ptr<GpuSqlDispatcher> &dispatcher);

    void exitBinaryOperation(GpuSqlParser::BinaryOperationContext *ctx) override;

    void exitIntLiteral(GpuSqlParser::IntLiteralContext *ctx) override;

    void exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext *ctx) override;

    void exitStringLiteral(GpuSqlParser::StringLiteralContext *ctx) override;

    void exitVarReference(GpuSqlParser::VarReferenceContext *ctx) override;


};


#endif //DROPDBASE_INSTAREA_GPUSQLLISTENER_H
