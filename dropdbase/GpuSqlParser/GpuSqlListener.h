//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLLISTENER_H
#define DROPDBASE_INSTAREA_GPUSQLLISTENER_H

#include "GpuSqlParserListener.h"
#include "Database.h"
#include "GpuSqlDispatcher.h"
#include <unordered_set>
#include <functional>
#include <string>
#include <memory>
#include <stack>

class GpuSqlListener : public GpuSqlParserListener {
private:
    const std::shared_ptr<Database> database;
    const std::shared_ptr<GpuSqlDispatcher> dispatcher;
    std::stack<std::string> parserStack;
    std::unordered_set<std::string> loadedTables;
    std::unordered_set<std::string> loadedColumns;
    std::unordered_set<std::string> groupByColumns;

    std::string stackTopAndPop();

public:
    GpuSqlListener(const std::shared_ptr<Database> &database, const std::shared_ptr<GpuSqlDispatcher> &dispatcher);

    void exitBinaryOperation(GpuSqlParser::BinaryOperationContext *ctx) override;


};


#endif //DROPDBASE_INSTAREA_GPUSQLLISTENER_H
