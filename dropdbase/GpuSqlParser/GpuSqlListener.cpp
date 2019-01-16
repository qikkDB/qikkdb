//
// Created by Martin Staňo on 2019-01-15.
//

#include "GpuSqlListener.h"

GpuSqlListener::GpuSqlListener(const std::shared_ptr<Database> &database,
                               const std::shared_ptr<GpuSqlDispatcher> &dispatcher) {

}

void GpuSqlListener::exitBinaryOperation(GpuSqlParser::BinaryOperationContext *ctx) {
    parserStack.push(ctx->op->getText());

    std::string operation = stackTopAndPop();
    std::string right = stackTopAndPop();
    std::string left = stackTopAndPop();

    if (operation == ">") {
        std::function<void()> function = std::bind(&GpuSqlDispatcher::greater, dispatcher.get());
        dispatcher->addFunction(function);
    }
}


std::string GpuSqlListener::stackTopAndPop() {
    std::string value = parserStack.top();
    parserStack.pop();
    return value;
}

void GpuSqlListener::exitIntLiteral(GpuSqlParser::IntLiteralContext *ctx) {
    parserStack.push(ctx->getText());
}

void GpuSqlListener::exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext *ctx) {
    parserStack.push(ctx->getText());
}

void GpuSqlListener::exitStringLiteral(GpuSqlParser::StringLiteralContext *ctx) {
    parserStack.push(ctx->getText());
}

void GpuSqlListener::exitVarReference(GpuSqlParser::VarReferenceContext *ctx) {
    std::string tableColumn = generateAndValidateColumnName(ctx);

    if(loadedColumns.find(tableColumn) == loadedColumns.end()) {
        std::function<void()> function = std::bind(&GpuSqlDispatcher::load, dispatcher.get());
        dispatcher->addFunction(function);
        loadedColumns.insert(tableColumn);
    }
    parserStack.push(tableColumn);
}

std::string GpuSqlListener::generateAndValidateColumnName(GpuSqlParser::VarReferenceContext *ctx) {
    std::string table;
    std::string column;

    std::string col = ctx->columnId()->column()->getText();

    if(ctx->columnId()->table()) {
        table = ctx->columnId()->table()->getText();
        column = ctx->columnId()->column()->getText();

        if(loadedTables.find(table) == loadedTables.end()) {
            throw TableNotFoundFromException();
        }
        if(true /*TODO: !_database.tables[table].ContainsColumn(column) */) {
            throw ColumnNotFoundException();
        }
    } else {
        int uses = 0;
        for(auto &tab : loadedTables) {
            if(true /* TODO: _database.tables[tab].ContainsColumn(col)*/) {
                table = tab;
                column = col;
                uses++;
            }
            if(uses > 1) {
                throw ColumnAmbiguityException();
            }
        }
        if(column.empty()) {
            throw ColumnNotFoundException();
        }
    }

    std::string tableColumn = table + "." + column;

    if(usingGroupBy && !insideAgg && groupByColumns.find(tableColumn) == groupByColumns.end()) {
        throw ColumnGroupByException();
    }

    return tableColumn;
}
