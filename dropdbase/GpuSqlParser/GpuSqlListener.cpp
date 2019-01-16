//
// Created by Martin Staňo on 2019-01-15.
//

#include "GpuSqlListener.h"

GpuSqlListener::GpuSqlListener(const std::shared_ptr<Database> &database,
                               const std::shared_ptr<GpuSqlDispatcher> &dispatcher) {
    tempCounter = 0;

}

void GpuSqlListener::exitBinaryOperation(GpuSqlParser::BinaryOperationContext *ctx) {
    std::tuple<std::string, DataType> right = stackTopAndPop();
    std::tuple<std::string, DataType> left = stackTopAndPop();

    std::string op = ctx->op->getText();
    stringToUpper(op);

    std::function<void()> function;

    if (op == ">") {
        function = std::bind(&GpuSqlDispatcher::greater, dispatcher.get());
    } else if (op == "<") {
        function = std::bind(&GpuSqlDispatcher::less, dispatcher.get());
    } else if (op == ">=") {
        function = std::bind(&GpuSqlDispatcher::greaterEqual, dispatcher.get());
    } else if (op == "<=") {
        function = std::bind(&GpuSqlDispatcher::lessEqual, dispatcher.get());
    } else if (op == "=") {
        function = std::bind(&GpuSqlDispatcher::equal, dispatcher.get());
    } else if (op == "!=") {
        function = std::bind(&GpuSqlDispatcher::notEqual, dispatcher.get());
    } else if (op == "AND") {
        function = std::bind(&GpuSqlDispatcher::logicalAnd, dispatcher.get());
    } else if (op == "OR") {
        function = std::bind(&GpuSqlDispatcher::logicalOr, dispatcher.get());
    } else if (op == "*") {
        function = std::bind(&GpuSqlDispatcher::mul, dispatcher.get());
    } else if (op == "/") {
        function = std::bind(&GpuSqlDispatcher::div, dispatcher.get());
    } else if (op == "+") {
        function = std::bind(&GpuSqlDispatcher::add, dispatcher.get());
    } else if (op == "-") {
        function = std::bind(&GpuSqlDispatcher::sub, dispatcher.get());
    } else if (op == "%") {
        function = std::bind(&GpuSqlDispatcher::mod, dispatcher.get());
    } else if (op == "CONTAINS") {
        function = std::bind(&GpuSqlDispatcher::contains, dispatcher.get());
    }
    dispatcher->addFunction(function);
    pushTempResult();
}


void GpuSqlListener::exitTernaryOperation(GpuSqlParser::TernaryOperationContext *ctx) {
    std::tuple<std::string, DataType> op1 = stackTopAndPop();
    std::tuple<std::string, DataType> op2 = stackTopAndPop();
    std::tuple<std::string, DataType> op3 = stackTopAndPop();

    std::string op = ctx->op->getText();
    stringToUpper(op);

    std::function<void()> function;

    if (op == "BETWEEN") {
        function = std::bind(&GpuSqlDispatcher::between, dispatcher.get());
    }
    dispatcher->addFunction(function);
    pushTempResult();
}

void GpuSqlListener::exitUnaryOperation(GpuSqlParser::UnaryOperationContext *ctx) {
    std::tuple<std::string, DataType> right = stackTopAndPop();

    std::string op = ctx->op->getText();
    stringToUpper(op);

    std::function<void()> function;

    if (op == "!") {
        function = std::bind(&GpuSqlDispatcher::logicalNot, dispatcher.get());
    } else if (op == "-") {
        function = std::bind(&GpuSqlDispatcher::minus, dispatcher.get());
    }
    dispatcher->addFunction(function);
    pushTempResult();
}

void GpuSqlListener::exitAggregation(GpuSqlParser::AggregationContext *ctx) {
    std::tuple<std::string, DataType> arg = stackTopAndPop();

    std::string op = ctx->AGG()->getText();
    stringToUpper(op);

    std::function<void()> function;

    if (op == "MIN") {
        function = std::bind(&GpuSqlDispatcher::min, dispatcher.get());
    } else if (op == "MAX") {
        function = std::bind(&GpuSqlDispatcher::max, dispatcher.get());
    } else if (op == "SUM") {
        function = std::bind(&GpuSqlDispatcher::sum, dispatcher.get());
    } else if (op == "COUNT") {
        function = std::bind(&GpuSqlDispatcher::count, dispatcher.get());
    } else if (op == "AVG") {
        function = std::bind(&GpuSqlDispatcher::avg, dispatcher.get());
    }
    dispatcher->addFunction(function);
    pushTempResult();
}


std::tuple<std::string, DataType> GpuSqlListener::stackTopAndPop() {
    std::tuple<std::string, DataType> value = parserStack.top();
    parserStack.pop();
    return value;
}

void GpuSqlListener::exitIntLiteral(GpuSqlParser::IntLiteralContext *ctx) {
    std::string token = ctx->getText();
    if (isLong(token)) {
        parserStack.push(std::make_tuple(token, LONG));
    } else {
        parserStack.push(std::make_tuple(token, INT));
    }
}

void GpuSqlListener::exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext *ctx) {
    std::string token = ctx->getText();
    if (isDouble(token)) {
        parserStack.push(std::make_tuple(token, DOUBLE));
    } else {
        parserStack.push(std::make_tuple(token, FLOAT));
    }
}

void GpuSqlListener::exitStringLiteral(GpuSqlParser::StringLiteralContext *ctx) {
    parserStack.push(std::make_tuple(ctx->getText(), STRING));
}


void GpuSqlListener::exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext *ctx) {
    parserStack.push(std::make_tuple(ctx->getText(), BOOLEAN));
}

void GpuSqlListener::exitVarReference(GpuSqlParser::VarReferenceContext *ctx) {
    std::string tableColumn = generateAndValidateColumnName(ctx);

    if (loadedColumns.find(tableColumn) == loadedColumns.end()) {
        std::function<void()> function = std::bind(&GpuSqlDispatcher::load, dispatcher.get());
        dispatcher->addFunction(function);
        dispatcher->addArgument<const char *>(tableColumn.data());
        loadedColumns.insert(tableColumn);
    }
    parserStack.push(std::make_tuple(ctx->getText(), COLUMN));
}

void GpuSqlListener::exitGeoReference(GpuSqlParser::GeoReferenceContext *ctx) {

    auto start = ctx->start->getStartIndex();
    auto stop = ctx->stop->getStopIndex();
    antlr4::misc::Interval interval(start, stop);
    std::string geoValue = ctx->geometry()->start->getInputStream()->getText(interval);

    if (isPolygon(geoValue)) {
        parserStack.push(std::make_tuple(geoValue, POLYGON));
    } else if (isPoint(geoValue)) {
        parserStack.push(std::make_tuple(geoValue, POINT));
    }
}


std::string GpuSqlListener::generateAndValidateColumnName(GpuSqlParser::VarReferenceContext *ctx) {
    std::string table;
    std::string column;

    std::string col = ctx->columnId()->column()->getText();

    if (ctx->columnId()->table()) {
        table = ctx->columnId()->table()->getText();
        column = ctx->columnId()->column()->getText();

        if (loadedTables.find(table) == loadedTables.end()) {
            throw TableNotFoundFromException();
        }
        if (true /*TODO: !_database.tables[table].ContainsColumn(column) */) {
            throw ColumnNotFoundException();
        }
    } else {
        int uses = 0;
        for (auto &tab : loadedTables) {
            if (true /* TODO: _database.tables[tab].ContainsColumn(col)*/) {
                table = tab;
                column = col;
                uses++;
            }
            if (uses > 1) {
                throw ColumnAmbiguityException();
            }
        }
        if (column.empty()) {
            throw ColumnNotFoundException();
        }
    }

    std::string tableColumn = table + "." + column;

    if (usingGroupBy && !insideAgg && groupByColumns.find(tableColumn) == groupByColumns.end()) {
        throw ColumnGroupByException();
    }

    return tableColumn;
}

void GpuSqlListener::exitFromTables(GpuSqlParser::FromTablesContext *ctx) {

    for (auto table : ctx->table()) {
        if (true /* TODO: !_database.tables.ContainsKey(table.GetText())*/) {
            throw TableNotFoundFromException();
        }
        loadedTables.insert(table->getText());
    }

}

void GpuSqlListener::pushTempResult() {
    std::string reg = std::string("R") + std::to_string(tempCounter);
    tempCounter++;
    parserStack.push(std::make_tuple(reg, REG));
}

bool GpuSqlListener::isLong(const std::string &value) {
    try {
        std::stoi(value);
    }
    catch (std::out_of_range &e) {
        std::stol(value);
        return true;
    }
    return false;
}

bool GpuSqlListener::isDouble(const std::string &value) {
    try {
        std::stof(value);
    }
    catch (std::out_of_range &e) {
        std::stod(value);
        return true;
    }
    return false;
}

bool GpuSqlListener::isPoint(const std::string &value) {
    try {
        ColmnarDB::Types::Point point = PointFactory::FromWkt(value);
        return true;
    }
    catch (std::invalid_argument &e) {
        return false;
    }
}

bool GpuSqlListener::isPolygon(const std::string &value) {
    try {
        ColmnarDB::Types::ComplexPolygon polygon = ComplexPolygonFactory::FromWkt(value);
        return true;
    }
    catch (std::invalid_argument &e) {
        return false;
    }
}

void GpuSqlListener::stringToUpper(std::string &str) {
    for (std::string::iterator p = str.begin(); str.end() != p; ++p) {
        *p = (char)(toupper(*p));
    }
}
