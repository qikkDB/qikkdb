//
// Created by Martin Staňo on 2019-01-15.
//

#include "GpuSqlListener.h"

GpuSqlListener::GpuSqlListener(const std::shared_ptr<Database> &database,
                               GpuSqlDispatcher &dispatcher) : database(database), dispatcher(dispatcher)
{
    tempCounter = 0;
}


void GpuSqlListener::exitBinaryOperation(GpuSqlParser::BinaryOperationContext *ctx)
{
    std::tuple<std::string, DataType> right = stackTopAndPop();
    std::tuple<std::string, DataType> left = stackTopAndPop();

    std::string op = ctx->op->getText();
    stringToUpper(op);

    std::function<void()> function;

    if (op == ">")
    {
        function = std::bind(&GpuSqlDispatcher::greater, &dispatcher);
    } else if (op == "<")
    {
        function = std::bind(&GpuSqlDispatcher::less, &dispatcher);
    } else if (op == ">=")
    {
        function = std::bind(&GpuSqlDispatcher::greaterEqual, &dispatcher);
    } else if (op == "<=")
    {
        function = std::bind(&GpuSqlDispatcher::lessEqual, &dispatcher);
    } else if (op == "=")
    {
        function = std::bind(&GpuSqlDispatcher::equal, &dispatcher);
    } else if (op == "!=")
    {
        function = std::bind(&GpuSqlDispatcher::notEqual, &dispatcher);
    } else if (op == "AND")
    {
        function = std::bind(&GpuSqlDispatcher::logicalAnd, &dispatcher);
    } else if (op == "OR")
    {
        function = std::bind(&GpuSqlDispatcher::logicalOr, &dispatcher);
    } else if (op == "*")
    {
        function = std::bind(&GpuSqlDispatcher::mul, &dispatcher);
    } else if (op == "/")
    {
        function = std::bind(&GpuSqlDispatcher::div, &dispatcher);
    } else if (op == "+")
    {
        function = std::bind(&GpuSqlDispatcher::add, &dispatcher);
    } else if (op == "-")
    {
        function = std::bind(&GpuSqlDispatcher::sub, &dispatcher);
    } else if (op == "%")
    {
        function = std::bind(&GpuSqlDispatcher::mod, &dispatcher);
    } else if (op == "CONTAINS")
    {
        function = std::bind(&GpuSqlDispatcher::contains, &dispatcher);
    }
    dispatcher.addFunction(std::move(function));
    pushTempResult();
}


void GpuSqlListener::exitTernaryOperation(GpuSqlParser::TernaryOperationContext *ctx)
{
    std::tuple<std::string, DataType> op1 = stackTopAndPop();
    std::tuple<std::string, DataType> op2 = stackTopAndPop();
    std::tuple<std::string, DataType> op3 = stackTopAndPop();

    std::string op = ctx->op->getText();
    stringToUpper(op);

    std::function<void()> function;

    if (op == "BETWEEN")
    {
        function = std::bind(&GpuSqlDispatcher::between, &dispatcher);
    }
    dispatcher.addFunction(std::move(function));
    pushTempResult();
}

void GpuSqlListener::exitUnaryOperation(GpuSqlParser::UnaryOperationContext *ctx)
{
    std::tuple<std::string, DataType> right = stackTopAndPop();

    std::string op = ctx->op->getText();
    stringToUpper(op);

    std::function<void()> function;

    if (op == "!")
    {
        function = std::bind(&GpuSqlDispatcher::logicalNot, &dispatcher);
    } else if (op == "-")
    {
        function = std::bind(&GpuSqlDispatcher::minus, &dispatcher);
    }
    dispatcher.addFunction(std::move(function));
    pushTempResult();
}

void GpuSqlListener::exitAggregation(GpuSqlParser::AggregationContext *ctx)
{
    std::tuple<std::string, DataType> arg = stackTopAndPop();

    std::string op = ctx->AGG()->getText();
    stringToUpper(op);

    std::function<void()> function;

    if (op == "MIN")
    {
        function = std::bind(&GpuSqlDispatcher::min, &dispatcher);
    } else if (op == "MAX")
    {
        function = std::bind(&GpuSqlDispatcher::max, &dispatcher);
    } else if (op == "SUM")
    {
        function = std::bind(&GpuSqlDispatcher::sum, &dispatcher);
    } else if (op == "COUNT")
    {
        function = std::bind(&GpuSqlDispatcher::count, &dispatcher);
    } else if (op == "AVG")
    {
        function = std::bind(&GpuSqlDispatcher::avg, &dispatcher);
    }
    dispatcher.addFunction(std::move(function));
    pushTempResult();
}

void GpuSqlListener::exitSelectColumns(GpuSqlParser::SelectColumnsContext *ctx)
{
    std::function<void()> function = std::bind(&GpuSqlDispatcher::done, &dispatcher);
    dispatcher.addFunction(std::move(function));
}

void GpuSqlListener::exitSelectColumn(GpuSqlParser::SelectColumnContext *ctx)
{
    std::tuple<std::string, DataType> arg = stackTopAndPop();
    std::function<void()> function = std::bind(&GpuSqlDispatcher::ret, &dispatcher);
    dispatcher.addFunction(std::move(function));
    std::string strArg = std::get<0>(arg);
    dispatcher.addArgument<std::string>(strArg, DataType::COLUMN_INT);
}

void GpuSqlListener::exitFromTables(GpuSqlParser::FromTablesContext *ctx)
{

    for (auto table : ctx->table())
    {
        if (false /* TODO: !_database.tables.ContainsKey(table.GetText())*/)
        {
            throw TableNotFoundFromException();
        }
        loadedTables.insert(table->getText());
    }

}

void GpuSqlListener::exitWhereClause(GpuSqlParser::WhereClauseContext *ctx)
{
    std::tuple<std::string, DataType> arg = stackTopAndPop();
    std::function<void()> function = std::bind(&GpuSqlDispatcher::fil, &dispatcher);
    dispatcher.addFunction(std::move(function));
    std::string strArg = std::get<0>(arg);
    dispatcher.addArgument<std::string>(strArg, DataType::REG);
}

void GpuSqlListener::exitGroupByColumns(GpuSqlParser::GroupByColumnsContext *ctx)
{

    for (auto column : ctx->columnId())
    {
        std::string tableColumn = generateAndValidateColumnName(column);
        if (loadedColumns.find(tableColumn) == loadedColumns.end())
        {
            std::function<void()> function = std::bind(&GpuSqlDispatcher::load, &dispatcher);
            dispatcher.addFunction(std::move(function));
            dispatcher.addArgument<std::string>(tableColumn, DataType::COLUMN_INT);
            loadedColumns.insert(tableColumn);
        }
        if (groupByColumns.find(tableColumn) == groupByColumns.end())
        {
            std::function<void()> function = std::bind(&GpuSqlDispatcher::groupBy, &dispatcher);
            dispatcher.addFunction(std::move(function));
            dispatcher.addArgument<std::string>(tableColumn, DataType::COLUMN_INT);
            groupByColumns.insert(tableColumn);
        }
    }
    usingGroupBy = true;
}

void GpuSqlListener::exitIntLiteral(GpuSqlParser::IntLiteralContext *ctx)
{
    std::string token = ctx->getText();
    if (isLong(token))
    {
        parserStack.push(std::make_tuple(token, DataType::LONG));
        dispatcher.addArgument<long>(std::stol(token), DataType::LONG);
    } else
    {
        parserStack.push(std::make_tuple(token, DataType::INT));
        dispatcher.addArgument<int>(std::stoi(token), DataType::INT);
    }
}

void GpuSqlListener::exitDecimalLiteral(GpuSqlParser::DecimalLiteralContext *ctx)
{
    std::string token = ctx->getText();
    if (isDouble(token))
    {
        parserStack.push(std::make_tuple(token, DataType::DOUBLE));
        dispatcher.addArgument<double>(std::stod(token), DataType::DOUBLE);
    } else
    {
        parserStack.push(std::make_tuple(token, DataType::FLOAT));
        dispatcher.addArgument<float>(std::stol(token), DataType::FLOAT);
    }
}

void GpuSqlListener::exitStringLiteral(GpuSqlParser::StringLiteralContext *ctx)
{
    parserStack.push(std::make_tuple(ctx->getText(), DataType::STRING));
    dispatcher.addArgument<std::string>(ctx->getText(), DataType::STRING);
}


void GpuSqlListener::exitBooleanLiteral(GpuSqlParser::BooleanLiteralContext *ctx)
{
    parserStack.push(std::make_tuple(ctx->getText(), DataType::BOOLEAN));
    dispatcher.addArgument<std::string>(ctx->getText(), DataType::BOOLEAN);
}

void GpuSqlListener::exitVarReference(GpuSqlParser::VarReferenceContext *ctx)
{
    std::string tableColumn = generateAndValidateColumnName(ctx->columnId());

    if (loadedColumns.find(tableColumn) == loadedColumns.end())
    {
        std::function<void()> function = std::bind(&GpuSqlDispatcher::load, &dispatcher);
        dispatcher.addFunction(std::move(function));
        dispatcher.addArgument<std::string>(tableColumn, DataType::COLUMN_INT);
        loadedColumns.insert(tableColumn);
    }
    parserStack.push(std::make_tuple(ctx->getText(), DataType::COLUMN_INT));
    dispatcher.addArgument<std::string>(tableColumn, DataType::COLUMN_INT);
}

void GpuSqlListener::exitGeoReference(GpuSqlParser::GeoReferenceContext *ctx)
{

    auto start = ctx->start->getStartIndex();
    auto stop = ctx->stop->getStopIndex();
    antlr4::misc::Interval interval(start, stop);
    std::string geoValue = ctx->geometry()->start->getInputStream()->getText(interval);

    if (isPolygon(geoValue))
    {
        parserStack.push(std::make_tuple(geoValue, DataType::POLYGON));
        dispatcher.addArgument<std::string>(geoValue, DataType::POLYGON);
    } else if (isPoint(geoValue))
    {
        parserStack.push(std::make_tuple(geoValue, DataType::POINT));
        dispatcher.addArgument<std::string>(geoValue, DataType::POINT);
    }
}


std::string GpuSqlListener::generateAndValidateColumnName(GpuSqlParser::ColumnIdContext *ctx)
{
    std::string table;
    std::string column;

    std::string col = ctx->column()->getText();

    if (ctx->table())
    {
        table = ctx->table()->getText();
        column = ctx->column()->getText();

        if (loadedTables.find(table) == loadedTables.end())
        {
            throw TableNotFoundFromException();
        }
        if (false /*TODO: !_database.tables[table].ContainsColumn(column) */)
        {
            throw ColumnNotFoundException();
        }
    } else
    {
        int uses = 0;
        column = ctx->column()->getText();
        for (auto &tab : loadedTables)
        {
            if (false /* TODO: _database.tables[tab].ContainsColumn(col)*/)
            {
                table = tab;
                column = col;
                uses++;
            }
            if (uses > 1)
            {
                throw ColumnAmbiguityException();
            }
        }
        if (column.empty())
        {
            throw ColumnNotFoundException();
        }
    }

    std::string tableColumn = table + "." + column;

    if (usingGroupBy && !insideAgg && groupByColumns.find(tableColumn) == groupByColumns.end())
    {
        throw ColumnGroupByException();
    }

    return tableColumn;
}

std::tuple<std::string, DataType> GpuSqlListener::stackTopAndPop()
{
    std::tuple<std::string, DataType> value = parserStack.top();
    parserStack.pop();
    return value;
}

void GpuSqlListener::pushTempResult()
{
    std::string reg = std::string("R") + std::to_string(tempCounter);
    tempCounter++;
    parserStack.push(std::make_tuple(reg, REG));
    dispatcher.addArgument<std::string>(reg, REG);
}

bool GpuSqlListener::isLong(const std::string &value)
{
    try
    {
        std::stoi(value);
    }
    catch (std::out_of_range &e)
    {
        std::stol(value);
        return true;
    }
    return false;
}

bool GpuSqlListener::isDouble(const std::string &value)
{
    try
    {
        std::stof(value);
    }
    catch (std::out_of_range &e)
    {
        std::stod(value);
        return true;
    }
    return false;
}

bool GpuSqlListener::isPoint(const std::string &value)
{
    try
    {
        ColmnarDB::Types::Point point = PointFactory::FromWkt(value);
        return true;
    }
    catch (std::invalid_argument &e)
    {
        return false;
    }
}

bool GpuSqlListener::isPolygon(const std::string &value)
{
    try
    {
        ColmnarDB::Types::ComplexPolygon polygon = ComplexPolygonFactory::FromWkt(value);
        return true;
    }
    catch (std::invalid_argument &e)
    {
        return false;
    }
}

void GpuSqlListener::stringToUpper(std::string &str)
{
    for (auto &c : str)
    {
        c = toupper(c);
    }
}