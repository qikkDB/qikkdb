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
        //TODO:Arguments
    }
}


std::string GpuSqlListener::stackTopAndPop() {
    std::string value = parserStack.top();
    parserStack.pop();
    return value;
}
