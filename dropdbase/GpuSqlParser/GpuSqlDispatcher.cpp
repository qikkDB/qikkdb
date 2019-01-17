//
// Created by Martin Staňo on 2019-01-15.
//

#include "GpuSqlDispatcher.h"

//TODO:Dispatch implementation


GpuSqlDispatcher::GpuSqlDispatcher(const std::shared_ptr<Database> &database) : database(database)
{
    blockIndex = 0;
}

void GpuSqlDispatcher::execute()
{
    for (auto &function : functions)
    {
        function();
    }
}

void GpuSqlDispatcher::load()
{
    auto col(arguments.read<std::string>());
    std::cout << "Load " << /*col <<*/ std::endl;
}

void GpuSqlDispatcher::fil()
{
    std::cout << "Filter" << std::endl;
}

void GpuSqlDispatcher::ret()
{
    std::cout << "Return" << std::endl;

}

void GpuSqlDispatcher::done()
{
    std::cout << "Done" << std::endl;

}

void GpuSqlDispatcher::addFunction(std::function<void()> &&function)
{
    functions.emplace_back(function);
}

void GpuSqlDispatcher::greater()
{
    std::cout << "Greater" << std::endl;

}

void GpuSqlDispatcher::less()
{
    std::cout << "Less" << std::endl;
}

void GpuSqlDispatcher::greaterEqual()
{
    std::cout << "GreaterEq" << std::endl;
}

void GpuSqlDispatcher::lessEqual()
{
    std::cout << "LessEq" << std::endl;
}

void GpuSqlDispatcher::equal()
{
    std::cout << "Equal " << std::endl;
}

void GpuSqlDispatcher::notEqual()
{
    std::cout << "NotEqual" << std::endl;
}

void GpuSqlDispatcher::logicalAnd()
{
    std::cout << "And" << std::endl;
}

void GpuSqlDispatcher::logicalOr()
{
    std::cout << "Or" << std::endl;
}

void GpuSqlDispatcher::mul()
{
    std::cout << "Multiply" << std::endl;
}

void GpuSqlDispatcher::div()
{
    std::cout << "Divide" << std::endl;
}

void GpuSqlDispatcher::add()
{
    std::cout << "Add" << std::endl;
}

void GpuSqlDispatcher::sub()
{
    std::cout << "Subtract" << std::endl;
}

void GpuSqlDispatcher::mod()
{
    std::cout << "Modulo" << std::endl;
}

void GpuSqlDispatcher::contains()
{
    std::cout << "Contains" << std::endl;
}

void GpuSqlDispatcher::between()
{
    std::cout << "Between" << std::endl;
}

void GpuSqlDispatcher::logicalNot()
{
    std::cout << "Not" << std::endl;
}

void GpuSqlDispatcher::minus()
{
    std::cout << "Minus" << std::endl;
}

void GpuSqlDispatcher::min()
{
    std::cout << "Min" << std::endl;
}

void GpuSqlDispatcher::max()
{
    std::cout << "Max" << std::endl;
}

void GpuSqlDispatcher::sum()
{
    std::cout << "Sum" << std::endl;
}

void GpuSqlDispatcher::count()
{
    std::cout << "Count" << std::endl;
}

void GpuSqlDispatcher::avg()
{
    std::cout << "Average" << std::endl;
}

void GpuSqlDispatcher::groupBy()
{
    std::cout << "Group by" << std::endl;
}