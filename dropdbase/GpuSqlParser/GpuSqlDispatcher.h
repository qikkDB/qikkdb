//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H
#define DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H

#include <functional>
#include <vector>
#include <iostream>
#include "MemoryStream.h"

class GpuSqlDispatcher
{

private:
    std::vector<std::function<void()>> functions;
    MemoryStream arguments;

public:
    void execute();

    void addFunction(std::function<void()> function);

    void load();

    void ret();

    void fil();

    void done();

    void greater();

    void less();

    void greaterEqual();

    void lessEqual();

    void equal();

    void notEqual();

    void logicalAnd();

    void logicalOr();

    void mul();

    void div();

    void add();

    void sub();

    void mod();

    void contains();

    void between();

    void logicalNot();

    void minus();

    void min();

    void max();

    void sum();

    void count();

    void avg();

    void groupBy();

    template<typename T>
    void addArgument(T argument)
    {
        arguments.insert<T>(argument);
    }
};


#endif //DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H
