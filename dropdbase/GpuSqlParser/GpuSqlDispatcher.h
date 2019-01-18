//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H
#define DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H

#include <functional>
#include <vector>
#include <iostream>
#include <memory>
#include <array>
#include "MemoryStream.h"
#include "DataType.h"
#include "Database.h"

class GpuSqlDispatcher
{

private:
    std::vector<std::function<void()>> functions;
    MemoryStream arguments;
    int blockIndex;
    const std::shared_ptr<Database> &database;

    std::array<std::function<void()>, DataType::DATA_TYPE_SIZE * DATA_TYPE_SIZE> greaterFunctions = {
            std::bind(&GpuSqlDispatcher::greater, this)
    };


public:
    explicit GpuSqlDispatcher(const std::shared_ptr<Database> &database);

    void execute();

    void addFunction(std::function<void()> &&function);

    void load();

    void ret();

    void fil();

    void done();

    void greater();

    template<typename T, typename U>
    void greaterColConst() {

    }

    template<typename T, typename U>
    void greaterConstCol() {

    }

    template<typename T, typename U>
    void greaterColCol() {

    }

    template<typename T, typename U>
    void greaterConstConst() {

    }

    void less();

    template<typename T, typename U>
    void lessColConst() {

    }

    template<typename T, typename U>
    void lessConstCol() {

    }

    template<typename T, typename U>
    void lessColCol() {

    }

    template<typename T, typename U>
    void lessConstConst() {

    }

    void greaterEqual();

    template<typename T, typename U>
    void greaterEqualColConst() {

    }

    template<typename T, typename U>
    void greaterEqualConstCol() {

    }

    template<typename T, typename U>
    void greaterEqualColCol() {

    }

    template<typename T, typename U>
    void greaterEqualConstConst() {

    }

    void lessEqual();

    template<typename T, typename U>
    void lessEqualColConst() {

    }

    template<typename T, typename U>
    void lessEqualConstCol() {

    }

    template<typename T, typename U>
    void lessEqualColCol() {

    }

    template<typename T, typename U>
    void lessEqualConstConst() {

    }

    void equal();

    template<typename T, typename U>
    void equalColConst() {

    }

    template<typename T, typename U>
    void equalConstCol() {

    }

    template<typename T, typename U>
    void equalColCol() {

    }

    template<typename T, typename U>
    void equalConstConst() {

    }

    void notEqual();

    template<typename T, typename U>
    void notEqualColConst() {

    }

    template<typename T, typename U>
    void notEqualConstCol() {

    }

    template<typename T, typename U>
    void notEqualColCol() {

    }

    template<typename T, typename U>
    void notEqualConstConst() {

    }

    void logicalAnd();

    template<typename T, typename U>
    void logicalAndColConst() {

    }

    template<typename T, typename U>
    void logicalAndConstCol() {

    }

    template<typename T, typename U>
    void logicalAndColCol() {

    }

    template<typename T, typename U>
    void logicalAndConstConst() {

    }

    void logicalOr();

    template<typename T, typename U>
    void logicalOrColConst() {

    }

    template<typename T, typename U>
    void logicalOrConstCol() {

    }

    template<typename T, typename U>
    void logicalOrColCol() {

    }

    template<typename T, typename U>
    void logicalOrConstConst() {

    }

    void mul();

    template<typename T, typename U>
    void mulColConst() {

    }

    template<typename T, typename U>
    void mulConstCol() {

    }

    template<typename T, typename U>
    void mulColCol() {

    }

    template<typename T, typename U>
    void mulConstConst() {

    }

    void div();

    template<typename T, typename U>
    void divColConst() {

    }

    template<typename T, typename U>
    void divConstCol() {

    }

    template<typename T, typename U>
    void divColCol() {

    }

    template<typename T, typename U>
    void divConstConst() {

    }

    void add();

    template<typename T, typename U>
    void addColConst() {

    }

    template<typename T, typename U>
    void addConstCol() {

    }

    template<typename T, typename U>
    void addColCol() {

    }

    template<typename T, typename U>
    void addConstConst() {

    }

    void sub();

    template<typename T, typename U>
    void subColConst() {

    }

    template<typename T, typename U>
    void subConstCol() {

    }

    template<typename T, typename U>
    void subColCol() {

    }

    template<typename T, typename U>
    void subConstConst() {

    }

    void mod();

    template<typename T, typename U>
    void modColConst() {

    }

    template<typename T, typename U>
    void modConstCol() {

    }

    template<typename T, typename U>
    void modColCol() {

    }

    template<typename T, typename U>
    void modConstConst() {

    }

    void contains();

    template<typename T, typename U>
    void containsColConst() {

    }

    template<typename T, typename U>
    void containsConstCol() {

    }

    template<typename T, typename U>
    void containsColCol() {

    }

    template<typename T, typename U>
    void containsConstConst() {

    }

    void between();

    void logicalNot();

    template<typename T>
    void logicalNotCol() {

    }

    template<typename T>
    void logicalNotConst() {

    }

    void minus();

    template<typename T>
    void minusCol() {

    }

    template<typename T>
    void minusConst() {

    }

    void min();

    template<typename T>
    void minCol() {

    }

    void max();

    template<typename T>
    void maxCol() {

    }

    void sum();

    template<typename T>
    void sumCol() {

    }

    void count();

    template<typename T>
    void countCol() {

    }

    void avg();

    template<typename T>
    void avgCol() {

    }

    void groupBy();


    template<typename T>
    void addArgument(T argument, DataType dataType)
    {
        arguments.insert<int>(dataType);
        arguments.insert<T>(argument);
    }
};


#endif //DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H
