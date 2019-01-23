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
#include "../Types/ComplexPolygon.pb.h"
#include "../Types/Point.pb.h"
#include "../DataType.h"
#include "../Database.h"

class GpuSqlDispatcher;

template<typename T>
void loadConst(GpuSqlDispatcher &dispatcher);

template<typename T>
void loadCol(GpuSqlDispatcher &dispatcher);

void loadReg(GpuSqlDispatcher &dispatcher);


template<typename T>
void retConst(GpuSqlDispatcher &dispatcher);

template<typename T>
void retCol(GpuSqlDispatcher &dispatcher);

void retReg(GpuSqlDispatcher &dispatcher);

void fil(GpuSqlDispatcher &dispatcher);

void done(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void greaterColConst(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void greaterConstCol(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void greaterColCol(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void greaterConstConst(GpuSqlDispatcher &dispatcher);

void greaterRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void lessColConst(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void lessConstCol(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void lessColCol(GpuSqlDispatcher &dispatcher);

void lessRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void lessConstConst(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void greaterEqualColConst(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void greaterEqualConstCol(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void greaterEqualColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void greaterEqualConstConst(GpuSqlDispatcher &dispatcher);

void greaterEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void lessEqualColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void lessEqualConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void lessEqualColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void lessEqualConstConst(GpuSqlDispatcher &dispatcher);


void lessEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void equalColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void equalConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void equalColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void equalConstConst(GpuSqlDispatcher &dispatcher);


void equalRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void notEqualColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void notEqualConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void notEqualColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void notEqualConstConst(GpuSqlDispatcher &dispatcher);


void notEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void logicalAndColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void logicalAndConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void logicalAndColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void logicalAndConstConst(GpuSqlDispatcher &dispatcher);


void logicalAndRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void logicalOrColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void logicalOrConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void logicalOrColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void logicalOrConstConst(GpuSqlDispatcher &dispatcher);


void logicalOrRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void mulColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void mulConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void mulColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void mulConstConst(GpuSqlDispatcher &dispatcher);


void mulRegReg(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void divColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void divConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void divColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void divConstConst(GpuSqlDispatcher &dispatcher);


void divRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void addColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void addConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void addColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void addConstConst(GpuSqlDispatcher &dispatcher);


void addRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void subColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void subConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void subColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void subConstConst(GpuSqlDispatcher &dispatcher);


void subRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void modColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void modConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void modColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void modConstConst(GpuSqlDispatcher &dispatcher);


void modRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void containsColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void containsConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void containsColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void containsConstConst(GpuSqlDispatcher &dispatcher);


void containsRegReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void logicalNotCol(GpuSqlDispatcher &dispatcher);


template<typename T>
void logicalNotConst(GpuSqlDispatcher &dispatcher);


void logicalNotReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void minusCol(GpuSqlDispatcher &dispatcher);


template<typename T>
void minusConst(GpuSqlDispatcher &dispatcher);


void minusReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void minCol(GpuSqlDispatcher &dispatcher);


template<typename T>
void minConst(GpuSqlDispatcher &dispatcher);


void minReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void maxCol(GpuSqlDispatcher &dispatcher);


template<typename T>
void maxConst(GpuSqlDispatcher &dispatcher);


void maxReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void sumCol(GpuSqlDispatcher &dispatcher);


template<typename T>
void sumConst(GpuSqlDispatcher &dispatcher);


void sumReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void countCol(GpuSqlDispatcher &dispatcher);


template<typename T>
void countConst(GpuSqlDispatcher &dispatcher);


void countReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void avgCol(GpuSqlDispatcher &dispatcher);


template<typename T>
void avgConst(GpuSqlDispatcher &dispatcher);


void avgReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void groupByConst(GpuSqlDispatcher &dispatcher);


template<typename T>
void groupByCol(GpuSqlDispatcher &dispatcher);


void groupByReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void invalidOperandTypesErrorHandlerColConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void invalidOperandTypesErrorHandlerConstCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void invalidOperandTypesErrorHandlerColCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void invalidOperandTypesErrorHandlerConstConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void invalidOperandTypesErrorHandlerRegCol(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void invalidOperandTypesErrorHandlerRegConst(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void invalidOperandTypesErrorHandlerColReg(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void invalidOperandTypesErrorHandlerConstReg(GpuSqlDispatcher &dispatcher);


class GpuSqlDispatcher
{

private:
    std::vector<std::function<void()>> functions;
    std::vector<std::function<void(GpuSqlDispatcher &)>> dispatcherFunctions;
    MemoryStream arguments;
    int blockIndex;
    const std::shared_ptr<Database> &database;

    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> greaterEqualFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> lessEqualFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> equalFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> notEqualFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalAndFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> logicalOrFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> mulFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> divFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> addFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> subFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> modFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> containsFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> logicalNotFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> minusFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> minFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> maxFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> sumFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> countFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> avgFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> loadFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> retFunctions;
    static std::array<std::function<void(GpuSqlDispatcher &)>,
            DataType::DATA_TYPE_SIZE> groupByFunctions;
    static std::function<void(GpuSqlDispatcher &)> filFunction;
    static std::function<void(GpuSqlDispatcher &)> doneFunction;

public:
    explicit GpuSqlDispatcher(const std::shared_ptr<Database> &database);

    void execute();

    void addGreaterFunction(DataType left, DataType right);

    void addLessFunction(DataType left, DataType right);

    void addGreaterEqualFunction(DataType left, DataType right);

    void addLessEqualFunction(DataType left, DataType right);

    void addEqualFunction(DataType left, DataType right);

    void addNotEqualFunction(DataType left, DataType right);

    void addLogicalAndFunction(DataType left, DataType right);

    void addLogicalOrFunction(DataType left, DataType right);

    void addMulFunction(DataType left, DataType right);

    void addDivFunction(DataType left, DataType right);

    void addAddFunction(DataType left, DataType right);

    void addSubFunction(DataType left, DataType right);

    void addModFunction(DataType left, DataType right);

    void addContainsFunction(DataType left, DataType right);

    void addLogicalNotFunction(DataType type);

    void addMinusFunction(DataType type);

    void addMinFunction(DataType type);

    void addMaxFunction(DataType type);

    void addSumFunction(DataType type);

    void addCountFunction(DataType type);

    void addAvgFunction(DataType type);

    void addLoadFunction(DataType type);

    void addRetFunction(DataType type);

    void addFilFunction();

    void addDoneFunction();

    void addGroupByFunction(DataType type);

    void addBetweenFunction(DataType op1, DataType op2, DataType op3);


    template<typename T>
    friend void loadConst(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void loadCol(GpuSqlDispatcher &dispatcher);

    friend void loadReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void retConst(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void retCol(GpuSqlDispatcher &dispatcher);

    friend void retReg(GpuSqlDispatcher &dispatcher);

    friend void fil(GpuSqlDispatcher &dispatcher);

    friend void done(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void greaterColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void greaterConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void greaterColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void greaterConstConst(GpuSqlDispatcher &dispatcher);

    friend void greaterRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void lessColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void lessConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void lessColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void lessConstConst(GpuSqlDispatcher &dispatcher);

    friend void lessRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void greaterEqualColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void greaterEqualConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void greaterEqualColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void greaterEqualConstConst(GpuSqlDispatcher &dispatcher);

    friend void greaterEqualRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void lessEqualColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void lessEqualConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void lessEqualColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void lessEqualConstConst(GpuSqlDispatcher &dispatcher);

    friend void lessEqualRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void equalColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void equalConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void equalColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void equalConstConst(GpuSqlDispatcher &dispatcher);

    friend void equalRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void notEqualColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void notEqualConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void notEqualColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void notEqualConstConst(GpuSqlDispatcher &dispatcher);

    friend void notEqualRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void logicalAndColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void logicalAndConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void logicalAndColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void logicalAndConstConst(GpuSqlDispatcher &dispatcher);

    friend void logicalAndRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void logicalOrColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void logicalOrConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void logicalOrColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void logicalOrConstConst(GpuSqlDispatcher &dispatcher);

    friend void logicalOrRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void mulColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void mulConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void mulColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void mulConstConst(GpuSqlDispatcher &dispatcher);

    friend void mulRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void divColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void divConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void divColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void divConstConst(GpuSqlDispatcher &dispatcher);

    friend void divRegReg(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void addColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void addConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void addColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void addConstConst(GpuSqlDispatcher &dispatcher);

    friend void addRegReg(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void subColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void subConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void subColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void subConstConst(GpuSqlDispatcher &dispatcher);

    friend void subRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void modColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void modConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void modColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void modConstConst(GpuSqlDispatcher &dispatcher);

    friend void modRegReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void containsColConst(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void containsConstCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void containsColCol(GpuSqlDispatcher &dispatcher);


    template<typename T, typename U>
    friend void containsConstConst(GpuSqlDispatcher &dispatcher);

    friend void containsRegReg(GpuSqlDispatcher &dispatcher);


    void between();

    template<typename T>
    friend void logicalNotCol(GpuSqlDispatcher &dispatcher);


    template<typename T>
    friend void logicalNotConst(GpuSqlDispatcher &dispatcher);

    friend void logicalNotReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void minusCol(GpuSqlDispatcher &dispatcher);


    template<typename T>
    friend void minusConst(GpuSqlDispatcher &dispatcher);

    friend void minusReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void minCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void minConst(GpuSqlDispatcher &dispatcher);

    friend void minReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void maxCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void maxConst(GpuSqlDispatcher &dispatcher);

    friend void maxReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void sumCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void sumConst(GpuSqlDispatcher &dispatcher);

    friend void sumReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void countCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void countConst(GpuSqlDispatcher &dispatcher);

    friend void countReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void avgCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void avgConst(GpuSqlDispatcher &dispatcher);

    friend void avgReg(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void groupByCol(GpuSqlDispatcher &dispatcher);

    template<typename T>
    friend void groupByConst(GpuSqlDispatcher &dispatcher);

    friend void groupByReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void invalidOperandTypesErrorHandlerColConst(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void invalidOperandTypesErrorHandlerConstCol(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void invalidOperandTypesErrorHandlerColCol(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void invalidOperandTypesErrorHandlerConstConst(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void invalidOperandTypesErrorHandlerColReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void invalidOperandTypesErrorHandlerConstReg(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void invalidOperandTypesErrorHandlerRegCol(GpuSqlDispatcher &dispatcher);

    template<typename T, typename U>
    friend void invalidOperandTypesErrorHandlerRegConst(GpuSqlDispatcher &dispatcher);

    template<typename T>
    void addArgument(T argument)
    {
        arguments.insert<T>(argument);
    }
};


class GpuSqlDispatcher;

template<typename T>
void loadConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T>
void loadCol(GpuSqlDispatcher &dispatcher)
{
    auto colName = dispatcher.arguments.read<std::string>();
    std::cout << "Load: " << colName << " " << typeid(T).name() << std::endl;
}

void loadReg(GpuSqlDispatcher &dispatcher);


template<typename T>
void retConst(GpuSqlDispatcher &dispatcher)
{
    T cnst = dispatcher.arguments.read<T>();
    std::cout << "RET: cnst" << typeid(T).name() << std::endl;
}

template<typename T>
void retCol(GpuSqlDispatcher &dispatcher)
{
    auto col = dispatcher.arguments.read<std::string>();
    std::cout << "RET: " << col << std::endl;

}

void retReg(GpuSqlDispatcher &dispatcher);

void fil(GpuSqlDispatcher &dispatcher);

void done(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void greaterColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void greaterConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void greaterColCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void greaterConstConst(GpuSqlDispatcher &dispatcher)
{

}

void greaterRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void lessColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void lessConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void lessColCol(GpuSqlDispatcher &dispatcher)
{

}

void lessRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void lessConstConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void greaterEqualColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void greaterEqualConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void greaterEqualColCol(GpuSqlDispatcher &dispatcher)
{

}


template<typename T, typename U>
void greaterEqualConstConst(GpuSqlDispatcher &dispatcher)
{

}

void greaterEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void lessEqualColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void lessEqualConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void lessEqualColCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void lessEqualConstConst(GpuSqlDispatcher &dispatcher)
{

}

void lessEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void equalColConst(GpuSqlDispatcher &dispatcher)
{
    U cnst = dispatcher.arguments.read<U>();
    auto colName = dispatcher.arguments.read<std::string>();
    std::cout << "EqualColConst: " << colName << " " << "const" << std::endl;
}

template<typename T, typename U>
void equalConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void equalColCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void equalConstConst(GpuSqlDispatcher &dispatcher)
{

}

void equalRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void notEqualColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void notEqualConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void notEqualColCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void notEqualConstConst(GpuSqlDispatcher &dispatcher)
{

}

void notEqualRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void logicalAndColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void logicalAndConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void logicalAndColCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void logicalAndConstConst(GpuSqlDispatcher &dispatcher)
{

}

void logicalAndRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void logicalOrColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void logicalOrConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void logicalOrColCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void logicalOrConstConst(GpuSqlDispatcher &dispatcher)
{

}

void logicalOrRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void mulColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void mulConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void mulColCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void mulConstConst(GpuSqlDispatcher &dispatcher)
{

}

void mulRegReg(GpuSqlDispatcher &dispatcher);


template<typename T, typename U>
void divColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void divConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void divColCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void divConstConst(GpuSqlDispatcher &dispatcher)
{

}

void divRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void addColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void addConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void addColCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void addConstConst(GpuSqlDispatcher &dispatcher)
{

}

void addRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void subColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void subConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void subColCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void subConstConst(GpuSqlDispatcher &dispatcher)
{

}

void subRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void modColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void modConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void modColCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void modConstConst(GpuSqlDispatcher &dispatcher)
{

}

void modRegReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void containsColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void containsConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void containsColCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void containsConstConst(GpuSqlDispatcher &dispatcher)
{

}

void containsRegReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void logicalNotCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T>
void logicalNotConst(GpuSqlDispatcher &dispatcher)
{

}

void logicalNotReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void minusCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T>
void minusConst(GpuSqlDispatcher &dispatcher)
{

}

void minusReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void minCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T>
void minConst(GpuSqlDispatcher &dispatcher)
{

}

void minReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void maxCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T>
void maxConst(GpuSqlDispatcher &dispatcher)
{

}


void maxReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void sumCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T>
void sumConst(GpuSqlDispatcher &dispatcher)
{

}

void sumReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void countCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T>
void countConst(GpuSqlDispatcher &dispatcher)
{

}

void countReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void avgCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T>
void avgConst(GpuSqlDispatcher &dispatcher)
{

}

void avgReg(GpuSqlDispatcher &dispatcher);

template<typename T>
void groupByConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T>
void groupByCol(GpuSqlDispatcher &dispatcher)
{

}

void groupByReg(GpuSqlDispatcher &dispatcher);

template<typename T, typename U>
void invalidOperandTypesErrorHandlerColConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void invalidOperandTypesErrorHandlerConstCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void invalidOperandTypesErrorHandlerColCol(GpuSqlDispatcher &dispatcher)
{

}


template<typename T, typename U>
void invalidOperandTypesErrorHandlerConstConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void invalidOperandTypesErrorHandlerRegCol(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void invalidOperandTypesErrorHandlerRegConst(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void invalidOperandTypesErrorHandlerColReg(GpuSqlDispatcher &dispatcher)
{

}

template<typename T, typename U>
void invalidOperandTypesErrorHandlerConstReg(GpuSqlDispatcher &dispatcher)
{

}


#endif //DROPDBASE_INSTAREA_GPUSQLDISPATCHER_H
