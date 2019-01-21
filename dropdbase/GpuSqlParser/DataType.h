//
// Created by Martin Staňo on 2019-01-16.
//

#ifndef DROPDBASE_INSTAREA_DATATYPE_H
#define DROPDBASE_INSTAREA_DATATYPE_H

enum DataType
{
    INT = 0,
    LONG = 1,
    FLOAT = 2,
    DOUBLE = 3,
    POINT = 4,
    POLYGON = 5,
    STRING = 6,
    BOOLEAN = 7,
    COLUMN_INT = 8,
    COLUMN_LONG = 9,
    COLUMN_FLOAT = 10,
    COLUMN_DOUBLE = 11,
    COLUMN_POINT = 12,
    COLUMN_POLYGON = 13,
    COLUMN_STRING = 14,
    COLUMN_BOOL = 15,
    REG = 16,
    DATA_TYPE_SIZE=17
};

#endif //DROPDBASE_INSTAREA_DATATYPE_H
