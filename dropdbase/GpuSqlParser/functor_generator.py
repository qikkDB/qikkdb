INT = "int32_t"
LONG = "int64_t"
FLOAT = "float"
DOUBLE = "double"
POINT = "ColmnarDB::Types::Point"
POLYGON = "ColmnarDB::Types::ComplexPolygon"
STRING = "std::string"
BOOL = "int8_t"

types = [INT,
         LONG,
         FLOAT,
         DOUBLE,
         POINT,
         POLYGON,
         STRING,
         BOOL]
all_types = [INT,
             LONG,
             FLOAT,
             DOUBLE,
             POINT,
             POLYGON,
             STRING,
             BOOL,
             INT,
             LONG,
             FLOAT,
             DOUBLE,
             POINT,
             POLYGON,
             STRING,
             BOOL]

arithmetic_operations = ["mul", "div", "add", "sub", "mod"]
geo_operations = ["contains"]
filter_operations = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual"]
logical_operations = ["logicalAnd", "logicalOr"]

numeric_types = [INT, LONG, FLOAT, DOUBLE]
floating_types = [FLOAT, DOUBLE]
geo_types = [POINT, POLYGON]
bool_types = [BOOL]

operations_binary = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual", "logicalAnd", "logicalOr",
                     "mul", "div", "add", "sub", "mod", "contains"]
operations_filter = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual"]
operations_logical = ["logicalAnd", "logicalOr"]
operations_arithmetic = ["mul", "div", "add", "sub", "mod"]
operations_unary = ["logicalNot", "minus", "min", "max", "sum", "count", "avg"]
operations_aggregation = ["min", "max", "sum", "count", "avg"]
operations_move = ["load", "ret", "groupBy"]
operations_ternary = ["between"]

for operation in operations_binary:
    declaration = "std::array<std::function<int32_t(GpuSqlDispatcher &)>," \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types):
                col = "Col"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types):
                row = "Col"

            if operation != 'contains' and (colVal in geo_types or rowVal in geo_types):
                op = "invalidOperandTypesErrorHandler"

            elif colVal == STRING or rowVal == STRING:
                op = "invalidOperandTypesErrorHandler"

            elif operation in arithmetic_operations and (colVal == BOOL or rowVal == BOOL):
                op = "invalidOperandTypesErrorHandler"

            elif operation == "mod" and (colVal in floating_types or rowVal in floating_types):
                op = "invalidOperandTypesErrorHandler"

            elif operation == "contains" and (colVal != POLYGON or rowVal != POINT):
                op = "invalidOperandTypesErrorHandler"

            else:
                op = operation
            function = op + col + row + "<" + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)

for operation in operations_binary:
    print("void add" + operation[0].upper() + operation[1:] + "Function(DataType left, DataType right);")

for operation in operations_binary:
    print(
        "void GpuSqlDispatcher::add" + operation[0].upper() + operation[1:] + "Function(DataType left, DataType right)")
    print('{')
    print("\tdispatcherFunctions.push_back(" + operation + "Functions[DataType::DATA_TYPE_SIZE * left + right]);")
    print('}')
    print('\n')

for operation in operations_unary:
    print("static std::array<std::function<int32_t(GpuSqlDispatcher &)>, " \
          "DataType::DATA_TYPE_SIZE> " + operation + "Functions;")

print('\n')

for operation in operations_unary:
    declaration = "std::array<std::function<int32_t(GpuSqlDispatcher &)>, " \
                  "DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types):
            col = "Col"

        if colVal in geo_types or colVal == STRING:
            op = "invalidOperandTypesErrorHandler"
        else:
            op = operation
        function = op + col + "<" + colVal + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)

for operation in operations_unary:
    print("void add" + operation[0].upper() + operation[1:] + "Function(DataType type);")

for operation in operations_unary:
    print(
        "void GpuSqlDispatcher::add" + operation[0].upper() + operation[1:] + "Function(DataType type)")
    print('{')
    print("\tdispatcherFunctions.push_back(" + operation + "Functions[type]);")
    print('}')
    print('\n')

print('\n')

for operation in operations_move:
    print("static std::array<std::function<int32_t(GpuSqlDispatcher &)>, " \
          "DataType::DATA_TYPE_SIZE> " + operation + "Functions;")

print('\n')

for operation in operations_move:
    declaration = "std::array<std::function<int32_t(GpuSqlDispatcher &)>, " \
                  "DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types):
            col = "Col"

        if (operation == 'ret' or operation == 'groupBy') and (colVal == STRING or colVal == BOOL or colVal in geo_types):
            function = "invalidOperandTypesErrorHandler" + col + "<" + colVal + ">"
        else:
            function = operation + col + "<" + colVal + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)

for operation in operations_move:
    print("void add" + operation[0].upper() + operation[1:] + "Function(DataType type);")

for operation in operations_move:
    print(
        "void GpuSqlDispatcher::add" + operation[0].upper() + operation[1:] + "Function(DataType type)")
    print('{')
    print("\tdispatcherFunctions.push_back(" + operation + "Functions[type]);")
    print('}')
    print('\n')

for operation in operations_filter:
    declaration = "std::array<std::function<int32_t(GpuSqlDispatcher &)>," \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types):
                col = "Col"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types):
                row = "Col"

            if colVal in geo_types or rowVal in geo_types:
                op = "invalidOperandTypesErrorHandler"

            elif colVal == STRING or rowVal == STRING:
                op = "invalidOperandTypesErrorHandler"

            elif operation in arithmetic_operations and (colVal == BOOL or rowVal == BOOL):
                op = "invalidOperandTypesErrorHandler"

            elif operation == "mod" and (colVal in floating_types or rowVal in floating_types):
                op = "invalidOperandTypesErrorHandler"

            else:
                op = "filter"
            function = op + col + row + "<FilterConditions::" + operation + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)

for operation in operations_logical:
    declaration = "std::array<std::function<int32_t(GpuSqlDispatcher &)>," \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types):
                col = "Col"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types):
                row = "Col"

            if colVal in geo_types or rowVal in geo_types:
                op = "invalidOperandTypesErrorHandler"

            elif colVal == STRING or rowVal == STRING:
                op = "invalidOperandTypesErrorHandler"

            elif operation in arithmetic_operations and (colVal == BOOL or rowVal == BOOL):
                op = "invalidOperandTypesErrorHandler"

            else:
                op = "logical"
            function = op + col + row + "<LogicOperations::" + operation + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)

for operation in operations_arithmetic:
    declaration = "std::array<std::function<int32_t(GpuSqlDispatcher &)>," \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types):
                col = "Col"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types):
                row = "Col"

            if colVal in geo_types or rowVal in geo_types:
                op = "invalidOperandTypesErrorHandler"

            elif colVal == STRING or rowVal == STRING:
                op = "invalidOperandTypesErrorHandler"

            elif operation in arithmetic_operations and (colVal == BOOL or rowVal == BOOL):
                op = "invalidOperandTypesErrorHandler"

            elif operation == "mod" and (colVal in floating_types or rowVal in floating_types):
                op = "invalidOperandTypesErrorHandler"

            else:
                op = "arithmetic"

            function = op + col + row + "<ArithmeticOperations::" + operation + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)


operation = "insertInto"

declaration = "std::array<std::function<int32_t(GpuSqlDispatcher &)>," \
                  "DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

for colIdx, colVal in enumerate(all_types):
    function = operation + "<" + colVal + ">"

    if colIdx == len(all_types) - 1:
        declaration += ("&" + function + "};")
    else:
        declaration += ("&" + function + ", ")

print(declaration)


for operation in operations_aggregation:
    declaration = "std::array<std::function<int32_t(GpuSqlDispatcher &)>, " \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types):
                col = "Col"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types):
                row = "Col"

            if (colVal in geo_types or colVal == STRING) or (rowVal in geo_types or rowVal == STRING) or (rowVal == BOOL or colVal == BOOL):
                op = "invalidOperandTypesErrorHandler"
            else:
                op = "aggregation"

            if operation == "count":
                colVal = LONG
            function = op + col + row + "<AggregationFunctions::" + operation + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)