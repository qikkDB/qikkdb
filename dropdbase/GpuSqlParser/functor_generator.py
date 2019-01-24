INT = "int32_t"
LONG = "int64_t"
FLOAT = "float"
DOUBLE = "double"
POINT = "ColmnarDB::Types::Point"
POLYGON = "ColmnarDB::Types::Polygon"
STRING = "std::string"
BOOL = "bool"
BYTE = "uint8_t"

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
             BOOL,
             BYTE]

arithmetic_operations = ["mul", "div", "add", "sub", "mod"]
geo_operations = ["contains"]
filter_operations = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual"]
logical_operations = ["logicalAnd", "logicalOr"]

numeric_types = [INT, LONG, FLOAT, DOUBLE]
geo_types = [POINT, POLYGON]
bool_types = [BYTE, BOOL]

operations_binary = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual", "logicalAnd", "logicalOr",
                     "mul", "div", "add", "sub", "mod", "contains"]
operations_unary = ["logicalNot", "minus", "min", "max", "sum", "count", "avg"]
operations_move = ["load", "ret", "groupBy"]
operations_ternary = ["between"]

for operation in operations_binary:
    declaration = "std::array<std::function<void(GpuSqlDispatcher &)>," \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):
        for rowIdx, rowVal in enumerate(all_types):

            if colIdx < len(types):
                col = "Const"
            elif colIdx >= len(types) and colVal != BYTE:
                col = "Col"
            else:
                col = "Reg"

            if rowIdx < len(types):
                row = "Const"
            elif rowIdx >= len(types) and rowVal != BYTE:
                row = "Col"
            else:
                row = "Reg"

            if row == "Reg" and col == "Reg":
                function = operation + col + row
            else:
                if row == "Reg" or col == "Reg":
                    op = "invalidOperandTypesErrorHandler"

                elif colVal in geo_types or rowVal in geo_types:
                    op = "invalidOperandTypesErrorHandler"

                elif colVal == STRING or rowVal == STRING:
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
    print("static std::array<std::function<void(GpuSqlDispatcher &)>, " \
          "DataType::DATA_TYPE_SIZE> " + operation + "Functions;")

print('\n')

for operation in operations_unary:
    declaration = "std::array<std::function<void(GpuSqlDispatcher &)>, " \
                  "DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types) and colVal != BYTE:
            col = "Col"
        else:
            col = "Reg"

        if col == "Reg":
            function = operation + col
        else:
            function = operation + col + "<" + colVal + ">"

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
    print("static std::array<std::function<void(GpuSqlDispatcher &)>, " \
          "DataType::DATA_TYPE_SIZE> " + operation + "Functions;")

print('\n')

for operation in operations_move:
    declaration = "std::array<std::function<void(GpuSqlDispatcher &)>, " \
                  "DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types) and colVal != BYTE:
            col = "Col"
        else:
            col = "Reg"

        if col == "Reg":
            function = operation + col
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
