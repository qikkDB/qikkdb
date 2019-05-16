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

bitwise_operations = ["bitwiseOr", "bitwiseAnd", "bitwiseXor", "bitwiseLeftShift", "bitwiseRightShift"]
arithmetic_operations = ["mul", "div", "add", "sub", "mod", "logarithm", "power"]
unary_arithmetic_operations = ['minus', 'absolute', 'sine', 'cosine', 'tangent', 'cotangent', 'arcsine', 'arccosine', 'arctangent',
                               'logarithm10', 'logarithmNatural', 'exponential', 'squareRoot', 'square', 'sign',
                               'round', 'floor', 'ceil']
geo_operations = ["contains"]
polygon_operations = ["intersect", "union"]
filter_operations = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual"]
logical_operations = ["logicalAnd", "logicalOr"]

numeric_types = [INT, LONG, FLOAT, DOUBLE]
floating_types = [FLOAT, DOUBLE]
geo_types = [POINT, POLYGON]
bool_types = [BOOL]

operations_binary = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual", "logicalAnd", "logicalOr",
                     "mul", "div", "add", "sub", "mod", "contains", "intersect", "union"]
operations_filter = ["greater", "less", "greaterEqual", "lessEqual", "equal", "notEqual"]
operations_logical = ["logicalAnd", "logicalOr"]
operations_arithmetic = ["mul", "div", "add", "sub", "mod", "bitwiseOr", "bitwiseAnd", "bitwiseXor", "bitwiseLeftShift",
                         "bitwiseRightShift", "power", "logarithm", "arctangent2", "root"]
operations_unary = ["logicalNot", "minus", "min", "max", "sum", "count", "avg", "year", "month", "day", "hour",
                    "minute", "second"]
operations_aggregation = ["min", "max", "sum", "count", "avg"]
operations_date = ["year", "month", "day", "hour", "minute", "second"]
operations_move = ["ld", "ret", "groupBy"]
operations_ternary = ["between"]

for operation in operations_binary:
    declaration = "std::array<GpuSqlDispatcher::DispatchFunction," \
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
            function = "GpuSqlDispatcher::" + op + col + row + "<" + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
print()

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
    print("static std::array<GpuSqlDispatcher::DispatchFunction, " \
          "DataType::DATA_TYPE_SIZE> " + operation + "Functions;")

print('\n')

for operation in operations_unary:
    declaration = "std::array<GpuSqlDispatcher::DispatchFunction, " \
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
        function = "GpuSqlDispatcher::" + op + col + "<" + colVal + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)
print()

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
    print("static std::array<GpuSqlDispatcher::DispatchFunction, " \
          "DataType::DATA_TYPE_SIZE> " + operation + "Functions;")

print('\n')

for operation in operations_move:
    declaration = "std::array<GpuSqlDispatcher::DispatchFunction, " \
                  "DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types):
            col = "Col"

        if (operation == 'groupBy') and (colVal == STRING or colVal == BOOL or colVal in geo_types):
            function = "GpuSqlDispatcher::" + "invalidOperandTypesErrorHandler" + col + "<" + colVal + ">"
        elif (operation == 'ret') and (colVal == BOOL):
            function = "GpuSqlDispatcher::" + "invalidOperandTypesErrorHandler" + col + "<" + colVal + ">"
        else:
            function = "GpuSqlDispatcher::" + operation + col + "<" + colVal + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)
print()

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
    declaration = "std::array<GpuSqlDispatcher::DispatchFunction," \
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
            function = "GpuSqlDispatcher::" + op + col + row + "<FilterConditions::" + operation + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in operations_logical:
    declaration = "std::array<GpuSqlDispatcher::DispatchFunction," \
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
            function = "GpuSqlDispatcher::" + op + col + row + "<LogicOperations::" + operation + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in operations_arithmetic:
    declaration = "std::array<GpuSqlDispatcher::DispatchFunction," \
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

            elif colVal == BOOL or rowVal == BOOL:
                op = "invalidOperandTypesErrorHandler"

            elif (operation == "mod" or operation in bitwise_operations) and (
                    colVal in floating_types or rowVal in floating_types):
                op = "invalidOperandTypesErrorHandler"

            else:
                op = "arithmetic"

            function = "GpuSqlDispatcher::" + op + col + row + "<ArithmeticOperations::" + operation + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in unary_arithmetic_operations:
    declaration = "std::array<GpuSqlDispatcher::DispatchFunction," \
                  "DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types):
            col = "Col"

        if colVal in geo_types:
            op = "invalidOperandTypesErrorHandler"

        elif colVal == STRING:
            op = "invalidOperandTypesErrorHandler"

        elif colVal == BOOL:
            op = "invalidOperandTypesErrorHandler"

        else:
            op = "arithmeticUnary"

        function = "GpuSqlDispatcher::" + op + col + "<ArithmeticUnaryOperations::" + operation + ", " + colVal + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)
print()

operation = "insertInto"

declaration = "std::array<GpuSqlDispatcher::DispatchFunction," \
              "DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

for colIdx, colVal in enumerate(all_types):
    function = "GpuSqlDispatcher::" + operation + "<" + colVal + ">"

    if colIdx == len(all_types) - 1:
        declaration += ("&" + function + "};")
    else:
        declaration += ("&" + function + ", ")

print(declaration)
print()

# Aggregations (without group by)
for operation in operations_aggregation:
    declaration = "std::array<GpuSqlDispatcher::DispatchFunction, " \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + \
                  operation + "AggregationFunctions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types):
            col = "Col"

        if (colVal in geo_types and operation != "count") or (colVal == STRING) or (colVal == BOOL):
            op = "invalidOperandTypesErrorHandler"
        else:
            op = "aggregation"
        retVal = colVal
        if operation == "count":
            retVal = LONG
        # TODO: for avg FLOAT/DOUBLE
        if op != "invalidOperandTypesErrorHandler":
            function = "GpuSqlDispatcher::" + op + col + "<AggregationFunctions::" + operation + ", " + retVal + ", " + colVal + ">"
        else:
            function = "GpuSqlDispatcher::" + op + col + "<AggregationFunctions::" + operation + ", " + colVal + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)
print()

# Aggregations with group by
for operation in operations_aggregation:
    declaration = "std::array<GpuSqlDispatcher::DispatchFunction, " \
                  "DataType::DATA_TYPE_SIZE * DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + \
                  operation + "GroupByFunctions = {"

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

            if (col != "Col" or row != "Col") or \
                    (colVal in geo_types or colVal == STRING) or \
                    (rowVal in geo_types or rowVal == STRING) or (
                    rowVal == BOOL or colVal == BOOL):
                op = "invalidOperandTypesErrorHandler"
            else:
                op = "aggregationGroupBy"
            retVal = colVal
            if operation == "count":
                retVal = LONG
            # TODO: for avg FLOAT/DOUBLE
            if op != "invalidOperandTypesErrorHandler":
                function = "GpuSqlDispatcher::" + op + "<AggregationFunctions::" + operation + ", " + retVal + ", " + colVal + ", " + rowVal + ">"
            else:
                function = "GpuSqlDispatcher::" + op + col + row + "<AggregationFunctions::" + operation + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in operations_date:
    declaration = "std::array<GpuSqlDispatcher::DispatchFunction, " \
                  "DataType::DATA_TYPE_SIZE> GpuSqlDispatcher::" + operation + "Functions = {"

    for colIdx, colVal in enumerate(all_types):

        if colIdx < len(types):
            col = "Const"
        elif colIdx >= len(types):
            col = "Col"

        if colVal != LONG:
            op = "invalidOperandTypesErrorHandler"
            function = "GpuSqlDispatcher::" + op + col + "<DateOperations::" + operation + ", " + colVal + ">"
        else:
            op = "dateExtract"
            function = "GpuSqlDispatcher::" + op + col + "<DateOperations::" + operation + ">"

        if colIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in polygon_operations:
    declaration = "std::array<GpuSqlDispatcher::DispatchFunction," \
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

            if colVal != POLYGON or rowVal != POLYGON:
                op = "invalidOperandTypesErrorHandler"

            else:
                op = "polygonOperation"
            function = "GpuSqlDispatcher::" + op + col + row + "<PolygonFunctions::poly" + operation.capitalize() + ", " + colVal + ", " + rowVal + ">"

            if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
                declaration += ("&" + function + "};")
            else:
                declaration += ("&" + function + ", ")

    print(declaration)
print()

for operation in filter_operations + logical_operations:
    print('\n')
    print('__device__ GpuVMFunction add_gpu_' + operation + '_function(int32_t dataTypes)')
    print('{')
    print('\tswitch(dataTypes)')
    print('\t{')
    namespace = ''
    if operation in filter_operations:
        namespace = 'FilterConditions::'
    if operation in logical_operations:
        namespace = 'LogicOperations::'
    for colIdx, colVal in enumerate(types):
        for rowIdx, rowVal in enumerate(types):
            dataTypeCombination = colVal + ', ' + rowVal
            validCombination = True

            if colVal in geo_types or rowVal in geo_types:
                validCombination = False

            elif colVal == STRING or rowVal == STRING:
                validCombination = False

            if validCombination:
                print('\t\tcase ' + str(colIdx * len(types) + rowIdx) + ':')
                print(
                    '\t\t\treturn &filterFunction<' + namespace + operation + ", " + dataTypeCombination + ">;")
                print('\t\tbreak;')

    print('\t\tdefault:')
    print('\t\t\treturn &invalidArgumentTypeHandler<' + namespace + operation + '>;')
    print('\t\tbreak;')
    print('\t}')
    print('}')

for operation in arithmetic_operations:
    print('\n')
    print('__device__ GpuVMFunction add_gpu_' + operation + '_function(int32_t dataTypes)')
    print('{')
    print('\tswitch(dataTypes)')
    print('\t{')
    namespace = 'ArithmeticOperations::'
    for colIdx, colVal in enumerate(types):
        for rowIdx, rowVal in enumerate(types):
            dataTypeCombination = colVal + ', ' + colVal + ', ' + rowVal
            validCombination = True

            if colVal in geo_types or rowVal in geo_types:
                validCombination = False

            elif colVal == STRING or rowVal == STRING:
                validCombination = False

            elif operation == 'mod' and (colVal in floating_types or rowVal in floating_types):
                validCombination = False

            if validCombination:
                print('\t\tcase ' + str(colIdx * len(types) + rowIdx) + ':')
                print(
                    '\t\t\treturn &arithmeticFunction<' + namespace + operation + ", " + dataTypeCombination + ">;")
                print('\t\tbreak;')

    print('\t\tdefault:')
    print('\t\t\treturn &invalidArgumentTypeHandler<' + namespace + operation + '>;')
    print('\t\tbreak;')
    print('\t}')
    print('}')

for operation in operations_date:
    print('\n')
    print('__device__ GpuVMFunction add_gpu_' + operation + '_function(int32_t dataTypes)')
    print('{')
    print('\tswitch(dataTypes)')
    print('\t{')
    namespace = 'DateOperations::'
    for colIdx, colVal in enumerate(types):
        for rowIdx, rowVal in enumerate(types):
            dataTypeCombination = colVal
            validCombination = True

            if colVal != LONG or rowVal != LONG:
                validCombination = False

            if validCombination:
                print('\t\tcase ' + str(colIdx * len(types) + rowIdx) + ':')
                print(
                    '\t\t\treturn &dateFunction<' + namespace + operation + ">;")
                print('\t\tbreak;')

    print('\t\tdefault:')
    print('\t\t\treturn &invalidArgumentTypeHandler<' + namespace + operation + '>;')
    print('\t\tbreak;')
    print('\t}')
    print('}')

for operation in ['contains']:
    print('\n')
    print('__device__ GpuVMFunction add_gpu_' + operation + '_function(int32_t dataTypes)')
    print('{')
    print('\tswitch(dataTypes)')
    print('\t{')
    namespace = 'DateOperations::'
    for colIdx, colVal in enumerate(types):
        for rowIdx, rowVal in enumerate(types):
            dataTypeCombination = colVal
            validCombination = True

            if colVal != POLYGON or rowVal != POINT:
                validCombination = False

            if validCombination:
                print('\t\tcase ' + str(colIdx * len(types) + rowIdx) + ':')
                print(
                    '\t\t\treturn &containsFunction;')
                print('\t\tbreak;')

    print('\t\tdefault:')
    print('\t\t\treturn &invalidArgumentTypeHandler<' + namespace + operation + '>;')
    print('\t\tbreak;')
    print('\t}')
    print('}')

for operation in ['logicalNot']:
    print('\n')
    print('__device__ GpuVMFunction add_gpu_' + operation + '_function(int32_t dataTypes)')
    print('{')
    print('\tswitch(dataTypes)')
    print('\t{')
    namespace = 'LogicOperations::'
    for colIdx, colVal in enumerate(types):
        for rowIdx, rowVal in enumerate(types):
            dataTypeCombination = colVal
            validCombination = True

            if colVal in geo_types or rowVal in geo_types:
                validCombination = False

            elif colVal == STRING or rowVal == STRING:
                validCombination = False

            if validCombination:
                print('\t\tcase ' + str(colIdx * len(types) + rowIdx) + ':')
                print(
                    '\t\t\treturn &logicalNotFunction<' + namespace + operation + ' ,' + colVal + ">;")
                print('\t\tbreak;')

    print('\t\tdefault:')
    print('\t\t\treturn &invalidArgumentTypeHandler<' + namespace + operation + '>;')
    print('\t\tbreak;')
    print('\t}')
    print('}')

for operation in ['pushCol']:
    print('\n')
    print('__device__ GpuVMFunction add_gpu_' + operation + '_function(int32_t dataTypes)')
    print('{')
    print('\tswitch(dataTypes)')
    print('\t{')
    for colIdx, colVal in enumerate(types):
        for rowIdx, rowVal in enumerate(types):
            dataTypeCombination = colVal
            validCombination = True

            if colVal in geo_types or rowVal in geo_types:
                validCombination = False

            elif colVal == STRING or rowVal == STRING:
                validCombination = False

            if validCombination:
                print('\t\tcase ' + str(colIdx * len(types) + rowIdx) + ':')
                print(
                    '\t\t\treturn &pushColFunction<' + colVal + '>;')
                print('\t\tbreak;')

    print('\t\tdefault:')
    print('\t\t\treturn &invalidArgumentTypeHandler<' + namespace + operation + '>;')
    print('\t\tbreak;')
    print('\t}')
    print('}')

for operation in ['pushConst']:
    print('\n')
    print('__device__ GpuVMFunction add_gpu_' + operation + '_function(int32_t dataTypes)')
    print('{')
    print('\tswitch(dataTypes)')
    print('\t{')
    for colIdx, colVal in enumerate(types):
        for rowIdx, rowVal in enumerate(types):
            dataTypeCombination = colVal
            validCombination = True

            if colVal in geo_types or rowVal in geo_types:
                validCombination = False

            elif colVal == STRING or rowVal == STRING:
                validCombination = False

            if validCombination:
                print('\t\tcase ' + str(colIdx * len(types) + rowIdx) + ':')
                print(
                    '\t\t\treturn &pushConstFunction<' + colVal + '>;')
                print('\t\tbreak;')

    print('\t\tdefault:')
    print('\t\t\treturn &invalidArgumentTypeHandler<' + namespace + operation + '>;')
    print('\t\tbreak;')
    print('\t}')
    print('}')

operation = "point"
declaration = "std::array<GpuSqlDispatcher::DispatchFunction," \
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

        if col == "Const" and row == "Const":
            op = "invalidOperandTypesErrorHandler"
        elif colVal not in numeric_types or rowVal not in numeric_types:
            op = "invalidOperandTypesErrorHandler"
        else:
            op = operation

        function = "GpuSqlDispatcher::" + op + col + row + "<" + colVal + ", " + rowVal + ">"

        if colIdx == len(all_types) - 1 and rowIdx == len(all_types) - 1:
            declaration += ("&" + function + "};")
        else:
            declaration += ("&" + function + ", ")

print(declaration)
