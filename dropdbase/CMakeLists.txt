cmake_minimum_required(VERSION 3.12)

set(PROJECT_NAME dropdbase_instarea)
project(${PROJECT_NAME})
#enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})

find_package(gtest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_PREFIX_PATH /usr/local/lib/cmake/antlr4)
set(ANTLR4_JAR_LOCATION /usr/local/lib/antlr-4.7.2-complete.jar)

find_package(antlr4-runtime REQUIRED)
include_directories(${ANTLR4_INCLUDE_DIR})
find_package(antlr4-generator REQUIRED)

SET(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.68.0 COMPONENTS system filesystem log REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

include_directories(.)
include_directories(messages)
include_directories(Types)
include_directories(GpuSqlParser)

add_executable(${PROJECT_NAME}
        messages/CSVImportMessage.pb.cc
        messages/CSVImportMessage.pb.h
        messages/InfoMessage.pb.cc
        messages/InfoMessage.pb.h
        messages/QueryMessage.pb.cc
        messages/QueryMessage.pb.h
        messages/QueryResponseMessage.pb.cc
        messages/QueryResponseMessage.pb.h
        messages/SetDatabaseMessage.pb.cc
        messages/SetDatabaseMessage.pb.h
        Types/ComplexPolygon.pb.cc
        Types/ComplexPolygon.pb.h
        Types/Point.pb.cc
        Types/Point.pb.h
        BlockBase.h
        ColumnBase.cpp
        ColumnBase.h
        ComplexPolygonFactory.cpp
        ComplexPolygonFactory.h
        IBlock.h
        main.cpp
        NativeGeoPoint.h
        PointFactory.cpp
        PointFactory.h
        Database.cpp
        Database.h
        Configuration.cpp
        Configuration.h
        GpuSqlParser/GpuSqlCustomParser.cpp
        GpuSqlParser/GpuSqlCustomParser.h
        GpuSqlParser/ParserExceptions.h
        GpuSqlParser/GpuSqlListener.cpp
        GpuSqlParser/GpuSqlListener.h
        GpuSqlParser/GpuSqlDispatcher.cpp
        GpuSqlParser/GpuSqlDispatcher.h
        GpuSqlParser/MemoryStream.h
        GpuSqlParser/MemoryStream.cpp
        GpuSqlParser/QueryType.h
        GpuSqlParser/GpuSqlLexer.cpp
        GpuSqlParser/GpuSqlLexer.h
        GpuSqlParser/GpuSqlParser.cpp
        GpuSqlParser/GpuSqlParser.h
        GpuSqlParser/GpuSqlParserBaseListener.cpp
        GpuSqlParser/GpuSqlParserBaseListener.h
        GpuSqlParser/GpuSqlParserListener.cpp
        GpuSqlParser/GpuSqlParserListener.h
        GpuSqlParser/GpuSqlParser.cpp
        GpuSqlParser/GpuSqlParser.h
        IColumn.h
        Table.cpp
        Table.h
        DatabaseGenerator.cpp
        DatabaseGenerator.h)

target_link_libraries(${PROJECT_NAME} ${PROTOBUF_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${GTEST_BOTH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} antlr4_static)
