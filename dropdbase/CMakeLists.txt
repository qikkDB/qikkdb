cmake_minimum_required(VERSION 3.10)
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

set(PROJECT_NAME dropdbase_instarea)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

###### Google Protocol Buffers external project ########

set(PROTOBUF_TAR_GZ https://github.com/protocolbuffers/protobuf/archive/v3.6.1.tar.gz)

ExternalProject_Add(
  protobuf-external
  PREFIX protobuf
  URL ${PROTOBUF_TAR_GZ}
  BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf
  CMAKE_CACHE_ARGS
    "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
    "-Dprotobuf_BUILD_TESTS:BOOL=OFF"
    "-Dprotobuf_BUILD_EXAMPLES:BOOL=OFF"
    "-Dprotobuf_WITH_ZLIB:BOOL=OFF"
    "-DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}"
    # other project specific parameters
  SOURCE_SUBDIR cmake
  BUILD_ALWAYS 1
  STEP_TARGETS build
  INSTALL_COMMAND ""
)

EXternalProject_Get_Property(protobuf-external source_dir)
include_directories(${source_dir}/src)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/protobuf)

###### Google Protocol Buffers external project end ########

###### Antlr4 external project ########

add_definitions(-DANTLR4CPP_STATIC)
# add external build for antlrcpp
include(ExternalAntlr4Cpp)
# add antrl4cpp artifacts to project environment
include_directories(${ANTLR4_INCLUDE_DIRS})

###### Antlr4 external project ########



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
	QueryEngine/CudaMemAllocator.cu
        BlockBase.h
        BlockBase.cpp
        ColumnBase.cpp
        ColumnBase.h
        ComplexPolygonFactory.cpp
        ComplexPolygonFactory.h
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
        GpuSqlParser/GpuSqlDispatcher.cu
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
if(WIN32)
  set(prefix "")
  set(suffix ".lib")
elseif(APPLE)
  set(prefix "lib")
  set(suffix ".a")
else()
  set(prefix "lib")
  set(suffix ".a")
endif()

target_link_libraries(${PROJECT_NAME} ${prefix}protobuf${suffix})
#target_link_libraries(${PROJECT_NAME} ${GTEST_BOTH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} antlr4_static)
