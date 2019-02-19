set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/messages/CSVImportMessage.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/messages/CSVImportMessage.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/messages/InfoMessage.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/messages/InfoMessage.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/messages/QueryMessage.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/messages/QueryMessage.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/messages/QueryResponseMessage.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/messages/QueryResponseMessage.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/messages/SetDatabaseMessage.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/messages/SetDatabaseMessage.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Types/ComplexPolygon.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Types/ComplexPolygon.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Types/Point.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Types/Point.pb.h
	    ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/CudaMemAllocator.cu
		${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUMemoryCache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/BlockBase.h
        ${CMAKE_CURRENT_SOURCE_DIR}/BlockBase.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ColumnBase.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ColumnBase.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ComplexPolygonFactory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ComplexPolygonFactory.h
		${CMAKE_CURRENT_SOURCE_DIR}/ConsoleHandler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ClientPoolWorker.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/ITCPWorker.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/NetworkMessage.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/TCPClientHandler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/NativeGeoPoint.h
        ${CMAKE_CURRENT_SOURCE_DIR}/PointFactory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/PointFactory.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Database.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Database.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Configuration.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Configuration.h
        ${CMAKE_CURRENT_SOURCE_DIR}/CSVDataImporter.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CSVDataImporter.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlCustomParser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlCustomParser.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/ParserExceptions.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlListener.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlListener.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlDispatcher.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/MemoryStream.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/MemoryStream.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/QueryType.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlLexer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlLexer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlParser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlParser.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlParserBaseListener.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlParserBaseListener.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlParserListener.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlParserListener.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlParser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlParser.h
        ${CMAKE_CURRENT_SOURCE_DIR}/IColumn.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Table.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Table.h
        ${CMAKE_CURRENT_SOURCE_DIR}/DatabaseGenerator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DatabaseGenerator.h)

set(CUDA_FILES 	${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/CudaMemAllocator.cu 
                ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlDispatcher.cu)

set(SOURCE_FILES ${SOURCE_FILES} PARENT_SCOPE)
set(CUDA_FILES ${CUDA_FILES} PARENT_SCOPE)

if(WIN32)
	set_source_files_properties(${SOURCE_FILES} main.cpp PROPERTIES COMPILE_FLAGS "/wd4251 /wd4275 /bigobj")
	set_source_files_properties(${CUDA_FILES} PROPERTIES COMPILE_FLAGS "-Xcudafe=\"--diag_suppress=field_without_dll_interface --diag_suppress=base_class_has_different_dll_interface\" -Xcompiler=\"/bigobj\"")
endif(WIN32)
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${CUDA_FILES} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_lower )

IF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	target_link_libraries(${PROJECT_NAME} ${prefix}protobufd${suffix})
ELSE(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	target_link_libraries(${PROJECT_NAME} ${prefix}protobuf${suffix})
ENDIF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")

target_link_libraries(${PROJECT_NAME} ${YAML_CPP_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME} antlr4_static)

