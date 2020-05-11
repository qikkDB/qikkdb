include_directories(.)
set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/messages/CSVImportMessage.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/messages/CSVImportMessage.pb.h
        ${CMAKE_CURRENT_SOURCE_DIR}/messages/BulkImportMessage.pb.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/messages/BulkImportMessage.pb.h
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
        ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUError.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUMemoryCache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUMemory.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/Compression/Compression.h
		${CMAKE_CURRENT_SOURCE_DIR}/Compression/GPUCompression.h
        ${CMAKE_CURRENT_SOURCE_DIR}/BlockBase.h
        ${CMAKE_CURRENT_SOURCE_DIR}/BlockBase.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ColumnBase.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ColumnBase.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ComplexPolygonFactory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ComplexPolygonFactory.h
		${CMAKE_CURRENT_SOURCE_DIR}/StringFactory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/StringFactory.h
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
		${CMAKE_CURRENT_SOURCE_DIR}/DataType.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/ConstraintType.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/CudaLogBoost.cpp
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
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlParserBaseListener.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlParserListener.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlParser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlParser.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlDispatcher.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/CpuWhereListener.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/CpuWhereListener.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/CpuSqlDispatcher.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/CpuSqlDispatcher.h
		${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/Dispatcher/GpuSqlDispatcherInsertFunctions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/DispatcherCpu/CpuDispatcherLogicFunctions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/DispatcherCpu/CpuDispatcherArithmeticFunctions.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/DispatcherCpu/CpuDispatcherArithmeticUnaryFunctions.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/DispatcherCpu/CpuDispatcherDateFunctions.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/DispatcherCpu/CpuDispatcherGeoFunctions.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/DispatcherCpu/CpuDispatcherStringFunctions.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/DispatcherCpu/CpuDispatcherCastFunctions.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/IColumn.h
        ${CMAKE_CURRENT_SOURCE_DIR}/Table.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Table.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ConstraintViolationError.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ConstraintViolationError.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DatabaseGenerator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/DatabaseGenerator.h)

set(CUDA_FILES 	${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/CudaMemAllocator.cu 
                ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUJoin.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUGroupByMultiKey.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUGroupByString.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUPolygonClipping.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUReconstruct.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUStringBinary.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUStringUnary.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUOrderBy.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUCast.cu
				${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUDate.cu
				${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/CudaMemAllocator.cu
				${CMAKE_CURRENT_SOURCE_DIR}/Compression/GPUCompression.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/Dispatcher/GpuSqlDispatcherBinaryFunctions.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/Dispatcher/GpuSqlDispatcherUnaryFunctions.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/Dispatcher/GpuSqlDispatcherAggregationFunctions.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlJoinDispatcher.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/GpuSqlJoinFunctions.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/Dispatcher/GpuSqlDispatcherVMFunctions.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GpuSqlParser/Dispatcher/GpuSqlDispatcherOrderByFunctions.cu
                ${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUMemory.cu
				${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/GPUMergeJoin.cu
				${CMAKE_CURRENT_SOURCE_DIR}/QueryEngine/GPUCore/NullValues.cu)

if(WIN32)
	set_source_files_properties(${SOURCE_FILES} main.cpp PROPERTIES COMPILE_FLAGS "/wd4251 /wd4275 /wd4244 /wd4267 /wd4305 /bigobj")
	set_source_files_properties(${CUDA_FILES} PROPERTIES COMPILE_FLAGS "-arch sm_60 -Xcudafe=\"--diag_suppress=field_without_dll_interface --diag_suppress=base_class_has_different_dll_interface\" -Xcompiler=\"/bigobj /wd4251 /wd4275 /wd4244 /wd4267 /wd4305\" --expt-relaxed-constexpr")
endif(WIN32)
if(NOT WIN32)
    set_source_files_properties(${CUDA_FILES} PROPERTIES COMPILE_FLAGS "-arch sm_60 --expt-relaxed-constexpr")
endif(NOT WIN32)
add_library(${PROJECT_NAME}_obj STATIC ${SOURCE_FILES} ${CUDA_FILES})
set_target_properties(${PROJECT_NAME}_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME}_obj PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(MSVC)
    set_target_properties(${PROJECT_NAME}_obj PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif(MSVC)

if(NOT WIN32)
    set_target_properties(${PROJECT_NAME}_obj PROPERTIES LINK_FLAGS_DEBUG "-rdynamic")
endif(NOT WIN32)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

if(MSVC)
    set_target_properties(${PROJECT_NAME} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif(MSVC)

if(NOT WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS_DEBUG "-rdynamic")
endif(NOT WIN32)

string( TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_lower )

target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}_obj)

IF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	target_link_libraries(${PROJECT_NAME}_obj PUBLIC ${prefix}protobufd${suffix})
    target_link_libraries(${PROJECT_NAME}_obj PUBLIC ${prefix}yaml-cppmtd${suffix})
ELSE(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	target_link_libraries(${PROJECT_NAME}_obj PUBLIC ${prefix}protobuf${suffix}) 
    target_link_libraries(${PROJECT_NAME}_obj PUBLIC ${prefix}yaml-cppmt${suffix})
ENDIF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")

target_link_libraries(${PROJECT_NAME}_obj PUBLIC ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_obj PUBLIC antlr4_static)
