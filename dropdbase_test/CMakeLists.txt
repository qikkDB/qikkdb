
set(TEST_FILES  ${CMAKE_CURRENT_SOURCE_DIR}/BlockTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/ColumnTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DatabaseTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DispatcherTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/TableTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/GPUFilterTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUArtithmeticTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/GPUAggregationTests.cu
				${CMAKE_CURRENT_SOURCE_DIR}/ClientHandlerTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/ColumnTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DatabaseTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/DispatcherTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/ProtocolMessageTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/TableTests.cpp
				${CMAKE_CURRENT_SOURCE_DIR}/TCPServerTests.cpp)

add_executable(${PROJECT_NAME}_test ${SOURCE_FILES} ${TEST_FILES})

target_include_directories(${PROJECT_NAME}_test PUBLIC ${GTEST_INCLUDE_DIR}) 




target_link_libraries(${PROJECT_NAME}_test ${prefix}protobuf${suffix})
target_link_libraries(${PROJECT_NAME}_test ${YAML_CPP_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_test ${Boost_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_test antlr4_static)
target_link_libraries(${PROJECT_NAME}_test gtest)
target_link_libraries(${PROJECT_NAME}_test gtest_main)