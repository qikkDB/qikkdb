# CMakeList.txt : CMake project for TellStoryDB.NetworkClient, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

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
		${CMAKE_CURRENT_SOURCE_DIR}/NetworkClient.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/NetworkClient.h
		${CMAKE_CURRENT_SOURCE_DIR}/NetworkMessage.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/NetworkMessage.h
		${CMAKE_CURRENT_SOURCE_DIR}/DataType.h
		)

# Add source to this project's executable.
add_executable (TellStoryDB.NetworkClient "main.cpp" ${SOURCE_FILES})
IF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	target_link_libraries(TellStoryDB.NetworkClient PUBLIC ${prefix}protobufd${suffix})
ELSE(CMAKE_BUILD_TYPE_lower STREQUAL "debug")
	target_link_libraries(TellStoryDB.NetworkClient PUBLIC ${prefix}protobuf${suffix})
ENDIF(CMAKE_BUILD_TYPE_lower STREQUAL "debug")

target_link_libraries(TellStoryDB.NetworkClient PUBLIC ${Boost_LIBRARIES})
