cmake_minimum_required (VERSION 3.2)
project (chilitags-server VERSION 0.1)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/module)

find_package(OpenCV REQUIRED core opencv_imgcodecs opencv_highgui opencv_imgproc opencv_videoio opencv_aruco)
if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "Could not find OpenCV")
endif()

find_package(redisimagehelper REQUIRED)
find_package(hiredis REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(ev REQUIRED)
find_package(cxxopts REQUIRED)
find_package(chilitags REQUIRED)

include_directories(${REDISIMAGEHELPER_INCLUDE_DIR})

set(CHILITAGS_DETECTION_SERVER
        src/server.cpp)
add_executable(chilitags-server ${CHILITAGS_DETECTION_SERVER})
target_link_libraries(chilitags-server ${OpenCV_LIBS} ${REDISIMAGEHELPER_LIBS} ${HIREDIS_LIBS} ${EV_LIBS} ${CHILITAGS_LIBS})
