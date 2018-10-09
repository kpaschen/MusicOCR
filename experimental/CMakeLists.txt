cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
project(DisplayImage)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
file(GLOB_RECURSE musicocr_source src/*)
add_library(musicocr SHARED ${musicocr_source})
target_include_directories(musicocr PUBLIC ${OpenCV_INCLUDE_DIRS} include)
target_link_libraries(musicocr ${OpenCV_LIBS})

add_executable(DisplayImage display_image.cpp)
target_link_libraries(DisplayImage musicocr)

add_executable(OcrShell ocr_shell.cpp)
target_link_libraries(OcrShell musicocr)

add_executable(TrainKnn train_knn.cpp)
target_link_libraries(TrainKnn musicocr)

add_executable(TestKnn test_knn.cpp)
target_link_libraries(TestKnn musicocr)

find_package(GTest REQUIRED)
enable_testing()
file(GLOB musicocr_test_source_files test/*.cpp)
add_executable(musicocr_test ${musicocr_test_source_files})
target_include_directories(musicocr_test PUBLIC ${OpenCV_INCLUDE_DIRS} include ${GTEST_INCLUDE_DIRS})

target_link_libraries(musicocr_test ${GTEST_LIBRARIES} -lpthread -lm)
target_link_libraries(musicocr_test musicocr)
