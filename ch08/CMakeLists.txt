cmake_minimum_required(VERSION 3.5)


project(ch08 LANGUAGES CXX)

set(SRC ${SRC} main.cpp main.hpp)
set(SRC ${SRC} mnist_read.cpp mnist_read.hpp)

add_executable(${PROJECT_NAME} ${SRC})
target_compile_definitions(${PROJECT_NAME} PRIVATE -DNUMCPP_NO_USE_BOOST)
target_link_libraries(${PROJECT_NAME} PRIVATE NumCpp)
