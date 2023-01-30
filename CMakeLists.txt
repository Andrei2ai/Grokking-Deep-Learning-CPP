cmake_minimum_required(VERSION 3.5)

project(Grokking-Deep-Learning-CPP)

################################################################################
## git
find_package(Git REQUIRED)

include(FetchContent)

FetchContent_Declare(NumCpp
    GIT_REPOSITORY https://github.com/dpilger26/NumCpp.git
)

set(NUMCPP_NO_USE_BOOST ON)
FetchContent_MakeAvailable(NumCpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ROOT_SRC ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(ch06)
add_subdirectory(ch08)
add_subdirectory(ch09)
