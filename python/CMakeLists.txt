
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(eigenpy 3.9 REQUIRED)

Python3_add_library(tartempion MODULE WITH_SOABI module.cpp)

target_link_libraries(tartempion PRIVATE eigenpy::eigenpy fonction)

