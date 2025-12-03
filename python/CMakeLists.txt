cmake_minimum_required(VERSION 3.16)
project(tartempion LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(eigenpy 3.9 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

Python3_add_library(tartempion MODULE WITH_SOABI module.cpp)
target_link_libraries(tartempion PRIVATE eigenpy::eigenpy fonction)

add_custom_command(TARGET tartempion POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:fonction> ${CMAKE_CURRENT_BINARY_DIR}/
)

set(PYTHON_SITE_PACKAGES
    "${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
)

install(TARGETS tartempion
        LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}/tartempion")

install(TARGETS fonction
        LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}/tartempion")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" "from .tartempion import *\n")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/__init__.py"
        DESTINATION "${PYTHON_SITE_PACKAGES}/tartempion")

set_target_properties(tartempion PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "@loader_path;@loader_path/.."
    INSTALL_RPATH "@loader_path"
)