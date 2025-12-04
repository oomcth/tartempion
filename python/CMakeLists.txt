cmake_minimum_required(VERSION 3.16)
project(tartempion LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(eigenpy 3.9 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

Python3_add_library(tartempion MODULE WITH_SOABI module.cpp)
 set_target_properties(
    tartempion
    PROPERTIES PREFIX ""
               SUFFIX ${PYTHON_EXT_SUFFIX}
               OUTPUT_NAME "${PYTHON_LIB_NAME}"
               LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tartempion"
               RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tartempion")
target_link_libraries(tartempion PRIVATE eigenpy::eigenpy fonction)

add_custom_command(TARGET tartempion POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:fonction> ${CMAKE_CURRENT_BINARY_DIR}/tartempion
)

set(PYTHON_SITE_PACKAGES
    "${CMAKE_INSTALL_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
)

if(EXISTS "${PYTHON_SITE_PACKAGES}/tartempion")
  message(STATUS "Removing old installation: ${PYTHON_SITE_PACKAGES}/tartempion")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${PYTHON_SITE_PACKAGES}/tartempion"
  )
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tartempion")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tartempion/__init__.py"
     "from .tartempion import *\n")

install(
    TARGETS tartempion
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}/tartempion"
)

install(
    TARGETS fonction
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}/tartempion"
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/tartempion/__init__.py"
    DESTINATION "${PYTHON_SITE_PACKAGES}/tartempion"
)
set_target_properties(tartempion PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "@loader_path;@loader_path/.."
    INSTALL_RPATH "@loader_path"
)


function(boostpy_add_stubs name)
    set(options VERBOSE)
    set(oneValueArgs MODULE OUTPUT PYTHON_PATH DEPENDS)
    set(multiValueArgs)
    cmake_parse_arguments(PARSE_ARGV 1 arg "${options}" "${oneValueArgs}" "${multiValueArgs}")

    if(NOT arg_MODULE)
        message(FATAL_ERROR "MODULE argument is required")
    endif()

    if(NOT arg_OUTPUT)
        message(FATAL_ERROR "OUTPUT argument is required")
    endif()

    if(NOT arg_PYTHON_PATH)
        set(pythonpath "")
    else()
        set(pythonpath "PYTHONPATH=${arg_PYTHON_PATH}")
    endif()

    if(arg_VERBOSE)
        set(loglevel "--log-level=DEBUG")
    endif()

    find_program(STUBGEN_EXECUTABLE pybind11-stubgen)
    set(stubgen_py
        ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../external-modules/pybind11-stubgen-e48d1f1/pybind11_stubgen.py
    )

    if(NOT EXISTS ${stubgen_py})
        message(
            FATAL_ERROR
            "Could not find 'pybind11_stubgen.py' at expected location: ${stubgen_py}"
        )
    endif()
    cmake_path(CONVERT ${stubgen_py} TO_CMAKE_PATH_LIST stubgen_py NORMALIZE)

    add_custom_command(
        OUTPUT ${arg_OUTPUT}
        COMMAND
            ${CMAKE_COMMAND} -E env ${pythonpath} ${Python3_EXECUTABLE} ${stubgen_py}
            -o ${arg_OUTPUT} ${arg_MODULE} ${loglevel} --boost-python
            --ignore-invalid=signature --no-setup-py --root-module-suffix ""
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${arg_DEPENDS}
        VERBATIM
        COMMENT "Generating boost python stubs for module '${arg_MODULE}'"
    )
    add_custom_target(${name} ALL DEPENDS ${arg_OUTPUT})
    if(arg_DEPENDS)
        add_dependencies(${name} ${arg_DEPENDS})
    endif()
endfunction()

boostpy_add_stubs(
    tartempion_stubs
    MODULE tartempion
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}"
    PYTHON_PATH ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS tartempion
    VERBOSE
)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/tartempion/__init__.pyi"
    DESTINATION "${PYTHON_SITE_PACKAGES}/tartempion"
)
install(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tartempion/tartempion"
    DESTINATION "${PYTHON_SITE_PACKAGES}/tartempion"
)

