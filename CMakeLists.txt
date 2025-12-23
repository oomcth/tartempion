
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build (Debug, Release, RelWithDebInfo, MinSizeRel)")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")


set(PROJECT_NAME pinocchio-minimal)
set(PROJECT_DESCRIPTION
    "Minimal example of a simple application depending on Pinocchio")
set(PROJECT_URL "https://github.com/oomcth/tartempion")
add_compile_options(-Wall -Wextra -Wpedantic)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fsanitize=float-divide-by-zero
                        -fsanitize=float-cast-overflow
                        -fsanitize=undefined
                        -fno-sanitize-recover=all)

    add_link_options(-fsanitize=float-divide-by-zero
                     -fsanitize=float-cast-overflow
                     -fsanitize=undefined
                     -fno-sanitize-recover=all)
endif()

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wunused
)
project(${PROJECT_NAME})

include(FetchContent)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(DEFINED ENV{CONDA_PREFIX})
        set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}" CACHE PATH "Installation prefix" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Installation prefix" FORCE)
    endif()

endif()

option(ENABLE_TRACY "Enable Tracy profiler" OFF)
option(DISCARD_NON_CONVERGENT_IK "Discard non convergent IK" OFF)
option(GENERATE_PYTHON_STUBS "Generate and install Python stub files (.pyi)" ON)

if(ENABLE_LTO)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        message(STATUS "Enabling LTO for ${CMAKE_BUILD_TYPE}")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "LTO enabled but ignored for build type ${CMAKE_BUILD_TYPE} (only Release and RelWithDebInfo supported)")
    endif()
endif()

option(COLLISIONS_SUPPORT "Enable collisions support" ON)
if(COLLISIONS_SUPPORT)
    add_compile_definitions(COLLISIONS_SUPPORT)
endif()

option(EIGEN_RUNTIME_MALLOC_CHECK "Enable Eigen runtime malloc check" OFF)
if(EIGEN_RUNTIME_MALLOC_CHECK)
  message(STATUS "Eigen runtime malloc check enabled")
  add_compile_definitions(EIGEN_RUNTIME_NO_MALLOC)
endif()

option(GENERATE_COMPILE_COMMANDS "Export compile_commands.json" ON)

if(GENERATE_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

find_package(pinocchio REQUIRED)
find_package(coal REQUIRED)
find_package(proxsuite REQUIRED)
find_package(OpenMP REQUIRED)
find_package(diffcoal REQUIRED)

if (DISCARD_NON_CONVERGENT_IK)
    add_compile_definitions(DISCARD_NON_CONVERGENT_IK)
endif()

if(ENABLE_TRACY)
    add_compile_definitions(ENABLE_TRACY)
    include(FetchContent)
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG v0.11.1
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(tracy)
endif()

add_library(fonction SHARED fonction.cpp qppass/qp.cpp qppass/forward_pass.cpp qppass/dik_cols.cpp qppass/renorm.cpp qppass/SE3InductiveBias.cpp qppass/simpleSE3loss.cpp qppass/testing.cpp)
target_include_directories(fonction PUBLIC .)
target_compile_definitions(fonction PRIVATE EIGEN_DONT_PARALLELIZE)

add_executable(main main.cpp)
add_executable(qppass qppass/forward_pass.cpp qppass/qp.cpp qppass/dik_cols.cpp qppass/renorm.cpp qppass/SE3InductiveBias.cpp qppass/simpleSE3loss.cpp)

if(ENABLE_TRACY)
    foreach(tgt fonction main qppass)
        target_compile_definitions(${tgt} PRIVATE TRACY_ENABLE)
        target_link_libraries(${tgt} PRIVATE Tracy::TracyClient)
    endforeach()
endif()

target_link_libraries(fonction PUBLIC diffcoal::diffcoal coal::coal pinocchio::pinocchio proxsuite::proxsuite OpenMP::OpenMP_CXX)
target_link_libraries(qppass PUBLIC diffcoal::diffcoal coal::coal pinocchio::pinocchio proxsuite::proxsuite OpenMP::OpenMP_CXX)
target_link_libraries(main PUBLIC coal::coal pinocchio::pinocchio proxsuite::proxsuite fonction)

add_subdirectory(python)