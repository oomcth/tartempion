
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (Debug, Release, RelWithDebInfo, MinSizeRel)")
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

option(ENABLE_TRACY "Enable Tracy profiler" OFF)
option(DISCARD_NON_CONVERGENT_IK "Discard non convergent IK" OFF)

option(COLLISIONS_SUPPORT "Enable collisions support" OFF)
if(COLLISIONS_SUPPORT)
    add_compile_definitions(COLLISIONS_SUPPORT)
endif()

option(EIGEN_RUNTIME_MALLOC_CHECK "Enable Eigen runtime malloc check" OFF)
if(EIGEN_RUNTIME_MALLOC_CHECK)
  message(STATUS "Eigen runtime malloc check enabled")
  add_compile_definitions(ENABLE_EIGEN_RUNTIME_MALLOC_CHECK)
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

add_library(fonction SHARED fonction.cpp qppass/qp.cpp qppass/forward_pass.cpp qppass/dik_cols.cpp qppass/renorm.cpp qppass/SE3InductiveBias.cpp qppass/simpleSE3loss.cpp)
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