
cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (Debug, Release, RelWithDebInfo, MinSizeRel)")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")


set(PROJECT_NAME pinocchio-minimal)
set(PROJECT_DESCRIPTION
    "Minimal example of a simple application depending on Pinocchio")
set(PROJECT_URL "http://github.com/stack-of-tasks/${PROJECT_NAME}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-Wall -Wextra -Wpedantic)

add_compile_options(
    -Wall                    # Warnings essentiels
    -Wextra                  # Warnings supplémentaires
    -Wpedantic               # Respect strict du standard
    # -Wsign-compare           # Comparaison signés/non-signés
    # -Wsign-conversion        # Conversion avec changement de signe
    # -Wconversion             # Toutes les conversions implicites
    -Wshadow                 # Variables qui en cachent d'autres
    -Wunused                 # Variables/fonctions non utilisées
)
project(${PROJECT_NAME})


option(EIGEN_ENABLE_RUNTIME_MALLOC_CHECK "Enable runtime checking of Eigen memory allocations" OFF)
if(EIGEN_ENABLE_RUNTIME_MALLOC_CHECK)
    add_compile_definitions(EIGEN_RUNTIME_NO_MALLOC)
endif()

# pinocchio dependency
find_package(pinocchio REQUIRED)
find_package(coal REQUIRED)
find_package(proxsuite REQUIRED)
find_package(OpenMP REQUIRED)
find_package(diffcoal REQUIRED)

add_library(fonction SHARED fonction.cpp qppass/qp.cpp qppass/forward_pass.cpp qppass/dik_cols.cpp qppass/renorm.cpp qppass/SE3InductiveBias.cpp qppass/simpleSE3loss.cpp)
target_include_directories(fonction PUBLIC .)
target_compile_definitions(fonction PRIVATE EIGEN_RUNTIME_NO_MALLOC EIGEN_DONT_PARALLELIZE)

# Create the main executable
add_executable(main main.cpp)
# add_executable(qp qppass/qp.cpp)
add_executable(qppass qppass/forward_pass.cpp qppass/qp.cpp qppass/dik_cols.cpp qppass/renorm.cpp qppass/SE3InductiveBias.cpp qppass/simpleSE3loss.cpp)

# Link main to its dependencies
target_link_libraries(fonction PUBLIC diffcoal::diffcoal coal::coal pinocchio::pinocchio proxsuite::proxsuite OpenMP::OpenMP_CXX)
# target_link_libraries(qp PUBLIC pinocchio::pinocchio OpenMP::OpenMP_CXX)
target_link_libraries(qppass PUBLIC diffcoal::diffcoal coal::coal pinocchio::pinocchio proxsuite::proxsuite OpenMP::OpenMP_CXX)
target_link_libraries(main PUBLIC coal::coal pinocchio::pinocchio proxsuite::proxsuite fonction)

add_subdirectory(python)
