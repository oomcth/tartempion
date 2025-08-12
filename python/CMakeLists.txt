# execute_process(
#   COMMAND /Users/mscheffl/miniconda3/envs/main/bin/python -c "import numpy; print(numpy.get_include())"
#   OUTPUT_VARIABLE NumPy_INCLUDE_DIR
#   OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# include_directories(${NumPy_INCLUDE_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


find_package(eigenpy 3.9 REQUIRED)

Python3_add_library(tartempion MODULE WITH_SOABI module.cpp)

target_link_libraries(tartempion PRIVATE eigenpy::eigenpy fonction)
