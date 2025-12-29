#include "dik_cols.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/model.hpp"
#include <filesystem>
#include <pinocchio/parsers/urdf.hpp>

void TEST(pinocchio::Model &rmodel, bool echo_);