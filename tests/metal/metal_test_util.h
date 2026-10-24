// SPDX-License-Identifier: AGPL-3.0-or-later
//--------------------------------------------------------------------------------------------------
//> @brief Helpers of the GPU tests: read the shader sources from the source tree.
//> @details The constitutive shader is the concatenation of the files below in dependency order, with
//> local includes dropped; the same list as msl_sources_cp in src/CMakeLists.txt (embedded build).
//--------------------------------------------------------------------------------------------------
#ifndef DAMASK_METAL_TEST_UTIL_H
#define DAMASK_METAL_TEST_UTIL_H

#include <fstream>
#include <sstream>
#include <string>

inline std::string slurp(const std::string& f) { std::ifstream s(f); std::stringstream b; b << s.rdbuf(); return b.str(); }

inline std::string strip_local_includes(std::string s) {
  size_t p = 0;
  while ((p = s.find("#include \"")) != std::string::npos) s.erase(p, s.find('\n', p) - p);
  return s;
}

//> constitutive shader source from the directory src/grid/metal
inline std::string cp_shader_source(const std::string& dir) {
  std::string s;
  for (const char* f : {"df64.h", "cp_kernel.h", "cp_slim.h", "cp_core.h", "cp_models_composite.h", "cp.metal"})
    s += slurp(dir + "/" + f);
  return strip_local_includes(s);
}

#endif
