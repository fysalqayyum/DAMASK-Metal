# SPDX-License-Identifier: AGPL-3.0-or-later
# Tests of the Apple GPU (Metal) backend, built with 'cmake --build build --target metal-tests' and
# run with 'ctest --test-dir build -L cpu' (host-only) or '-L gpu' (needs an Apple GPU).
# Same flags as the validated builds: no FP contraction, C++17.
set(metal_dir ${PROJECT_SOURCE_DIR}/src/grid/metal)
add_custom_target(metal-tests)

function(metal_test name source label)
  add_executable(${name} EXCLUDE_FROM_ALL ${source} ${ARGN})
  target_compile_options(${name} PRIVATE -O2 -ffp-contract=off -std=c++17 $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_include_directories(${name} PRIVATE ${metal_dir})
  if(label STREQUAL "gpu")
    target_link_libraries(${name} PRIVATE "-framework Foundation" "-framework Metal"
                          "-framework MetalPerformanceShaders" "-framework MetalPerformanceShadersGraph")
  endif()
  add_dependencies(metal-tests ${name})
endfunction()

# host (double/df64 emulation): no GPU needed
metal_test(test_df64            test_df64.cpp            cpu)
metal_test(test_cp_kernel       test_cp_kernel.cpp       cpu)
metal_test(test_cp_slim         test_cp_slim.cpp         cpu)
metal_test(test_cp_core         test_cp_core.cpp         cpu)
metal_test(test_twintrip_policy test_twintrip_policy.cpp cpu)
foreach(t test_df64 test_cp_kernel test_cp_slim test_cp_core)
  add_test(NAME metal.${t} COMMAND ${t})
endforeach()
add_test(NAME metal.test_twintrip_policy COMMAND test_twintrip_policy ${CMAKE_CURRENT_SOURCE_DIR}/data/twintrip_alpha.gold)

# GPU: shaders are read from the source tree (not the embedded copy), so edits are tested directly
metal_test(test_cp_gpu          test_cp_gpu.mm           gpu ${metal_dir}/DAMASK_Metal.mm)
metal_test(test_cp_gpu_models   test_cp_gpu_models.mm    gpu ${metal_dir}/DAMASK_Metal.mm)
metal_test(test_gamma           test_gamma.mm            gpu ${metal_dir}/DAMASK_Metal.mm)
target_include_directories(test_gamma PRIVATE ${PETSC_INCLUDE_DIRS})                              # fftw3.h (PETSc download)
target_link_directories(test_gamma PRIVATE ${PETSC_LIBRARY_DIRS})
target_link_libraries(test_gamma PRIVATE fftw3)
add_test(NAME metal.test_cp_gpu        COMMAND test_cp_gpu        ${metal_dir})
add_test(NAME metal.test_cp_gpu_models COMMAND test_cp_gpu_models ${metal_dir})
add_test(NAME metal.test_gamma         COMMAND test_gamma         ${metal_dir}/spectral.metal)

# replay benchmark of captured batches (DAMASK_METAL_CAPTURE), a tool rather than a test
metal_test(bench_cp_replay      bench_cp_replay.mm       gpu ${metal_dir}/DAMASK_Metal.mm)

set_tests_properties(metal.test_df64 metal.test_cp_kernel metal.test_cp_slim metal.test_cp_core
                     metal.test_twintrip_policy PROPERTIES LABELS "cpu;metal")
set_tests_properties(metal.test_cp_core PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})   # data/Mg_geometry.txt
set_tests_properties(metal.test_cp_gpu metal.test_cp_gpu_models metal.test_gamma PROPERTIES
                     LABELS "gpu;metal" RUN_SERIAL TRUE)                                            # one GPU job at a time
