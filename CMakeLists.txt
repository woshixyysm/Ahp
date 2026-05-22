cmake_minimum_required(VERSION 3.20)
project(AArch64HeteroOpt C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_SANITIZERS "Enable sanitizers (address|undefined|thread)" "")
if(USE_SANITIZERS)
  add_compile_options(-fsanitize=${USE_SANITIZERS} -fno-omit-frame-pointer)
  add_link_options(-fsanitize=${USE_SANITIZERS})
endif()

option(ENABLE_LTO "Enable LTO for the plugin" OFF)
if(ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT lto_ok)
  if(lto_ok)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endif()

find_program(CLANG_TIDY_EXE NAMES "clang-tidy-21" "clang-tidy-20" "clang-tidy")
if(CLANG_TIDY_EXE AND EXISTS "${CMAKE_SOURCE_DIR}/.clang-tidy")
  set(AHP_CLANG_TIDY_CMD "${CLANG_TIDY_EXE}"
      "--config-file=${CMAKE_SOURCE_DIR}/.clang-tidy")
  message(STATUS "clang-tidy: ${CLANG_TIDY_EXE}")
endif()

# ---------------------------------------------------------------------------
# AArch64 cross-compile sysroot detection
# ---------------------------------------------------------------------------
if(NOT AARCH64_SYSROOT)
  foreach(_c "/usr/aarch64-linux-gnu" "/usr/local/aarch64-linux-gnu")
    if(EXISTS "${_c}/include")
      set(AARCH64_SYSROOT "${_c}")
      break()
    endif()
  endforeach()
endif()

if(NOT AARCH64_GCC_INCLUDES)
  file(GLOB _gcc_dirs
    "/usr/lib/gcc-cross/aarch64-linux-gnu/*/include"
    "/usr/lib/gcc/aarch64-linux-gnu/*/include")
  list(GET _gcc_dirs 0 AARCH64_GCC_INCLUDES)
endif()

if(NOT AARCH64_STDCXX_INCLUDES AND AARCH64_SYSROOT)
  file(GLOB _cxx_dirs "${AARCH64_SYSROOT}/include/c++/*")
  list(GET _cxx_dirs 0 _cxx_base)
  if(_cxx_base)
    set(AARCH64_STDCXX_INCLUDES
        "${_cxx_base}"
        "${_cxx_base}/aarch64-linux-gnu")
  endif()
endif()

# ---------------------------------------------------------------------------
# LLVM
# ---------------------------------------------------------------------------
find_package(LLVM QUIET CONFIG)

if(LLVM_FOUND)
  message(STATUS "Found LLVM ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}")
  if(LLVM_VERSION_MAJOR LESS 18)
    message(FATAL_ERROR "Ahp requires LLVM >= 18 (got ${LLVM_VERSION_MAJOR})")
  endif()

  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
  include(HandleLLVMOptions)

  separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
  add_definitions(${LLVM_DEFINITIONS_LIST})
  include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
  add_compile_definitions(HAS_LLVM_HEADERS=1)

  add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wunreachable-code -fno-rtti)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-O0 -g3)
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(-O2 -g -DNDEBUG)
  else()
    add_compile_options(-O3 -DNDEBUG)
  endif()

  add_llvm_pass_plugin(AArch64HeteroOpt lib/AArch64HeteroOpt.cpp)
  target_include_directories(AArch64HeteroOpt PRIVATE include)
  if(AHP_CLANG_TIDY_CMD)
    set_target_properties(AArch64HeteroOpt PROPERTIES
      CXX_CLANG_TIDY "${AHP_CLANG_TIDY_CMD}")
  endif()
  set_target_properties(AArch64HeteroOpt PROPERTIES
    OUTPUT_NAME "AArch64HeteroOpt" SOVERSION "0")

  install(TARGETS AArch64HeteroOpt LIBRARY DESTINATION lib/llvm/plugins)
  install(DIRECTORY include/
    DESTINATION include/AArch64HeteroOpt FILES_MATCHING PATTERN "*.h")

  enable_testing()

  # Standalone tests (no LLVM runtime linkage)
  foreach(test_name test_target_profile test_features)
    add_executable(${test_name} test/${test_name}.cpp)
    target_include_directories(${test_name} PRIVATE include)
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    add_test(NAME "${test_name}-standalone" COMMAND ${test_name})
  endforeach()

  # AArch64 cross-compile integration tests
  find_program(CLANG_EXE
    NAMES "clang-${LLVM_VERSION_MAJOR}" "clang"
    HINTS ${LLVM_TOOLS_BINARY_DIR})

  if(CLANG_EXE AND AARCH64_SYSROOT AND AARCH64_STDCXX_INCLUDES)
    message(STATUS "AArch64 sysroot: ${AARCH64_SYSROOT}")

    set(_cross_flags
      -target aarch64-linux-gnu
      --sysroot=${AARCH64_SYSROOT})
    foreach(_inc ${AARCH64_STDCXX_INCLUDES})
      list(APPEND _cross_flags -I${_inc})
    endforeach()
    # Add aarch64-linux-gnu subdir for bits/c++config.h
    list(APPEND _cross_flags
      -I${AARCH64_SYSROOT}/include/c++/13/aarch64-linux-gnu)
    if(AARCH64_GCC_INCLUDES)
      list(APPEND _cross_flags -I${AARCH64_GCC_INCLUDES})
    endif()

    foreach(TARGET_CPU oryon-1 ampere1)
      add_test(NAME "hetero-opt-${TARGET_CPU}"
        COMMAND ${CLANG_EXE} ${_cross_flags}
          -mcpu=${TARGET_CPU} -O2
          -fpass-plugin=$<TARGET_FILE:AArch64HeteroOpt>
          -S -o /dev/null
          ${CMAKE_SOURCE_DIR}/test/test_verify.cpp)
      set_tests_properties("hetero-opt-${TARGET_CPU}" PROPERTIES
        FAIL_REGULAR_EXPRESSION "error:|warning: -fpass-plugin")
    endforeach()


  else()
    message(STATUS "AArch64 integration tests skipped (no cross-compile sysroot)")
  endif()

else()
  message(STATUS "LLVM not found — standalone tests only")
  enable_testing()
  foreach(test_name test_target_profile test_features test_syntax)
    add_executable(${test_name} test/${test_name}.cpp)
    target_include_directories(${test_name} PRIVATE include)
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    target_compile_options(${test_name} PRIVATE -Wall -Wextra -Wno-unused-parameter)
    add_test(NAME "${test_name}-standalone" COMMAND ${test_name})
  endforeach()
endif()
