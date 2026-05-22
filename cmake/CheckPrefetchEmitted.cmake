cmake_minimum_required(VERSION 3.20)
separate_arguments(CROSS_INCLUDES_LIST NATIVE_COMMAND "${CROSS_INCLUDES}")
execute_process(
  COMMAND ${CLANG} -target aarch64-linux-gnu --sysroot=${SYSROOT}
          ${CROSS_INCLUDES_LIST} -mcpu=oryon-1 -O2
          -fpass-plugin=${PLUGIN} -S
          -o ${CMAKE_BINARY_DIR}/pf_check.s ${SRC}
  RESULT_VARIABLE rc ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "Compile failed: ${err}")
endif()
file(READ "${CMAKE_BINARY_DIR}/pf_check.s" asm)
string(REGEX MATCHALL "prfm" hits "${asm}")
list(LENGTH hits n)
if(n EQUAL 0)
  message(FATAL_ERROR "No prfm instructions emitted on oryon-1")
endif()
message(STATUS "prfm count=${n} OK")
