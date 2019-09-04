

# For Intel Haswell, test for if the AVX2 and FMA ISAs work
set(CMP_CHECK_X64_INTEL_HASWELL
    TEST_AVX2
    TEST_FMA
    )

# For Intel Sandy Bridge, test for if the AVX ISA works
set(CMP_CHECK_X64_INTEL_SANDY_BRIDGE
    TEST_AVX
    )

# For Intel Core, test for if the SSE3 ISA works
set(CMP_CHECK_X64_INTEL_CORE
    TEST_SSE3
    )

# For AMD Bulldozer, test for if the AVX and FMA ISAs work
set(CMP_CHECK_X64_AMD_BULLDOZER
    TEST_AVX
    TEST_FMA
    )

# For the Cortex A57, test for if the VFPv4 and NEONv2 ISAs work
set(CMP_CHECK_ARMV8A_ARM_CORTEX_A57
    TEST_VFPv4
    TEST_NEONv2
    )

# For the Cortex A53, test for if the VFPv4 and NEONv2 ISAs work
set(CMP_CHECK_ARMV8A_ARM_CORTEX_A53
    TEST_VFPv4
    TEST_NEONv2
    )

# For the Cortex A15, test for if the VFPv3 and NEON ISAs work
set(CMP_CHECK_ARMV7A_ARM_CORTEX_A15
    TEST_VFPv3
    TEST_NEON
    )

# For the Cortex A7, test for if the VFPv3 and NEON ISAs work
set(CMP_CHECK_ARMV7A_ARM_CORTEX_A7
    TEST_VFPv3
    TEST_NEON
    )

# Iterate over each target to test the compilation
foreach(CHK_TARGET ${ALLOWED_TARGETS})
  # The main source file to test with
  set(CMP_CHECK_SRCS
      ${CMAKE_CURRENT_SOURCE_DIR}/compile_test.c
      )

  set(C_FLAGS_CHK "${CMAKE_C_FLAGS} ${C_FLAGS_TARGET_${CHK_TARGET}}")

  # Add the assembly test files and the compile definitions
  foreach(CHECK ${CMP_CHECK_${CHK_TARGET}})
      list( APPEND CMP_CHECK_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/isa_tests/${CHECK}.S )
      string(APPEND C_FLAGS_CHK " -D${CHECK}")
  endforeach()

  # If not crosscompiling, then try running the executable
  if( NOT CMAKE_CROSSCOMPILING )

    try_run( CHK_TARGET_RUN_${CHK_TARGET}                     # Variable to save the run result to
             CHK_TARGET_BUILD_${CHK_TARGET}                   # Variable to save the build result to
             "${CMAKE_BINARY_DIR}/compilerTest/${CHK_TARGET}" # Directory to compile in
             SOURCES ${CMP_CHECK_SRCS}                        # Source to compile
             CMAKE_FLAGS
              "-DCOMPILE_DEFINITIONS=${C_FLAGS_CHK}"
              "-DCMAKE_ASM_FLAGS=\"${CMAKE_ASM_FLAGS} ${ASM_FLAGS_TARGET_${CHK_TARGET}}\" "
             OUTPUT_VARIABLE CHK_OUTPUT${CHK_TARGET}
            )

    if(${CHK_TARGET_BUILD_${CHK_TARGET}})
      message(STATUS "Testing target ${CHK_TARGET} compilation: [done]")

      if(${CHK_TARGET_RUN_${CHK_TARGET}} STREQUAL "0")
        message(STATUS "Testing target ${CHK_TARGET} run: [done]")
      else()
        message(STATUS "Testing target ${CHK_TARGET} run: [failed]")
      endif()
    else()
      message(STATUS "Testing target ${CHK_TARGET} compilation: [failed]")
    endif()

  else()
    try_compile( CHK_TARGET_${CHK_TARGET}     # Variable to save the result to
                 "${CMAKE_BINARY_DIR}/compilerTest/${CHK_TARGET}"   # Directory to compile in
                 SOURCES ${CMP_CHECK_SRCS}    # Source to compile
                 CMAKE_FLAGS
                    "-DCOMPILE_DEFINITIONS=${C_FLAGS_CHK}"
                    "-DCMAKE_ASM_FLAGS=\"${CMAKE_ASM_FLAGS} ${ASM_FLAGS_TARGET_${CHK_TARGET}}\" "
                 OUTPUT_VARIABLE CHK_OUTPUT${CHK_TARGET}
                )
    if(${CHK_TARGET_BUILD_${CHK_TARGET}})
      message(STATUS "Testing target ${CHK_TARGET} compilation: [done]")
    else()
      message(STATUS "Testing target ${CHK_TARGET} compilation: [failed]")
    endif()

  endif()

  #message("Compile output: ${CHK_OUTPUT${CHK_TARGET}}")
endforeach()

