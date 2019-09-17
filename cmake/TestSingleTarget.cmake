include(${CMAKE_SOURCE_DIR}/cmake/isa_tests/isa_test.cmake)

function(TestSingleTarget)
  set(TEST_TARGET ${TARGET})

  # This function will test the compilation and running of the
  # target specified in TEST_TARGET
  TestForISA()

  if(${CHK_TARGET_BUILD})
    message(STATUS "Testing target ${TEST_TARGET}: compilation [success]")

    if(NOT ${BLASFEO_CROSSCOMPILING} )
      if(${CHK_TARGET_RUN})
        message(STATUS "Testing target ${TEST_TARGET}: run [success]")
      else()
        message(STATUS "Testing target ${TEST_TARGET}: run [failed]")
      endif()
    endif()

  else()
    message(STATUS "Testing target ${TEST_TARGET}: compilation [failed]")
    message("Compile output:")
    message(${CHK_TARGET_OUTPUT})
    message(FATAL_ERROR "Unable to compile for target ${TEST_TARGET}")
  endif()

endfunction()
