include(${CMAKE_SOURCE_DIR}/cmake/isa_tests/isa_test.cmake)

function(TestSingleTarget)
  set(TEST_TARGET ${TARGET})

  # This function will test the compilation and running of the
  # target specified in TEST_TARGET
  TestForISA()

  if(${CHK_TARGET_BUILD})
    message(STATUS "Testing target ${TEST_TARGET}: compilation [success]")

    if(${CHK_TARGET_RUN})
      message(STATUS "Testing target ${TEST_TARGET}: run [success]")

    else()
      message(STATUS "Testing target ${TEST_TARGET}: run [failed]")
    endif()

  else()
    message(FATAL_ERROR "Testing target ${TEST_TARGET}: compilation [failed]")
  endif()

endfunction()
