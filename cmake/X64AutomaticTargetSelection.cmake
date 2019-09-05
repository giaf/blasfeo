include(${CMAKE_SOURCE_DIR}/cmake/isa_tests/isa_test.cmake)

function(X64AutomaticTargetSelection)

  # Iterate over each target to test the compilation and running
  foreach(TEST_TARGET ${X64_AUTOMATIC_TARGETS})
    # This function will test the compilation and running of the
    # target specified in TEST_TARGET
    TestForISA()

    if(${CHK_TARGET_BUILD})
      message(STATUS "Testing target ${TEST_TARGET}: compilation [success]")

      if(${CHK_TARGET_RUN})
        message(STATUS "Testing target ${TEST_TARGET}: run [success]")

        # It both compiles and runs, so pass it up to the parent to use
        set(TARGET ${TEST_TARGET} PARENT_SCOPE)
        return()

      else()
        message(STATUS "Testing target ${TEST_TARGET}: run [failed]")
      endif()

    else()
      message(STATUS "Testing target ${TEST_TARGET}: compilation [failed]")
    endif()

  endforeach()

  message(FATAL_ERROR "Unable to identify a target to use. Please select one manually.")

endfunction()
