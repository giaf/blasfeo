
if(${BLASFEO_LA_MODE} MATCHES HIGH_PERFORMANCE)

    add_subdirectory(c99)

    if(${BLASFEO_TARGET} MATCHES X64_INTEL_HASWELL)
    
        add_subdirectory(avx2)
        add_subdirectory(avx)

    elseif(${BLASFEO_TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)
    
        add_subdirectory(avx)

    elseif(${BLASFEO_TARGET} MATCHES X64_INTEL_CORE)

        target_sources(blasfeo PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/sse3/kernel_dgemm_4x4_lib4.S
            ${CMAKE_CURRENT_SOURCE_DIR}/sse3/kernel_dgemv_4_lib4.S)

    elseif(${BLASFEO_TARGET} MATCHES X64_AMD_BULLDOZER)

        target_sources(blasfeo PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/fma/kernel_dgemm_4x4_lib4.S)

    elseif(${BLASFEO_TARGET} MATCHES ARMV8A_ARM_CORTEX_A57)

        target_sources(blasfeo PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/armv8a/kernel_dgemm_8x4_lib4.S
            ${CMAKE_CURRENT_SOURCE_DIR}/armv8a/kernel_dgemm_4x4_lib4.S
            ${CMAKE_CURRENT_SOURCE_DIR}/armv8a/kernel_sgemm_16x4_lib4.S
            ${CMAKE_CURRENT_SOURCE_DIR}/armv8a/kernel_sgemm_12x4_lib4.S
            ${CMAKE_CURRENT_SOURCE_DIR}/armv8a/kernel_sgemm_8x8_lib4.S
            ${CMAKE_CURRENT_SOURCE_DIR}/armv8a/kernel_sgemm_8x4_lib4.S
            ${CMAKE_CURRENT_SOURCE_DIR}/armv8a/kernel_sgemm_4x4_lib4.S)

    elseif(${BLASFEO_TARGET} MATCHES ARMV7A_ARM_CORTEX_A15)

        target_sources(blasfeo PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/armv7a/kernel_dgemm_4x4_lib4.S
            ${CMAKE_CURRENT_SOURCE_DIR}/armv7a/kernel_sgemm_12x4_lib4.S
            ${CMAKE_CURRENT_SOURCE_DIR}/armv7a/kernel_sgemm_8x4_lib4.S
            ${CMAKE_CURRENT_SOURCE_DIR}/armv7a/kernel_sgemm_4x4_lib4.S)

    endif()

endif()
