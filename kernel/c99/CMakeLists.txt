
set(C99_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemv_4_lib4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemm_4x4_lib4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemm_diag_lib4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemv_4_lib4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_ssymv_4_lib4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgetrf_pivot_4_lib4.c)

if(${BLASFEO_TARGET} MATCHES X64_INTEL_CORE OR
   ${BLASFEO_TARGET} MATCHES X64_AMD_BULLDOZER OR
   ${BLASFEO_TARGET} MATCHES ARMV8A_ARM_CORTEX_A57 OR
   ${BLASFEO_TARGET} MATCHES ARMV7A_ARM_CORTEX_A15 OR
   ${BLASFEO_TARGET} MATCHES GENERIC)

    list(APPEND C99_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_4x4_lib4.c
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_diag_lib4.c
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dsymv_4_lib4.c
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgetrf_pivot_4_lib4.c
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgeqrf_4_lib4.c
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgecp_lib4.c
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgetr_lib4.c
        
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgecp_lib4.c
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgetr_lib4.c)

endif()

target_sources(blasfeo PRIVATE ${C99_SRC})
