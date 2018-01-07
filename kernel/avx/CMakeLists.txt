
set(AVX_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_diag_lib4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemv_4_lib4.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgeqrf_4_lib4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgecp_lib4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgetr_lib4.c

    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemm_diag_lib8.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgetr_lib8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgead_lib8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgecpsc_lib8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemv_8_lib8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemv_4_lib8.S)

if(${BLASFEO_TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)
    list(APPEND AVX_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_8x4_lib4.S
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_8x2_lib4.S
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_4x4_lib4.S
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_4x2_lib4.S
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_12x4_lib4.S
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_10xX_lib4.S
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_6xX_lib4.S
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemv_12_lib4.S
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemv_8_lib4.S
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dsymv_6_lib4.S
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgetrf_pivot_4_lib4.c
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgebp_lib4.S
        
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemm_16x4_lib8.S
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemm_8x8_lib8.S
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemm_8x4_lib8.S)
endif()

target_sources(blasfeo PRIVATE ${AVX_SRC})
