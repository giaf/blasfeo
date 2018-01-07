
set(AVX2_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_4x4_lib4.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_8x4_lib4.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_8x8_lib4.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_12x4_lib4.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_10xX_lib4.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_8x2_lib4.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_6xX_lib4.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemm_4x2_lib4.S

    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgemv_8_lib4.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dsymv_6_lib4.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgetrf_pivot_4_lib4.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgebp_lib4.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgelqf_4_lib4.S
    # ${CMAKE_CURRENT_SOURCE_DIR}/kernel_dgetr_lib4.c

    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemm_24x4_lib8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemm_16x4_lib8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemm_8x8_lib8.S
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel_sgemm_8x4_lib8.S)

target_sources(blasfeo PRIVATE ${AVX2_SRC})
