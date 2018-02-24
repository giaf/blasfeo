
function(determine_instruction_set_flags BLASFEO_TARGET)

set(ISA_FLAGS "")

if(NOT (CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang"))
    set(BLASFEO_TARGET "GENERIC" CACHE STRING "Target architecture" FORCE)
endif()

if(${BLASFEO_TARGET} MATCHES X64_INTEL_HASWELL)
    set(ISA_FLAGS "-m64 -mavx2 -mfma")
elseif(${BLASFEO_TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)
    set(ISA_FLAGS "-m64 -mavx")
elseif(${BLASFEO_TARGET} MATCHES X64_INTEL_CORE)
    set(ISA_FLAGS "-m64 -msse3")
elseif(${BLASFEO_TARGET} MATCHES X64_AMD_BULLDOZER)
    set(ISA_FLAGS "-m64 -mavx -mfma")
elseif(${BLASFEO_TARGET} MATCHES ARMV8A_ARM_CORTEX_A57)
    set(ISA_FLAGS "-march=armv8-a+crc+crypto+fp+simd")
elseif(${BLASFEO_TARGET} MATCHES ARMV7A_ARM_CORTEX_A15)
    set(ISA_FLAGS "-marm -mfloat-abi=hard -mfpu=neon-vfpv4 -mcpu=cortex-a15")
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -mfpu=neon-vfpv4" PARENT_SCOPE)
elseif(${BLASFEO_TARGET} MATCHES GENERIC)
    # Skip
elseif(${BLASFEO_TARGET} MATCHES AUTOMATIC)

    unset(BLASFEO_TARGET CACHE)

    include(CheckCSourceRuns)

    set(TEST_AVX2
        "#include <immintrin.h> \n \
        int main() { \
            __m256i a, b, dst; \
            dst = _mm256_add_epi64(a, b); \
            return 0; \
        }")

    set(CMAKE_REQUIRED_FLAGS "-mavx2")

    check_c_source_runs("${TEST_AVX2}" HAS_AVX2)


    set(TEST_AVX
        "#include <immintrin.h> \n \
        int main() { \
            __m256d a, b, dst; \
            dst = _mm256_add_pd(a, b); \
            return 0; \
        }")

    set(CMAKE_REQUIRED_FLAGS "-mavx")

    check_c_source_runs("${TEST_AVX}" HAS_AVX)


    set(TEST_FMA
        "#include <immintrin.h> \n \
        int main() { \
            __m128d a, b, c, dst; \
            dst = _mm_fmadd_pd(a, b, c); \
            return 0; \
        }")

    set(CMAKE_REQUIRED_FLAGS "-mfma")

    check_c_source_runs("${TEST_FMA}" HAS_FMA)


    set(TEST_SSE3
        "#include <pmmintrin.h> \n \
        int main() { \
            __m128d a, b, dst; \
            dst = _mm_addsub_pd(a, b); \
            return 0; \
        }")

    set(CMAKE_REQUIRED_FLAGS "-msse3")

    check_c_source_runs("${TEST_SSE3}" HAS_SSE3)

    if(HAS_AVX2)
        set(ISA_FLAGS "${ISA_FLAGS} -mavx2")
        set(BLASFEO_TARGET "X64_INTEL_HASWELL" CACHE STRING "Target architecture")
    endif()

    if(HAS_FMA)
        set(ISA_FLAGS "${ISA_FLAGS} -mfma")
        set(BLASFEO_TARGET "X64_AMD_BULLDOZER" CACHE STRING "Target architecture")
    endif()

    if(HAS_AVX)
        set(ISA_FLAGS "${ISA_FLAGS} -mavx")
        set(BLASFEO_TARGET "X64_INTEL_SANDY_BRIDGE" CACHE STRING "Target architecture")
    endif()

    if(HAS_SSE3)
        set(ISA_FLAGS "${ISA_FLAGS} -msse3")
        set(BLASFEO_TARGET "X64_INTEL_CORE" CACHE STRING "Target architecture")
    endif()

else()
    message(FATAL_ERROR "Target architecture ${BLASFEO_TARGET} unknown")
endif()

# architecture-specific flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ISA_FLAGS} -DTARGET_${BLASFEO_TARGET}" PARENT_SCOPE)

message(STATUS "Target architecture is ${BLASFEO_TARGET}")


endfunction()
