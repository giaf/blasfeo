
if(${BLASFEO_LA_MODE} MATCHES HIGH_PERFORMANCE)

    set(AUX_SRC ${CMAKE_CURRENT_SOURCE_DIR}/d_aux_lib4.c)

    if(${BLASFEO_TARGET} MATCHES X64_INTEL_HASWELL OR
       ${BLASFEO_TARGET} MATCHES X64_INTEL_SANDY_BRIDGE)

       list(APPEND AUX_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/s_aux_lib8.c
            ${CMAKE_CURRENT_SOURCE_DIR}/m_aux_lib48.c)

    elseif(${BLASFEO_TARGET} MATCHES X64_INTEL_CORE OR
           ${BLASFEO_TARGET} MATCHES X64_AMD_BULLDOZER OR
           ${BLASFEO_TARGET} MATCHES ARMV8A_ARM_CORTEX_A57 OR
           ${BLASFEO_TARGET} MATCHES ARMV7A_ARM_CORTEX_A15 OR
           ${BLASFEO_TARGET} MATCHES GENERIC)
	
        list(APPEND AUX_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/s_aux_lib4.c
            ${CMAKE_CURRENT_SOURCE_DIR}/m_aux_lib44.c)

    endif()

else() # Reference BLAS

    set(AUX_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/d_aux_lib.c
        ${CMAKE_CURRENT_SOURCE_DIR}/s_aux_lib.c
        ${CMAKE_CURRENT_SOURCE_DIR}/m_aux_lib.c)

endif()

if(BLASFEO_EXT_DEP)

	set(EXT_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/d_aux_ext_dep_lib.c
        ${CMAKE_CURRENT_SOURCE_DIR}/d_aux_ext_dep_lib4.c
        ${CMAKE_CURRENT_SOURCE_DIR}/s_aux_ext_dep_lib.c
        ${CMAKE_CURRENT_SOURCE_DIR}/s_aux_ext_dep_lib4.c
		${CMAKE_CURRENT_SOURCE_DIR}/v_aux_ext_dep_lib.c
        ${CMAKE_CURRENT_SOURCE_DIR}/i_aux_ext_dep_lib.c)

endif()

if(BLASFEO_TESTING)
    list(APPEND AUX_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/d_aux_libref.c
        ${CMAKE_CURRENT_SOURCE_DIR}/s_aux_libref.c)
endif()

target_sources(blasfeo PRIVATE ${AUX_SRC} ${EXT_SRC})
