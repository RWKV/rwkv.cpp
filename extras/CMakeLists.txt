function(rwkv_add_extra source)
    get_filename_component(EXTRA_TARGET ${source} NAME_WE)
    add_executable(rwkv_${EXTRA_TARGET} ${source})

    target_link_libraries(rwkv_${EXTRA_TARGET} PRIVATE ggml rwkv)
    
     if (RWKV_STATIC)
        if(RWKV_HIPBLAS)
            message(FATAL_ERROR "Static linking not supported for HIP/ROCm")
        else()
            get_target_property(target_LINK_OPTIONS rwkv_${EXTRA_TARGET} LINK_OPTIONS)
            list(REMOVE_ITEM target_LINK_OPTIONS "-static")
            set_target_properties(rwkv_${EXTRA_TARGET} PROPERTIES LINK_OPTIONS "${target_LINK_OPTIONS}")
        endif()
    endif()
endfunction()

rwkv_add_extra(cpu_info.c)
rwkv_add_extra(quantize.c)
