cmake_minimum_required(VERSION 3.10)

project( torch )

add_compile_options(
    -fstrict-aliasing
    $<$<COMPILE_LANGUAGE:CXX>:-fstrict-enums>
    -fvisibility=hidden
    $<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>
    -fPIC
    -flto
    -fwhole-program-vtables
    $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++2a>
    $<$<NOT:$<COMPILE_LANGUAGE:CXX>>:-std=gnu11>
    -fno-exceptions
    -fno-threadsafe-statics
    -fno-rtti
    -fomit-frame-pointer
    -ffunction-sections
    -fmerge-all-constants
    -fno-stack-protector
    -Wall
    -Wextra
    -Wstrict-aliasing
)

if ( EMSCRIPTEN )
    add_compile_options( "SHELL:-s WASM=1" "SHELL:-s STRICT=1" )

    set( CMAKE_EXECUTABLE_SUFFIX ".html" )
endif()

if ( EMSCRIPTEN )
    add_link_options(
        "SHELL:--llvm-lto 3"
        "SHELL:--llvm-opts 3"
        "SHELL:-s ALLOW_MEMORY_GROWTH=1"
        --no-heap-copy
        "SHELL:-s INITIAL_MEMORY=209715200"
    )
endif()

include( ${CMAKE_CURRENT_LIST_DIR}/TorchLib/TorchLib.cmake )

add_library( TorchTestLib STATIC libtest.cpp libtest.hpp )
target_link_libraries( TorchTestLib PRIVATE TorchLib )

set_source_files_properties( libtest.cpp PROPERTIES COMPILE_OPTIONS "-fexceptions" )

add_executable( TorchTest TorchTest.cpp )
target_link_libraries( TorchTest PRIVATE TorchTestLib )


set( TEST_FILES
    data_in.bin
    spatial_convolution_map.bin
    spatial_max_pooling.bin
    test_model_result.bin
    lena_image.bin
    spatial_convolution_mm_padding.bin
    spatial_subtractive_normalization.bin
    testmodel.bin
    linear.bin
    spatial_divisive_normalization.bin
    spatial_subtractive_normalization_2d.bin
    threshold.bin
    spatial_contrastive_normalization.bin
    spatial_divisive_normalization_2d.bin
    spatial_up_sampling_nearest.bin
    spatial_convolution.bin
    spatial_lp_pooling.bin
    tanh_result.bin
)

if ( EMSCRIPTEN )
    file( COPY ${TEST_FILES} DESTINATION ${CMAKE_BINARY_DIR}/res )
    target_link_options( TorchTest PRIVATE "SHELL:--preload-file ${CMAKE_BINARY_DIR}/res@/" "SHELL:-s STRICT=0" )
else()
    file( COPY ${TEST_FILES} DESTINATION ${CMAKE_BINARY_DIR} )
endif()
