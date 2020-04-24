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
    -fno-threadsafe-statics
    -fno-rtti
    -fomit-frame-pointer
    -ffunction-sections
    -fmerge-all-constants
    -fno-stack-protector
    -Wall
    -Wextra
    -Wstrict-aliasing
    "SHELL:-s WASM=1"
)

add_link_options(
    "SHELL:--llvm-lto 3"
    "SHELL:--llvm-opts 3"
    "SHELL:-s STRICT=1"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    --no-heap-copy
    "SHELL:-s INITIAL_MEMORY=209715200"
)

include( ${CMAKE_CURRENT_LIST_DIR}/TorchLib.cmake )

add_executable( TorchTest TorchTest.cpp )
target_link_libraries( TorchTest PRIVATE TorchLib )
