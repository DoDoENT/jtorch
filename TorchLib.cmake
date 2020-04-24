if( NOT TARGET TorchLib )

    # include( ${CMAKE_CURRENT_LIST_DIR}/BlasLibrary.cmake )

    # Obtain all source files
    include( ${CMAKE_CURRENT_LIST_DIR}/TorchLib.srcs.cmake )

    add_library( TorchLib STATIC ${SOURCES} )
    target_include_directories( TorchLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include )

    # target_link_libraries( TorchLib PUBLIC BlasLibrary )

    # if ( NOT MB_ALLOW_EXCEPTIONS )
        # always allow C++ exceptions for TorchLib
        # target_compile_options( TorchLib PRIVATE ${TNUN_compiler_exceptions_on} )
    # endif()

endif()
