if( NOT TARGET TorchLib )

    include( ${CMAKE_CURRENT_LIST_DIR}/../BlasLibrary/BlasLibrary.cmake )

    # Obtain all source files
    include( ${CMAKE_CURRENT_LIST_DIR}/TorchLib.srcs.cmake )

    add_library( TorchLib STATIC ${SOURCES} )
    target_include_directories( TorchLib PUBLIC ${CMAKE_CURRENT_LIST_DIR}/Source )

    target_link_libraries( TorchLib PUBLIC BlasLibrary )
endif()
