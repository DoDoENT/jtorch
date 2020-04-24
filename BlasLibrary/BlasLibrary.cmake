if( NOT TARGET BlasLibrary )

    add_library( Eigen INTERFACE )
    target_include_directories( Eigen INTERFACE ${CMAKE_SOURCE_DIR}/eigen ${CMAKE_SOURCE_DIR}/unsupported )
    target_compile_definitions( Eigen INTERFACE
        EIGEN_MAX_STATIC_ALIGN_BYTES=16
        EIGEN_DEFAULT_TO_ROW_MAJOR
        EIGEN_DEFAULT_DENSE_INDEX_TYPE=std::int32_t
        EIGEN_NO_AUTOMATIC_RESIZING
        EIGEN_MPL2_ONLY
        EIGEN_FAST_MATH
        EIGEN_CMAKE_INCLUDED
        EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS=16
    )

    # Obtain all source files
    include( ${CMAKE_CURRENT_LIST_DIR}/BlasLibrary.srcs.cmake )

    add_library( BlasLibrary STATIC ${SOURCES} )

    target_link_libraries( BlasLibrary PUBLIC Eigen )
    target_include_directories( BlasLibrary PUBLIC ${CMAKE_CURRENT_LIST_DIR}/Source )
endif()
