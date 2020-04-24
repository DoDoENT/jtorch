set( SOURCES "" )

set( Source
    ${CMAKE_CURRENT_LIST_DIR}/Source/Blas.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Blas.hpp
)
source_group( "Source" FILES ${Source} )
list( APPEND SOURCES ${Source} )

set( Source_Eigen
    ${CMAKE_CURRENT_LIST_DIR}/Source/Eigen/BlasEigen.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Eigen/BlasEigen.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Eigen/BlasHeader.hpp
)


source_group( "Source\\Eigen" FILES ${Source_Eigen} )
list( APPEND SOURCES ${Source_Eigen} )

set( Source_Naive
    ${CMAKE_CURRENT_LIST_DIR}/Source/Naive/BlasNaive.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Naive/BlasNaive.hpp
)
source_group( "Source\\Naive" FILES ${Source_Naive} )
list( APPEND SOURCES ${Source_Naive} )
