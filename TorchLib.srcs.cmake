set( SOURCES "" )

set( SRC_ROOT ${CMAKE_CURRENT_LIST_DIR}/src/jtorch )

set( Source
    ${SRC_ROOT}/linear.cpp
    ${SRC_ROOT}/threshold.cpp
    ${SRC_ROOT}/reshape.cpp
    ${SRC_ROOT}/sequential.cpp
    ${SRC_ROOT}/spatial_convolution.cpp
    ${SRC_ROOT}/spatial_dropout.cpp
    ${SRC_ROOT}/spatial_max_pooling.cpp
    ${SRC_ROOT}/tanh.cpp
    ${SRC_ROOT}/torch_data.cpp
    ${SRC_ROOT}/torch_stage.cpp
)
source_group( "Source" FILES ${Source} )
list( APPEND SOURCES ${Source} )

