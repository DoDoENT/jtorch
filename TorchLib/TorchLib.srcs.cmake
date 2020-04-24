set( SOURCES "" )

set( Source
    ${CMAKE_CURRENT_LIST_DIR}/Source/Linear.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Linear.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/ReLU.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/ReLU.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Reshape.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Reshape.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Sequential.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Sequential.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/SpatialConvolution.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/SpatialConvolution.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/SpatialConvolutionFactory.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/SpatialConvolutionGemm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/SpatialConvolutionGemm.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/SpatialDropout.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/SpatialDropout.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/SpatialMaxPooling.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/SpatialMaxPooling.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Tanh.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Tanh.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Tensor.hpp
    # ${CMAKE_CURRENT_LIST_DIR}/Source/TorchData.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/TorchData.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/TorchStage.cpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/TorchStage.hpp
)
source_group( "Source" FILES ${Source} )
list( APPEND SOURCES ${Source} )

set( Source_Utils
    ${CMAKE_CURRENT_LIST_DIR}/Source/Utils/InputStream.hpp
    ${CMAKE_CURRENT_LIST_DIR}/Source/Utils/VectorManaged.hpp
)
source_group( "Source\\Utils" FILES ${Source_Utils} )
list( APPEND SOURCES ${Source_Utils} )
