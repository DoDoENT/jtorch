#include "TorchStage.hpp"

#include "FileUtils.hpp"
#include "Linear.hpp"
#include "ReLU.hpp"
#include "Reshape.hpp"
#include "Sequential.hpp"
#include "SpatialConvolutionFactory.hpp"
#include "SpatialDropout.hpp"
#include "SpatialMaxPooling.hpp"
#include "Tanh.hpp"

#include <cstddef>

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

namespace mtorch {

  TorchStage::TorchStage() = default;
  TorchStage::~TorchStage() = default;

  TorchStage* TorchStage::loadFromFile( std::string_view const file ) noexcept
  {
    auto buf = FileUtils::fileReadToBuffer( file.data() );

    if ( !buf.empty() )
    {
      // Now recursively load the network
      return TorchStage::loadFromBuffer( buf );
    }
    else
    {
      return nullptr;
    }
  }

  TorchStage* TorchStage::loadFromBuffer( std::vector< std::uint8_t > const & buffer ) noexcept
  {
    InputStream istream{ buffer };
    // Now recursively load the network
    return TorchStage::loadFromStream( istream );
  }

  TorchStage* TorchStage::loadFromStream( InputStream & stream ) noexcept
  {
    // Read in the enum type:
    int type = stream.read< int >();
    // Now load in the module
    TorchStage* node = NULL;
    switch (type) {
    case SEQUENTIAL_STAGE:
      node = Sequential::loadFromStream( stream );
      break;
    case TANH_STAGE:
      node = Tanh::loadFromStream( stream );
      break;
    case THRESHOLD_STAGE:
      node = Threshold::loadFromStream( stream );
      break;
    case LINEAR_STAGE:
      node = Linear::loadFromStream( stream );
      break;
    case RESHAPE_STAGE:
      node = Reshape::loadFromStream( stream );
      break;
    case SPATIAL_CONVOLUTION_STAGE:
        node = SpatialConvolutionFactory::loadFromStream( stream );
      break;
    case SPATIAL_MAX_POOLING_STAGE:
      node = SpatialMaxPooling::loadFromStream( stream );
      break;
    case SPATIAL_CONVOLUTION_MM_STAGE:
        node = SpatialConvolutionFactory::loadFromStream( stream );
      break;
    case SPATIAL_DROPOUT:
      node = SpatialDropout::loadFromStream( stream );
      break;
    default:
      std::terminate();
    }

    return node;
  }



}  // namespace mtorch
