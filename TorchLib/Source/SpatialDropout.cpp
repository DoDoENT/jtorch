#include <math.h>         // for fabsf, floor, log10, pow
#include <stddef.h>       // for NULL
#include <stdexcept>      // for runtime_error

#include "SpatialDropout.hpp"
#include "Tensor.hpp"     // for Tensor, TO_TENSOR_PTR
#include "TorchData.hpp"  // for TorchData, TorchDataType


#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }


namespace mtorch {

  SpatialDropout::SpatialDropout(const float p) : TorchStage() {
    p_ = p;
  }

  SpatialDropout::~SpatialDropout() {
  }

  void SpatialDropout::init(TorchData& input, TorchData **output)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("SpatialDropout::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;

    *output = Tensor<float>::clone(in);

  }

  void SpatialDropout::forwardProp(TorchData& input, TorchData **output) {
    init(input, output);

    Tensor<float>::copy(*TO_TENSOR_PTR(*output), (Tensor<float>&)input);
    Tensor<float>::mul(*TO_TENSOR_PTR(*output), 1 - p_);
  }

  TorchStage* SpatialDropout::loadFromStream( InputStream & stream ) noexcept
  {
    float p = stream.read< float >();
    return new SpatialDropout(p);
  }

}  // namespace mtorch
