#include <math.h>         // for fabsf, floor, log10, pow
#include <string.h>       // for NULL, memcpy
#include <stdexcept>      // for runtime_error

#include "Reshape.hpp"
#include "Tensor.hpp"     // for Tensor
#include "TorchData.hpp"  // for TorchData, TorchDataType


#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }


namespace mtorch {

  Reshape::Reshape(const uint32_t dim, const uint32_t* size) : TorchStage() {
    odim_ = dim;
    osize_ = new uint32_t[odim_];
    memcpy(osize_, size, sizeof(osize_[0]) * odim_);
  }

  Reshape::~Reshape() {
    SAFE_DELETE_ARR(osize_);
  }

  uint32_t Reshape::outNElem() const {
    if (odim_ == 0) {
      return 0;
    }
    uint32_t ret = 1;
    for (uint32_t i = 0; i < odim_; i++) {
      ret *= osize_[i];
    }
    return ret;
  }

  void Reshape::init(TorchData& input, TorchData **output) {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("Reshape::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;

    uint32_t nelems = outNElem();
    if (in.nelems() != nelems) {
      throw std::runtime_error("Reshape::init() - Bad input size!");
    }

    *output = in.view(odim_, osize_);  // rets header that uses same storage

  }

  void Reshape::forwardProp(TorchData& input, TorchData **output) {
    init(input, output);
    // Nothing to do.  init will initialize our tensor view that points to the
    // same storage as the input.
  }

  TorchStage* Reshape::loadFromStream( InputStream & stream ) noexcept
  {
    uint32_t dim = stream.read< uint32_t >();
    uint32_t* size = new uint32_t[dim];
    stream.readArray< uint32_t >( size, dim );

    TorchStage* stage = new Reshape(dim, size);
    SAFE_DELETE_ARR(size);
    return stage;
  }

}  // namespace mtorch
