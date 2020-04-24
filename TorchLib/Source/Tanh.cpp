#include <math.h>         // for fabsf, floor, log10, pow, tanhf
#include <stddef.h>       // for NULL
#include <cstdint>        // for uint32_t
#include <stdexcept>      // for runtime_error

#include "Tanh.hpp"
#include "Tensor.hpp"     // for Tensor, TO_TENSOR_PTR
#include "TorchData.hpp"  // for TorchData, TorchDataType


#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

namespace mtorch {

  Tanh::Tanh() : TorchStage() {
  }

  Tanh::~Tanh() {
  }

  void Tanh::init(TorchData& input, TorchData **output)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("Tanh::init() - FloatTensor expected!");
    }

    Tensor<float>& in = (Tensor<float>&)input;
    TorchData* temp = new Tensor<float>(in.dim(), in.size());

    *output = temp;
  }

  void Tanh::forwardProp(TorchData& input, TorchData **output) {
    init(input, output);
    float* data = ((Tensor<float>&)input).getData();
    uint32_t nelem = TO_TENSOR_PTR(*output)->nelems();
    
	for (uint32_t i = 0; i < nelem; i++)
	{
        TO_TENSOR_PTR(*output)->setDataAt(tanhf(data[i]), i);
	}   
  }

  TorchStage* Tanh::loadFromStream( InputStream & ) noexcept
  {
    // Nothing to do for Tanh
    return new Tanh();
  }

}  // namespace mtorch
