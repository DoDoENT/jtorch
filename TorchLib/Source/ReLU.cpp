#include <math.h>         // for fabsf, floor, log10, pow
#include <stddef.h>       // for NULL
#include <cstdint>        // for uint32_t
#include <stdexcept>      // for runtime_error

#include "ReLU.hpp"
#include "Tensor.hpp"     // for Tensor, TO_TENSOR_PTR
#include "TorchData.hpp"  // for TorchData, TorchDataType


#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }


namespace mtorch {

  Threshold::Threshold() : TorchStage() {
    threshold = 1e-6f;
    val = 0;
  }

  Threshold::~Threshold() {
  }

  void Threshold::init(TorchData& input, TorchData **output)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("Threshold::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;

    *output = new Tensor<float>(in.dim(), in.size());

  }

  void Threshold::forwardProp(TorchData& input, TorchData **output) {

    init(input, output);
	float* data = ((Tensor<float>&)input).getData();
    uint32_t nelem = TO_TENSOR_PTR(*output)->nelems();

	for (uint32_t i = 0; i < nelem; i++)
	{
        TO_TENSOR_PTR(*output)->setDataAt(data[i] > threshold ? data[i] : val, i);
	}
  }

  TorchStage* Threshold::loadFromStream( InputStream & stream ) noexcept
  {
    Threshold* ret = new Threshold();

    ret->threshold = stream.read< decltype( ret->threshold ) >();
    ret->val = stream.read< decltype( ret->val ) >();

    // WTF?!? This was here before - hardcoded values after reading from stream?!?
    ret->threshold = 1e-6f;
    ret->val = 0;

    return ret;
  }

}  // namespace mtorch
