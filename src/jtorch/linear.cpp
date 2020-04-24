#include <math.h>         // for fabsf, floor, log10, pow
#include <stddef.h>       // for NULL
#include <stdexcept>      // for runtime_error

#include "jtorch/linear.h"
#include "jtorch/tensor.h"
#include "jtorch/torch_data.h"


#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }


namespace jtorch {

  Linear::Linear(const uint32_t n_inputs, const uint32_t n_outputs)
    : TorchStage() {
    n_inputs_ = n_inputs;
    n_outputs_ = n_outputs;

    // NOTE: For efficiency we store the weight matrix transposed!
    // (we want the matrix vector multiply to be strided properly)
    uint32_t size_[2] = {n_outputs_, n_inputs_};
    weights_ = new Tensor<float>(2, size_);
    biases_ = new Tensor<float>(1, &n_outputs_);
  }

  Linear::~Linear() {
    SAFE_DELETE(weights_);
    SAFE_DELETE(biases_);
  }

  void Linear::setWeights(const float* weights) {
    weights_->setData(weights);
  }

  void Linear::setWeightsFromStream( InputStream & stream )
  {
      weights_->setDataFromStream( stream );
  }

  void Linear::setBiases(const float* biases) {
    biases_->setData(biases);
  }

  void Linear::setBiasesFromStream( InputStream & stream )
  {
      biases_->setDataFromStream( stream );
  }

  void Linear::init(TorchData& input, TorchData **output)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("Linear::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (in.dim() != 1 || in.size()[0] != n_inputs_) {
      throw std::runtime_error("Linear::init() - ERROR: input size mismatch!");
    }
    *output = new Tensor<float>(1, &n_outputs_);
  }

  void Linear::forwardProp(TorchData& input, TorchData** output) {
    init(input, output);
	float* A = weights_->getData();
    float* X = ((Tensor<float>&)input).getData();
    Tensor<float>* Y = TO_TENSOR_PTR(*output);
	uint32_t M = (uint32_t)n_outputs_;
	uint32_t N = (uint32_t)n_inputs_;
	// Perform the linear accumulation
	for (uint32_t i = 0; i < M; i++) {
		float sum = 0;
		for (uint32_t k = 0; k < N; k++) {
			sum += A[i + M * k] * X[k];
		}
		Y->setDataAt(sum, i);
	}
    // Now add in the bias
	Tensor<float>::accumulate(*Y, *biases_);
  }

  TorchStage * Linear::loadFromStream( InputStream & stream ) noexcept
  {
    int32_t n_outputs = stream.read< int32_t >();
    int32_t n_inputs  = stream.read< int32_t >();
    Linear* ret = new Linear(n_inputs, n_outputs);

    ret->setWeightsFromStream( stream );
    ret->setBiasesFromStream( stream );

    return ret;
  }

}  // namespace jtorch
