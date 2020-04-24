#include <math.h>         // for INFINITY, fabsf, floor, log10, pow
#include <stddef.h>       // for NULL
#include <algorithm>      // for max
#include <stdexcept>      // for runtime_error

#include "SpatialMaxPooling.hpp"
#include "Tensor.hpp"     // for Tensor, TO_TENSOR_PTR
#include "TorchData.hpp"  // for TorchData, TorchDataType


#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

namespace mtorch {

  SpatialMaxPooling::SpatialMaxPooling(const uint32_t kw, const uint32_t kh, const uint32_t dw,
                                        const uint32_t dh, const uint32_t padw, const uint32_t padh) : TorchStage() {
    kw_ = kw;
    kh_ = kh;
    dw_ = dw;
    dh_ = dh;
    padw_ = padw;
    padh_ = padh;
  }

  SpatialMaxPooling::~SpatialMaxPooling() {
  }

  void SpatialMaxPooling::init(TorchData& input, TorchData **output)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("SpatialMaxPooling::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (in.dim() != 2 && in.dim() != 3) {
      throw std::runtime_error("Input dimension must be 2D or 3D!");
    }

    if (in.size()[0] % kw_ != 0 ||
        in.size()[1] % kh_ != 0) {
      throw std::runtime_error("width or height is not a multiple of "
        "the poolsize!");
    }
    uint32_t* out_size = new uint32_t[in.dim()];
    out_size[0] = in.size()[0] / kw_;
    out_size[1] = in.size()[1] / kh_;
    for (uint32_t i = 2; i < in.dim(); i++) {
      out_size[i] = in.size()[i];
    }
    *output = new Tensor<float>(in.dim(), out_size);
    SAFE_DELETE_ARR(out_size);

  }

  void SpatialMaxPooling::forwardProp(TorchData& input, TorchData **output) {
    init(input, output);
	float* input_data = ((Tensor<float>&)input).getData();
	uint32_t input_height = ((Tensor<float>&)input).size()[1];
	uint32_t input_width = ((Tensor<float>&)input).size()[0];
    const uint32_t* out_size = TO_TENSOR_PTR(*output)->size();
	uint32_t width = out_size[0];
	uint32_t height = out_size[1];
	bool two_dim = ((Tensor<float>&)input).dim() == 2;
    if (two_dim) {
		for (uint32_t x_out = 0; x_out < width; x_out++){
			for (uint32_t y_out = 0; y_out < height; y_out++){
				// Initilize the output to the bias
				float out_val = -INFINITY;
                uint32_t vstart = y_out * kh_;
                uint32_t vend = (y_out + 1) * kh_ - 1;
				// Get a pointer to the current input feature (that corresponds to this
				// output feature;
				float* input_f = input_data;
				for (uint32_t v = vstart; v <= vend; v++) {
                    uint32_t istart = v * input_width + x_out * kw_;
                    uint32_t iend = v * input_width + (x_out + 1) * kw_ - 1;
					for (uint32_t i = istart; i <= iend; i++) {
						out_val = std::max(out_val, input_f[i]);
					}
				}
				uint32_t index = x_out + width * y_out;
                TO_TENSOR_PTR(*output)->setDataAt(out_val, index);
			}
		}
    } else {
		uint32_t feats = out_size[2];
		for (uint32_t x_out = 0; x_out < width; x_out++){
			for (uint32_t y_out = 0; y_out < height; y_out++){
				for (uint32_t f_out = 0; f_out < feats; f_out++)
				{
					// Initilize the output to the bias
					float out_val = -INFINITY;
                    uint32_t vstart = y_out * kh_;
                    uint32_t vend = (y_out + 1) * kh_ - 1;
					// Get a pointer to the current input feature (that corresponds to this
					// output feature;
					float* input_f = &input_data[f_out * input_width * input_height];
					for (uint32_t v = vstart; v <= vend; v++) {
                        uint32_t istart = v * input_width + x_out * kw_;
                        uint32_t iend = v * input_width + (x_out + 1) * kw_ - 1;

						for (uint32_t i = istart; i <= iend; i++) {
							out_val = std::max(out_val, input_f[i]);
						}
					}
					uint32_t index = x_out + width * (y_out + height * f_out);
                    TO_TENSOR_PTR(*output)->setDataAt(out_val, index);
				}
			}
		}

    }

  }

  TorchStage* SpatialMaxPooling::loadFromStream( InputStream & stream ) noexcept
  {
    int kw, kh, dw, dh, padw, padh;
    kw = stream.read< int >();
    kh = stream.read< int >();
    dw = stream.read< int >();
    dh = stream.read< int >();
    padw = stream.read< int >();
    padh = stream.read< int >();
    return new SpatialMaxPooling(kw, kh, dw, dh, padw, padh);
  }

}  // namespace mtorch
