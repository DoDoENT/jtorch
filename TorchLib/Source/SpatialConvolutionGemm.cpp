#include <math.h>         // for fabsf, floor, log10, pow
#include <stddef.h>       // for NULL
#include <stdexcept>      // for runtime_error

#include "Blas.hpp"       // for gemm, im2col
#include "SpatialConvolutionGemm.hpp"
#include "Tensor.hpp"     // for Tensor, TO_TENSOR_PTR
#include "TorchData.hpp"  // for TorchData, TorchDataType

namespace mtorch {
class TorchStage;
}  // namespace mtorch

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }


namespace mtorch {

SpatialConvolutionGemm::SpatialConvolutionGemm(const uint32_t feats_in,
    const uint32_t feats_out, const uint32_t filt_height,
    const uint32_t filt_width, const uint32_t padw, const uint32_t padh) {

    filt_width_ = filt_width;
    filt_height_ = filt_height;
    feats_in_ = feats_in;
    feats_out_ = feats_out;
    padw_ = padw;
    padh_ = padh;

    uint32_t dim = 4;
    uint32_t size[4] = {filt_width_, filt_height_, feats_in_, feats_out_};

    weights_ = new Tensor<float>(dim, size);
    biases_ = new Tensor<float>(1, &feats_out_);
}

SpatialConvolutionGemm::~SpatialConvolutionGemm() {
    SAFE_DELETE(weights_);
    SAFE_DELETE(biases_);
}

void SpatialConvolutionGemm::setWeights(const float* weights) {
    weights_->setData(weights);
}

void SpatialConvolutionGemm::setBiases(const float* biases) {
    biases_->setData(biases);
}

void SpatialConvolutionGemm::setWeightsFromStream( InputStream & stream )
{
    weights_->setDataFromStream( stream );
}

void SpatialConvolutionGemm::setBiasesFromStream( InputStream & stream )
{
    biases_->setDataFromStream( stream );
}

void SpatialConvolutionGemm::init(TorchData& input, TorchData **output, Tensor<float>** ones, Tensor<float>** columns)  {
    if (input.type() != TorchDataType::TENSOR_DATA) {
      throw std::runtime_error("SpatialConvolution::init() - "
        "FloatTensor expected!");
    }
    Tensor<float>& in = (Tensor<float>&)input;
    if (in.dim() != 3) {
      throw std::runtime_error("SpatialConvolution::init() - Input not 3D!");
    }
    if (in.size()[2] != feats_in_) {
      throw std::runtime_error("SpatialConvolution::init() - ERROR: "
        "incorrect number of input features!");
    }

    const uint32_t inputWidth = in.size()[0];
    const uint32_t inputHeight = in.size()[1];
    const uint32_t outputWidth = inputWidth - filt_width_ + 1 + 2 * padw_;
    const uint32_t outputHeight = inputHeight - filt_height_ + 1 + 2 * padh_;

    // Resize output
    uint32_t out_dim[3];
    out_dim[0] = outputWidth;
    out_dim[1] = outputHeight;
    out_dim[2] = feats_out_;
    *output = new Tensor<float>(3, out_dim);

    // Resize temporary columns
    uint32_t columns_dim[2];
    columns_dim[0] = outputHeight * outputWidth;
    columns_dim[1] = feats_in_ * filt_width_ * filt_height_;
    *columns = new Tensor<float>(2, columns_dim);

    // Define a buffer of ones, for bias accumulation
    uint32_t ones_dim[2];
    ones_dim[0] = outputWidth;
    ones_dim[1] = outputHeight;
    *ones = new Tensor<float>(2, ones_dim);

}

void SpatialConvolutionGemm::forwardProp(TorchData& input, TorchData **output) {

    Tensor<float>* ones = NULL;
    Tensor<float>* columns = NULL;

    init(input, output, &ones, &columns);

    Tensor<float>& in = (Tensor<float>&)input;
    Tensor<float>* out = (Tensor<float>*)(*output);

    int inputWidth = (int) in.size()[0];
    int inputHeight = (int) in.size()[1];
    const uint32_t* out_size = TO_TENSOR_PTR(*output)->size();
    int outputWidth = (int) out_size[0];
    int outputHeight = (int) out_size[1];
    int nInputPlane = (int) feats_in_;
    int nOutputPlane = (int) feats_out_;
    int kH = (int) filt_height_;
    int kW = (int) filt_width_;
    int padw = (int) padw_;
    int padh = (int) padh_;
    int dH = 1;
    int dW = 1;

    // Do Bias first:
    int m = nOutputPlane;
    int n = outputHeight * outputWidth;
    int k = 1;
    Tensor<float>::fill(*ones, 1);

    Blas::gemm('t', 'n', n, m, k, 1, ones->getData(), k,
                biases_->getData(), k, 0, out->getData(), n);

    // Extract columns:
    Blas::im2col((&in)->getData(), nInputPlane, inputHeight, inputWidth, kH, kW, padh,
                padw, dH, dW, columns->getData());

    m = nOutputPlane;
    n = outputHeight * outputWidth;
    k = nInputPlane * kH * kW;

    Blas::gemm('n', 'n', n, m, k, 1, columns->getData(), n,
                weights_->getData(), k, 1, out->getData(), n);

    SAFE_DELETE(ones);
    SAFE_DELETE(columns);
}

TorchStage* SpatialConvolutionGemm::loadFromStream(InputStream & stream) noexcept
{
      int32_t filt_width, filt_height, n_input_features, n_output_features,
        padw, padh;

      filt_width = stream.read< int32_t >();
      filt_height = stream.read< int32_t >();
      n_input_features = stream.read< int32_t >();
      n_output_features = stream.read< int32_t >();
      padw = stream.read< int32_t >();
      padh = stream.read< int32_t >();


    SpatialConvolutionGemm* ret = new SpatialConvolutionGemm(n_input_features,
      n_output_features, filt_height, filt_width, padw, padh);

    ret->setWeightsFromStream( stream );
    ret->setBiasesFromStream( stream );

    return ret;
}

}  // namespace mtorch
