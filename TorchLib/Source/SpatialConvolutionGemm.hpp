//
//  SpatialConvolutionGemm.hpp
//
//  Created by Jonathan Tompson on 5/15/13.
//

#pragma once
#include <cstdint>                 // for uint32_t
#include <string>                  // for istream

#include "SpatialConvolution.hpp"  // for SpatialConvolution
#include "Tensor.hpp"              // for Tensor


namespace mtorch {

class TorchData;
class TorchStage;

  class SpatialConvolutionGemm final : public SpatialConvolution {
  public:
    // Constructor / Destructor
    SpatialConvolutionGemm(const uint32_t feats_in, const uint32_t feats_out,
      const uint32_t filt_height, const uint32_t filt_width,
      const uint32_t padw = 0, const uint32_t padh = 0);
    virtual ~SpatialConvolutionGemm() override;

    virtual void forwardProp(TorchData& input, TorchData **output) override;

    virtual void setWeights(const float* weights) override;
    virtual void setBiases(const float* biases) override;

    virtual void setWeightsFromStream( InputStream & ) override;
    virtual void setBiasesFromStream( InputStream & ) override;

    virtual Tensor<float>* weights() override { return weights_; }
    virtual Tensor<float>* biases() override { return biases_; }

    static TorchStage* loadFromStream( InputStream & stream ) noexcept;

  protected:

    void init(TorchData& input, TorchData **output, Tensor<float>** ones, Tensor<float>** columns);

    // Non-copyable, non-assignable.
    SpatialConvolutionGemm(SpatialConvolutionGemm&);
    SpatialConvolutionGemm& operator=(const SpatialConvolutionGemm&);
  };

};  // namespace mtorch
