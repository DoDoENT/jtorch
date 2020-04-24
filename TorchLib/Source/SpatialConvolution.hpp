//
//  SpatialConvolution.hpp
//
//  Created by Jonathan Tompson on 5/15/13.
//

#pragma once

#include "TorchStage.hpp"
#include "TorchData.hpp"

#include <string>
#include <cstdint>

namespace mtorch {

  template <typename T> class Tensor;
  
  class SpatialConvolution : public TorchStage {
  public:
    // Constructor / Destructor
    SpatialConvolution();
    virtual ~SpatialConvolution();

    virtual TorchStageType type() const { return SPATIAL_CONVOLUTION_STAGE; }
    virtual std::string name() const { return "SpatialConvolution"; }

    virtual void setWeights(const float* weights) = 0;
    virtual void setBiases(const float* biases) = 0;
    virtual void setWeightsFromStream( InputStream & ) = 0;
    virtual void setBiasesFromStream( InputStream & ) = 0;
    virtual Tensor<float>* weights() = 0;
    virtual Tensor<float>* biases() = 0;

  protected:
    uint32_t filt_width_;
    uint32_t filt_height_;
    uint32_t feats_in_;
    uint32_t feats_out_;
    uint32_t padw_;
    uint32_t padh_;

    Tensor<float>* weights_;
    Tensor<float>* biases_;

    // Non-copyable, non-assignable.
    SpatialConvolution(SpatialConvolution&);
    SpatialConvolution& operator=(const SpatialConvolution&);
  };
  
};  // namespace mtorch
