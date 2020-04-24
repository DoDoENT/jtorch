//
//  Linear.hpp
//
//  Created by Jonathan Tompson on 4/1/13.
//

#pragma once

#include "torch_stage.h"

#include <cstdint>
#include <string>


namespace jtorch {

class TorchData;
  template <typename T> class Tensor;

  class Linear : public TorchStage {
  public:
    // Constructor / Destructor
    Linear(const uint32_t n_inputs, const uint32_t n_outputs);
    virtual ~Linear();

    virtual TorchStageType type() const { return LINEAR_STAGE; }
    virtual std::string name() const { return "Linear"; }
    virtual void forwardProp(TorchData& input, TorchData **output);

    void setWeights(const float* weights);
    void setWeightsFromStream( InputStream & stream );
    void setBiases(const float* biases);
    void setBiasesFromStream( InputStream & stream );
    Tensor<float>* weights() { return weights_; }
    Tensor<float>* biases() { return biases_; }

    static TorchStage* loadFromStream( InputStream & stream ) noexcept;

  protected:
    uint32_t n_inputs_;
    uint32_t n_outputs_;

    Tensor<float>* weights_;  // n_outputs (rows) * n_inputs (columns), stored row major
    Tensor<float>* biases_;  // n_outputs

    void init(TorchData& input, TorchData **output);

    // Non-copyable, non-assignable.
    Linear(Linear&);
    Linear& operator=(const Linear&);
  };

};  // namespace jtorch
