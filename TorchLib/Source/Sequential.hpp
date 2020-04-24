//
//  Sequential.hpp
//
//  Created by Jonathan Tompson on 4/2/13.
//

#pragma once
#include <cstdint>         // for uint32_t
#include <string>          // for string, istream
#include <vector>          // for vector

#include "TorchStage.hpp"  // for ::SEQUENTIAL_STAGE, TorchStage, TorchStageType

namespace data_str {
template <typename T> class VectorManaged;
}  // namespace data_str

namespace mtorch {

class TorchData;

  typedef enum {
     UNDEFINED = -1,
     MLP = 0,
     CONVNET = 1,
  } NetworkType;

  class Sequential : public TorchStage {
  public:
    // Constructor / Destructor
    Sequential();
    virtual ~Sequential();

    virtual TorchStageType type() const { return SEQUENTIAL_STAGE; }
    virtual NetworkType network_type() const;
    virtual std::string name() const { return "Sequential"; }
    virtual void forwardProp(TorchData& input, TorchData **output);
    void forwardProp(std::vector<float> &image_data, int image_dim, TorchData **output);
    std::vector<int> labels();

    void add(TorchStage* stage);
    TorchStage* get(const uint32_t i);
    uint32_t size() const;


    static Sequential* loadFromStream( InputStream & stream ) noexcept;

  protected:
    data_str::VectorManaged<TorchStage*>* network_;
    NetworkType network_type_;
    std::vector<int> labels_;
    // Non-copyable, non-assignable.
    Sequential(Sequential&);
    Sequential& operator=(const Sequential&);
  };

};  // namespace mtorch
