//
//  Reshape.hpp
//
//  Created by Jonathan Tompson on 4/1/13.
//
//  Works a little differently to the torch version...  It takes a 3D tensor
//  and just makes it into a 1D array, so it's not as general purpose.
//
//  But really this 1D array is just a straight copy of the input data (since
//  we define tensors as float* anyway).
//

#pragma once
#include <cstdint>         // for uint32_t
#include <string>          // for string, istream

#include "TorchStage.hpp"  // for ::RESHAPE_STAGE, TorchStage, TorchStageType

namespace mtorch {
  
class TorchData;

  class Reshape : public TorchStage {
  public:
    // Constructor / Destructor
    // For 1D tensor: set sz1 = -1, for 2D tensor: set sz2 = -1
    Reshape(const uint32_t dim, const uint32_t* size);
    virtual ~Reshape();

    virtual TorchStageType type() const { return RESHAPE_STAGE; }
    virtual std::string name() const { return "Reshape"; }
    virtual void forwardProp(TorchData& input, TorchData **output);

    static TorchStage* loadFromStream( InputStream & stream ) noexcept;

  protected:
    uint32_t odim_;
    uint32_t* osize_;
    void init(TorchData& input, TorchData **output);

    uint32_t outNElem() const;

    // Non-copyable, non-assignable.
    Reshape(Reshape&);
    Reshape& operator=(const Reshape&);
  };
  
};  // namespace mtorch
