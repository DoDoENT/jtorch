//
//  Tanh.hpp
//
//  Created by Jonathan Tompson on 4/1/13.
//

#pragma once
#include <string>          // for string, istream

#include "TorchStage.hpp"  // for ::TANH_STAGE, TorchStage, TorchStageType

namespace mtorch {
  
class TorchData;

  class Tanh : public TorchStage {
  public:
    // Constructor / Destructor
    Tanh();
    virtual ~Tanh();

    virtual TorchStageType type() const { return TANH_STAGE; }
    virtual std::string name() const { return "Tanh"; }
    virtual void forwardProp(TorchData& input, TorchData **output);

    static TorchStage* loadFromStream( InputStream & ) noexcept;

  protected:
    void init(TorchData& input, TorchData **output);

    // Non-copyable, non-assignable.
    Tanh(Tanh&);
    Tanh& operator=(const Tanh&);
  };
  
};  // namespace mtorch
