//
//  Threshold.hpp
//
//  Created by Jonathan Tompson on 4/1/13.
//

#pragma once
#include <string>          // for string, istream

#include "TorchStage.hpp"  // for ::THRESHOLD_STAGE, TorchStage, TorchStageType

namespace mtorch {
  
class TorchData;

  class Threshold : public TorchStage {
  public:
    // Constructor / Destructor
    Threshold();
    virtual ~Threshold();

    virtual TorchStageType type() const { return THRESHOLD_STAGE; }
    virtual std::string name() const { return "Threshold"; }
    virtual void forwardProp(TorchData& input, TorchData **output);

    float threshold;  // Single threshold value
    float val;  // Single output value (when input < threshold)

    static TorchStage* loadFromStream( InputStream & stream ) noexcept;

  protected:
    void init(TorchData& input, TorchData **output);

    // Non-copyable, non-assignable.
    Threshold(Threshold&);
    Threshold& operator=(const Threshold&);
  };
  
};  // namespace mtorch
