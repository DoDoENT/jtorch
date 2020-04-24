//
//  SpatialDropout.hpp
//
//  Created by Jonathan Tompson on 2/6/2015.
//
//  This is a feed forward (testing) only version.  No actual dropout is 
//  implemented.
//

#pragma once
#include <string>          // for string, istream

#include "TorchStage.hpp"  // for ::SPATIAL_DROPOUT, TorchStage, TorchStageType


namespace mtorch {

class TorchData;

  class SpatialDropout : public TorchStage {
  public:
    // Constructor / Destructor
    SpatialDropout(const float p);
    virtual ~SpatialDropout();

    virtual TorchStageType type() const { return SPATIAL_DROPOUT; }
    virtual std::string name() const { return "SpatialDropout"; }
    virtual void forwardProp(TorchData& input, TorchData **output);

    static TorchStage* loadFromStream( InputStream & stream ) noexcept;

  protected:
    float p_;

    void init(TorchData& input, TorchData **output);

    // Non-copyable, non-assignable.
    SpatialDropout(SpatialDropout&);
    SpatialDropout& operator=(const SpatialDropout&);
  };
  
};  // namespace mtorch
