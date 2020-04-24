//
//  SpatialMaxPooling.hpp
//
//  Created by Jonathan Tompson on 4/1/13.
//


#pragma once
#include <cstdint>         // for uint32_t
#include <string>          // for string, istream

#include "TorchStage.hpp"  // for ::SPATIAL_MAX_POOLING_STAGE, TorchStage, TorchStageType

namespace mtorch {
  
class TorchData;

  class SpatialMaxPooling : public TorchStage {
  public:
    // Constructor / Destructor
    SpatialMaxPooling(const uint32_t kw, const uint32_t kh, const uint32_t dw,
                      const uint32_t dh, const uint32_t padw, const uint32_t padh);
    virtual ~SpatialMaxPooling();

    virtual TorchStageType type() const { return SPATIAL_MAX_POOLING_STAGE; }
    virtual std::string name() const { return "SpatialMaxPooling"; }
    virtual void forwardProp(TorchData& input, TorchData **output);

    static TorchStage* loadFromStream( InputStream & stream ) noexcept;

  protected:
    uint32_t kw_;
    uint32_t kh_;
    uint32_t dw_;
    uint32_t dh_;
    uint32_t padw_;
    uint32_t padh_;

    void init(TorchData& input, TorchData **output);

    // Non-copyable, non-assignable.
    SpatialMaxPooling(SpatialMaxPooling&);
    SpatialMaxPooling& operator=(const SpatialMaxPooling&);
  };
  
};  // namespace mtorch
