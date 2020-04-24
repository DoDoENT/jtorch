
#pragma once


#include "SpatialConvolution.hpp"
#include "SpatialConvolutionGemm.hpp"


namespace mtorch {

class SpatialConvolutionFactory {

public:
    static TorchStage* loadFromStream( InputStream & stream ) noexcept
    {
        return SpatialConvolutionGemm::loadFromStream(stream);
    }

    static SpatialConvolution* create(const uint32_t feats_in, const uint32_t feats_out,
                              const uint32_t filt_height, const uint32_t filt_width,
                              const uint32_t padw = 0, const uint32_t padh = 0) {

        return new SpatialConvolutionGemm(feats_in, feats_out, filt_height, filt_width, padw, padh);
    }

};

}
