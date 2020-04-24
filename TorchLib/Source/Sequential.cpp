#include <math.h>                   // for fabsf, floor, log10, pow
#include <stdlib.h>                 // for NULL, exit
#include <ostream>                  // for istream, stringstream, operator<<, basic_ostream
#include <stdexcept>                // for runtime_error

#include "Tensor.hpp"               // for Tensor
#include "TorchData.hpp"            // for TorchData
#include "Utils/VectorManaged.hpp"  // for VectorManaged

#include "Sequential.hpp"


#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }


namespace mtorch {

  Sequential::Sequential() {
    // Create an empty container
    network_ = new data_str::VectorManaged<TorchStage*>(1);
    network_type_ = UNDEFINED;
  }

  Sequential::~Sequential() {
    SAFE_DELETE(network_);
  }

  void Sequential::add(TorchStage* stage) {
    network_->pushBack(stage);
  }

  TorchStage* Sequential::get(const uint32_t i) {
    return (*network_)[i];
  }

  uint32_t Sequential::size() const {
    return network_->size();
  }

  NetworkType Sequential::network_type() const{
    return network_type_;
  }

  std::vector<int> Sequential::labels(){
      return labels_;
  }

  Sequential* Sequential::loadFromStream( InputStream & stream ) noexcept
  {

    Sequential* ret = new Sequential();

    [[ maybe_unused ]] int type = stream.read< int >();
    int nn_type = stream.read< int >();
    if (nn_type == 1){
        ret->network_type_ = CONVNET;
    } else if (nn_type == 0) {
        ret->network_type_ = MLP;
    } else {
        ret->network_type_ = UNDEFINED;
    }

    int n_classes = stream.read< int >();
    ret->labels_.reserve( n_classes );
    for(int i = 0; i < n_classes; i++){
        ret->labels_.push_back( stream.read< int >() );
    }

    int n_nodes = stream.read< int >();

    ret->network_->capacity(n_nodes);
    for (int32_t i = 0; i < n_nodes; i++) {
      ret->network_->pushBack(TorchStage::loadFromStream(stream));
    }
    return ret;
  }

  void Sequential::forwardProp(TorchData& input, TorchData** output) {

    if (network_ == NULL) {
      throw std::runtime_error("Sequential::forwardProp() - ERROR: "
        "Network is empty!");
    }
    TorchData* data1 = &input;
    TorchData* data2 = NULL;
    (*network_)[0]->forwardProp(*data1, &data2);
    for (uint32_t i = 1; i < network_->size(); i++) {
      SAFE_DELETE(data1);
      data1 = data2;
      (*network_)[i]->forwardProp(*data1, &data2);
    }
    SAFE_DELETE(data1);
    *output = data2;


  }

  void Sequential::forwardProp(std::vector<float> &image_data, int image_dim, TorchData** output)
  {
      if (network_ == NULL) {
        throw std::runtime_error("Sequential::forwardProp() - ERROR: "
          "Network is empty!");
      }
      int tensor_dim;
      if (network_type_ == MLP){
          tensor_dim = 2;
      } else if (network_type_ == CONVNET){
          tensor_dim = 3;
      } else {
          exit(-1);
      }
      int tensor_size[3] = {image_dim, image_dim, 1};

      TorchData* input = new mtorch::Tensor<float>(tensor_dim, tensor_size, image_data.data());

      (*network_)[0]->forwardProp(*input, output);
      for (uint32_t i = 1; i < network_->size(); i++) {
        SAFE_DELETE(input);
        input = *output;
        (*network_)[i]->forwardProp(*input, output);
      }
      SAFE_DELETE(input);

  }

}  // namespace mtorch
