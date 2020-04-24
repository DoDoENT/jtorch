#include <stddef.h>  // for NULL

#include "TorchData.hpp"

#define SAFE_DELETE(x) if (x != NULL) { delete x; x = NULL; }
#define SAFE_DELETE_ARR(x) if (x != NULL) { delete[] x; x = NULL; }

namespace mtorch {

  TorchData::TorchData() {
  }

  TorchData::~TorchData() {
  }

}  // namespace mtorch
