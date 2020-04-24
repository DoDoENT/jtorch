//
//  Tensor.hpp
//
//  Created by Jonathan Tompson on 5/14/13.
//
//  Simplified C++ replica of torch.Tensor.  Up to 3D is supported.
//

#pragma once


#include "TorchData.hpp"

#include "Utils/InputStream.hpp"

#include <iomanip>
#include <iostream>
#include <cfloat>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>

#define mtorch_TENSOR_PRECISON 4
#define EPSILON (2 * FLT_EPSILON)

#define TO_TENSOR_PTR(x) ((x)->type() == mtorch::TorchDataType::TENSOR_DATA ? (mtorch::Tensor<float>*)(x) : NULL)

namespace mtorch {

  template <typename T>
  class Tensor : public TorchData {
  public:
    Tensor(const uint32_t dim, const uint32_t* size);
    // Tensor(const cv::Mat& mat_image, int n_dim);
    Tensor(const int dim, const int* size, float *data);
    virtual ~Tensor();

    virtual TorchDataType type() const { return TENSOR_DATA; }

	// setData and getData are EXPENSIVE --> They require a CPU to GPU copy
	void setData(const T* data);
	void setDataAt(const T data, int index);
    void setDataFromStream( InputStream & stream );
	T* getData();

    // View returns a new view on the same object.  The caller owns the new
    // memory (ie, it is transferred).
    Tensor<T>* view(const uint32_t dim, const uint32_t* size);

    uint32_t dim() const { return dim_; }
    const uint32_t* size() const { return size_; }
    bool isSameSizeAs(const Tensor<T>& src) const;

    // Print --> EXPENSIVE
    virtual void print();  // print to std::cout

    // Some simple tensor math operations
    static void copy(Tensor<T>& dst, Tensor<T>& src);
    static void add(Tensor<T>& dst, Tensor<T>& x, Tensor<T>& y);
    static void mul(Tensor<T>& x, float mul_value);
    static void div(Tensor<T>& x, float div_value);
    static void accumulate(Tensor<T>& dst, Tensor<T>& src);
    static void zero(Tensor<T>& x);
    static void fill(Tensor<T>& x, float value);
    // slowSum - This does a CPU copy because I haven't written a reduction
    // operator yet
    static float slowSum(Tensor<T>& x);

    // Some tensor math operations that return new tensors
    static Tensor<T>* clone(Tensor<T>& x);
    static Tensor<T>* gaussian1D(const uint32_t kernel_size);  // sigma = size / 2
    static Tensor<T>* gaussian(const uint32_t kernel_size);
    static Tensor<T>* loadFromFile(const std::string& file);
    static void saveToFile(Tensor<T>* tensor, const std::string& file);

    //inline const jcl::JCLBuffer& storage() const { return storage_; }
    inline uint32_t nelems() const;

    uint32_t* calcStride() const;  // memory returned is owned by caller

  protected:
	T* data_;
    uint32_t dim_;
    uint32_t* size_;  // size_[0] is lowest contiguous dimension,
                      // size_[2] is highest dimension

    Tensor();  // Default constructor used internally (in view function)

    // Non-copyable, non-assignable.
    Tensor(Tensor&);
    Tensor& operator=(const Tensor&);
  };

  template <typename T>
  Tensor<T>::Tensor(const uint32_t dim, const uint32_t* size) {
    this->dim_ = dim;
    this->size_ = new uint32_t[dim];
    memcpy(this->size_, size, sizeof(this->size_[0]) * dim);
	this->data_ = new T[this->nelems()]();
  }

  template <typename T>
  Tensor<T>::Tensor(const int dim, const int* size, float* data) {
    this->dim_ = dim;
    this->size_ = new uint32_t[dim];
    memcpy(this->size_, size, sizeof(this->size_[0]) * dim);
    this->data_ = NULL;
    setData(data);
  }

//   template <typename T>
//   Tensor<T>::Tensor(const cv::Mat& mat_image, int dim) {
//     this->dim_ = dim;
//     this->size_ = new uint32_t[this->dim_];
//     if (dim == 2){
//         size_[0] = (uint32_t)mat_image.cols;
//         size_[1] = (uint32_t)mat_image.rows;
//     } else if (dim == 3){

//         size_[0] = (uint32_t)mat_image.cols;
//         size_[1] = (uint32_t)mat_image.rows;
//         size_[2] = 1;
//     }

//     this->data_ = new T[this->nelems()]();
//     for(uint32_t i = 0; i < this->nelems(); i++){
//         data_[i] = (T)mat_image.data[i];
//     }
//   }

  template <typename T>
  Tensor<T>::Tensor() {
    // Default constructor returns an empty header.  Used internally (ie
    // private).
    dim_ = 0;
    size_ = NULL;
    data_ = NULL;
  }

  template <typename T>
  Tensor<T>::~Tensor() {
    if (size_ != NULL) {
      delete[] size_;
    }
    if (data_ != NULL){
      delete[] data_;
	}
  }

  template <typename T>
  uint32_t* Tensor<T>::calcStride() const {
    uint32_t* stride = new uint32_t[dim_];
    stride[0] = 1;
    for (uint32_t i = 1; i < dim_; i++) {
      stride[i] = stride[i-1] * size_[i-1];
    }
    return stride;
  }

  template <typename T>
  uint32_t Tensor<T>::nelems() const {
    uint32_t nelem = 1;
    for (uint32_t i = 0; i < dim_; i++) {
      nelem *= size_[i];
    }
    return nelem;
  }

  template <typename T>
  bool Tensor<T>::isSameSizeAs(const Tensor<T>& src) const {
    if (dim_ != src.dim_) {
      return false;
    }
    for (uint32_t i = 0; i < dim_; i++) {
      if (size_[i] != src.size_[i]) {
        return false;
      }
    }
    return true;
  }


  template <typename T>
  void Tensor<T>::setData(const T* data) {
      if (this->data_ != NULL){
		  delete[] this->data_;
	  }
	  this->data_ = new T[this->nelems()];
	  memcpy(this->data_, data, sizeof(this->data_[0]) * this->nelems());
  }

template< typename T >
void Tensor<T>::setDataFromStream( InputStream & stream )
{
    if (this->data_ != NULL){
        delete[] this->data_;
    }
    this->data_ = new T[ this->nelems() ];
    stream.readArray( this->data_, this->nelems() );

}

  template <typename T>
  void Tensor<T>::setDataAt(const T data, int index){
	  this->data_[index] = data;
  }

  template <typename T>
  T* Tensor<T>::getData() {
	  return this->data_;
  }


  template <typename T>
  void Tensor<T>::print() {
    std::streamsize prec = std::cout.precision();
    std::cout.precision(mtorch_TENSOR_PRECISON);
    T* d = getData();
    T max_val = std::numeric_limits<T>::min();
    for (uint32_t i = 0; i < nelems(); i++) {
      max_val = std::max<T>(max_val, d[i]);
    }
    T scale = (T)pow(10.0, floor(log10((double)max_val + EPSILON)));

#if defined(WIN32) || defined(_WIN32)
      std::cout.setf(0, std::ios::showpos);
#else
      std::cout.setf(std::ios::showpos);
#endif

    if (dim_ == 1) {
      // Print a 1D tensor
      std::cout << "  tensor[*] =" << std::endl;
      if (fabsf((float)scale - 1.0f) > EPSILON) {
        std::cout << " " << scale << " * " << std::endl;
      }
      std::cout.setf(std::ios::showpos);
      for (uint32_t u = 0; u < size_[0]; u++) {
        if (u == 0) {
          std::cout << " (0) ";
        } else {
          std::cout << "     ";
        }
        std::cout << std::fixed << d[u] / scale << std::endl;;
        std::cout.unsetf(std::ios_base::floatfield);
      }
    } else if (dim_ == 2) {
      // Print a 2D tensor
      std::cout << "  tensor[*,*] =" << std::endl;
      if (fabsf((float)scale - 1.0f) > EPSILON) {
        std::cout << " " << scale << " * " << std::endl;
      }
      std::cout.setf(std::ios::showpos);
      for (uint32_t v = 0; v < size_[1]; v++) {
        if (v == 0) {
          std::cout << " (0,0) ";
        } else {
          std::cout << "       ";
        }
        std::cout.setf(std::ios::showpos);
        for (uint32_t u = 0; u < size_[0]; u++) {
          std::cout << std::fixed << d[v * size_[0] + u] / scale;
          std::cout.unsetf(std::ios_base::floatfield);
          if (u != size_[0] - 1) {
            std::cout << ", ";
          } else {
            std::cout << std::endl;
          }
        }
      }
    } else {
      // Print a nD tensor
      int32_t odim = 1;
      for (uint32_t i = 2; i < dim_; i++) {
        odim *= size_[i];
      }

      uint32_t* stride = calcStride();

      for (int32_t i = 0; i < odim; i++) {
        std::cout << "  tensor[";
        for (uint32_t cur_dim = dim_-1; cur_dim >= 2; cur_dim--) {
          std::cout << (i % stride[cur_dim]) << ",";
        }

        std::cout << "*,*] =";
        std::cout << std::endl;
        if (fabsf((float)scale - 1.0f) > EPSILON) {
          std::cout << " " << scale << " * " << std::endl;
        }

        T* data = &d[i * size_[1] * size_[0]];
        for (uint32_t v = 0; v < size_[1]; v++) {
          if (v == 0) {
            std::cout << " (0,0) ";
          } else {
            std::cout << "       ";
          }
          std::cout.setf(std::ios::showpos);
          for (uint32_t u = 0; u < size_[0]; u++) {
            std::cout << std::fixed << data[v * size_[0] + u] / scale;
            std::cout.unsetf(std::ios_base::floatfield);
            if (u != size_[0] - 1) {
              std::cout << ", ";
            } else {
              std::cout << std::endl;
            }
          }
        }
      }

      delete[] stride;
    }
    std::cout.precision(prec);
    std::cout << std::resetiosflags(std::ios_base::showpos);

    std::cout << "[mtorch.";
    std::cout << " of dimension ";
    for (int32_t i = (int32_t)dim_-1; i >= 0; i--) {
      std::cout << size_[i];
      if (i > 0) {
        std::cout << "x";
      }
    }
    std::cout << "]" << std::endl;
  };



  template <typename T>
  Tensor<T>* Tensor<T>::view(const uint32_t dim, const uint32_t* size) {

      if (dim == 0) {
        throw std::runtime_error("ERROR - view() - zero dimension not allowed!");
      }
      uint32_t view_nelem = 1;
      for (uint32_t i = 0; i < dim; i++) {
        view_nelem *= size[i];
      }

      if (view_nelem != nelems()) {
        throw std::runtime_error("ERROR - view() - Size mismatch!");
      }

      Tensor<T>* return_header = new Tensor<T>();
      return_header->dim_ = dim;
      return_header->size_ = new uint32_t[dim];
      memcpy(return_header->size_, size, sizeof(return_header->size_[0]) * dim);
      return_header->setData(data_);
      return return_header;
  }

  template <typename T>
  Tensor<T>* Tensor<T>::gaussian1D(const uint32_t kernel_size) {
    const uint32_t size = kernel_size;
    Tensor<T>* ret = new Tensor<T>(1, &size);
    const float sigma = 0.25f;
    const float amplitude = 1.0f;
    const float center = (float)kernel_size/2.0f + 0.5f;
    T* data = new T[kernel_size];
    for (uint32_t i = 0; i < kernel_size; i++) {
      data[i] = (T)amplitude * expf(-(powf(((float)(i+1) - center) /
        (sigma*(float)kernel_size), 2.0f) / 2.0f));
    }
    ret->setData(data);
    delete[] data;
    return ret;
  }

  template <typename T>
  Tensor<T>* Tensor<T>::gaussian(const uint32_t kernel_size) {
    const uint32_t size[2] = {kernel_size, kernel_size};
    Tensor<T>* ret = new Tensor<T>(2, size);
    const float sigma = 0.25f;
    const float amplitude = 1.0f;
    const float center = (float)kernel_size/2.0f + 0.5f;
    T* data = new T[kernel_size * kernel_size];
    for (uint32_t v = 0; v < kernel_size; v++) {
      for (uint32_t u = 0; u < kernel_size; u++) {
        float du = ((float)(u+1) - center) / (sigma*(float)kernel_size);
        float dv = ((float)(v+1) - center) / (sigma*(float)kernel_size);
        data[v * kernel_size + u] = (T)amplitude * expf(-(du * du + dv * dv) /
          2.0f);
      }
    }
    ret->setData(data);
    delete[] data;
    return ret;
  }

  template <typename T>
  Tensor<T>* Tensor<T>::clone(Tensor<T>& x) {
	Tensor<T>* ret = new Tensor<T>(x.dim_, x.size_);
	ret->setData(x.getData());
    return ret;
  }

  template <typename T>
  void Tensor<T>::copy(Tensor<T>& dst, Tensor<T>& src) {
	  T* data = src.getData();
	  dst.setData(data);
  }

  template <typename T>
  void Tensor<T>::add(Tensor<T>& dst, Tensor<T>& x, Tensor<T>& y) {
    uint32_t nelem = dst.nelems();
	T* src1 = x.getData();
	T* src2 = y.getData();
	for (uint32_t i = 0; i < nelem; i++)
	{
		dst.setDataAt(src1[i] + src2[i], i);
	}
  }

  template <typename T>
  void Tensor<T>::mul(Tensor<T>& x, float mul_val) {
	  uint32_t nelem = x.nelems();
	  T* data = x.getData();
	  for (uint32_t i = 0; i < nelem; i++)
	  {
		  x.setDataAt(data[i] * mul_val, i);
	  }
  }

  template <typename T>
  void Tensor<T>::div(Tensor<T>& x, float div_val) {
	  uint32_t nelem = x.nelems();
	  T* data = x.getData();
	  for (uint32_t i = 0; i < nelem; i++)
	  {
		  x.setDataAt(data[i] / div_val, i);
	  }
  }

  template <typename T>
  void Tensor<T>::accumulate(Tensor<T>& dst, Tensor<T>& src) {
	  uint32_t nelem = dst.nelems();
	  T* base = dst.getData();
	  T* addition = src.getData();
	  for (uint32_t i = 0; i < nelem; i++)
	  {
		  dst.setDataAt(base[i] + addition[i], i);
	  }
  }

  template <typename T>
  float Tensor<T>::slowSum(Tensor<T>& x) {
	float* temp = x.getData();
    float sum = 0.0f;
    for (uint32_t i = 0; i < x.nelems(); i++) {
      sum += temp[i];
    }
    return sum;
  }


	template <typename T>
	void Tensor<T>::zero(Tensor<T>& dst) {
		Tensor<T>::fill(dst, 0);
	}

	template <typename T>
	void Tensor<T>::fill(Tensor<T>& dst, float value) {
		int32_t nelems = dst.nelems();
		for (int32_t i = 0; i < nelems; i++) {
			dst.setDataAt(value, i);
		}
	}

  template <typename T>
  Tensor<T>* Tensor<T>::loadFromFile(const std::string& file) {
    Tensor<T>* new_tensor = NULL;
    std::ifstream ifile(file.c_str(), std::ios::in|std::ios::binary);
    if (ifile.is_open()) {
      ifile.seekg(0, std::ios::beg);
      // Now load the Tensor
      int32_t dim;
      ifile.read((char*)(&dim), sizeof(dim));
      uint32_t* size = new uint32_t[dim];
      for (int32_t i = 0; i < dim; i++) {
        int32_t cur_size;
        ifile.read((char*)(&cur_size), sizeof(cur_size));
        size[dim-i-1] = (uint32_t)cur_size;
      }
      new_tensor = new Tensor<T>(dim, size);

      T* data = new T[new_tensor->nelems()];
      ifile.read((char*)(data), sizeof(data[0]) * new_tensor->nelems());
      new_tensor->setData(data);
      delete[] data;
      ifile.close();
      delete[] size;
    } else {
      std::stringstream ss;
      ss << "Tensor<T>::loadFromFile() - ERROR: Could not open file ";
      ss << file << '\n';
      throw std::runtime_error(ss.str());
    }
    return new_tensor;
  }

  template <typename T>
  void Tensor<T>::saveToFile(Tensor<T>* tensor, const std::string& file) {
    std::ofstream ofile(file.c_str(), std::ios::out|std::ios::binary);
    if (ofile.is_open()) {
      // Now save the Tensor
      int32_t dim = tensor->dim_;
      ofile.write((char*)(&dim), sizeof(dim));
      for (int32_t i = dim-1; i >= 0; i--) {
        int32_t cur_size = tensor->size_[i];
        ofile.write((char*)(&cur_size), sizeof(cur_size));
      }
	  T* data = tensor->getData();
      ofile.write((char*)(data), sizeof(data[0]) * tensor->nelems());
      ofile.close();
    } else {
      std::stringstream ss;
      ss << "Tensor<T>::saveToFile() - ERROR: Could not open file ";
      ss << file << '\n';
      throw std::runtime_error(ss.str());
    }
  }

};  // namespace mtorch
