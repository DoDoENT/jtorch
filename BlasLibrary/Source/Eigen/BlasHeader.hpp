// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <Eigen/Core>
#include <Eigen/Jacobi>


#define NOTR    0
#define TR      1
#define ADJ     2

#define LEFT    0
#define RIGHT   1

#define UP      0
#define LO      1

#define NUNIT   0
#define UNIT    1

#define INVALID 0xff

#define OP(X)   (   ((X)=='N' || (X)=='n') ? NOTR   \
                  : ((X)=='T' || (X)=='t') ? TR     \
                  : ((X)=='C' || (X)=='c') ? ADJ    \
                  : INVALID)

#define SIDE(X) (   ((X)=='L' || (X)=='l') ? LEFT   \
                  : ((X)=='R' || (X)=='r') ? RIGHT  \
                  : INVALID)

#define UPLO(X) (   ((X)=='U' || (X)=='u') ? UP     \
                  : ((X)=='L' || (X)=='l') ? LO     \
                  : INVALID)

#define DIAG(X) (   ((X)=='N' || (X)=='n') ? NUNIT  \
                  : ((X)=='U' || (X)=='u') ? UNIT   \
                  : INVALID)


inline bool check_op(const char* op)
{
  return OP(*op)!=0xff;
}

inline bool check_side(const char* side)
{
  return SIDE(*side)!=0xff;
}

inline bool check_uplo(const char* uplo)
{
  return UPLO(*uplo)!=0xff;
}

using namespace Eigen;

typedef float Scalar;
typedef NumTraits<Scalar>::Real RealScalar;
typedef std::complex<RealScalar> Complex;

enum
{
  IsComplex = Eigen::NumTraits<float>::IsComplex,
  Conj = IsComplex
};

typedef Matrix<Scalar,Dynamic,Dynamic,ColMajor> PlainMatrixType;
typedef Map<Matrix<Scalar,Dynamic,Dynamic,ColMajor>, 0, OuterStride<> > MatrixType;
typedef Map<Matrix<Scalar,Dynamic,1>, 0, InnerStride<Dynamic> > StridedVectorType;
typedef Map<Matrix<Scalar,Dynamic,1> > CompactVectorType;

template<typename T>
Map<Matrix<T,Dynamic,Dynamic,ColMajor>, 0, OuterStride<> >
matrix(T* data, int rows, int cols, int stride)
{
  return Map<Matrix<T,Dynamic,Dynamic,ColMajor>, 0, OuterStride<> >(data, rows, cols, OuterStride<>(stride));
}

template<typename T>
Map<Matrix<T,Dynamic,1>, 0, InnerStride<Dynamic> > vector(T* data, int size, int incr)
{
  return Map<Matrix<T,Dynamic,1>, 0, InnerStride<Dynamic> >(data, size, InnerStride<Dynamic>(incr));
}

template<typename T>
Map<Matrix<T,Dynamic,1> > vector(T* data, int size)
{
  return Map<Matrix<T,Dynamic,1> >(data, size);
}

template<typename T>
T* get_compact_vector(T* x, int n, int incx)
{
  if(incx==1)
    return x;

  T* ret = new Scalar[n];
  if(incx<0) vector(ret,n) = vector(x,n,-incx).reverse();
  else       vector(ret,n) = vector(x,n, incx);
  return ret;
}

template<typename T>
T* copy_back(T* x_cpy, T* x, int n, int incx)
{
  if(x_cpy==x)
    return 0;

  if(incx<0) vector(x,n,-incx).reverse() = vector(x_cpy,n);
  else       vector(x,n, incx)           = vector(x_cpy,n);
  return x_cpy;
}

#define EIGEN_BLAS_FUNC(X) EIGEN_CAT(eigen_,X)

static inline int EIGEN_BLAS_FUNC(gemm)(char *opa, char *opb, int *m, int *n, int *k, RealScalar *palpha, RealScalar *pa, int *lda, RealScalar *pb, int *ldb, RealScalar *pbeta, RealScalar *pc, int *ldc)
{
//   std::cerr << "in gemm " << *opa << " " << *opb << " " << *m << " " << *n << " " << *k << " " << *lda << " " << *ldb << " " << *ldc << " " << *palpha << " " << *pbeta << "\n";
  typedef void (*functype)(DenseIndex, DenseIndex, DenseIndex, const Scalar *, DenseIndex, const Scalar *, DenseIndex, Scalar *, DenseIndex, Scalar, internal::level3_blocking<Scalar,Scalar>&, Eigen::internal::GemmParallelInfo<DenseIndex>*);
  static functype func[12];

  static bool init = false;
  if(!init)
  {
    for(int kk=0; kk<12; ++kk)
      func[kk] = 0;
    func[NOTR  | (NOTR << 2)] = (internal::general_matrix_matrix_product<DenseIndex,Scalar,ColMajor,false,Scalar,ColMajor,false,ColMajor>::run);
    func[TR    | (NOTR << 2)] = (internal::general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,false,Scalar,ColMajor,false,ColMajor>::run);
    func[ADJ   | (NOTR << 2)] = (internal::general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,ColMajor,false,ColMajor>::run);
    func[NOTR  | (TR   << 2)] = (internal::general_matrix_matrix_product<DenseIndex,Scalar,ColMajor,false,Scalar,RowMajor,false,ColMajor>::run);
    func[TR    | (TR   << 2)] = (internal::general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,false,Scalar,RowMajor,false,ColMajor>::run);
    func[ADJ   | (TR   << 2)] = (internal::general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,RowMajor,false,ColMajor>::run);
    func[NOTR  | (ADJ  << 2)] = (internal::general_matrix_matrix_product<DenseIndex,Scalar,ColMajor,false,Scalar,RowMajor,Conj, ColMajor>::run);
    func[TR    | (ADJ  << 2)] = (internal::general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,false,Scalar,RowMajor,Conj, ColMajor>::run);
    func[ADJ   | (ADJ  << 2)] = (internal::general_matrix_matrix_product<DenseIndex,Scalar,RowMajor,Conj, Scalar,RowMajor,Conj, ColMajor>::run);
    init = true;
  }

  Scalar* a = reinterpret_cast<Scalar*>(pa);
  Scalar* b = reinterpret_cast<Scalar*>(pb);
  Scalar* c = reinterpret_cast<Scalar*>(pc);
  Scalar alpha  = *reinterpret_cast<Scalar*>(palpha);
  Scalar beta   = *reinterpret_cast<Scalar*>(pbeta);

  if(beta!=Scalar(1))
  {
    if(beta==Scalar(0)) matrix(c, *m, *n, *ldc).setZero();
    else                matrix(c, *m, *n, *ldc) *= beta;
  }

  internal::gemm_blocking_space<ColMajor,Scalar,Scalar,Dynamic,Dynamic,Dynamic> blocking(*m,*n,*k, 1, false);

  int code = OP(*opa) | (OP(*opb) << 2);
  func[code](*m, *n, *k, a, *lda, b, *ldb, c, *ldc, alpha, blocking, 0);
  return 0;
}

