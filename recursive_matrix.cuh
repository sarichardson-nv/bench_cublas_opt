#ifndef _RECURSIVE_MATRIX_CUH
#define _RECURSIVE_MATRIX_CUH

#include <Eigen/Core>
#include <Eigen/Dense>




template <typename Scalar_, int MatrixDim_, int Flags_>
struct RecursiveMatrix : RecursiveMatrix<Scalar_, (MatrixDim_ + 1) / 2, Flags_> {
    using Scalar = Scalar_;

    static constexpr int matrix_dim = MatrixDim_;
    static constexpr int matrix_size = matrix_dim * matrix_dim;


    using Element = RecursiveMatrix<Scalar_, (MatrixDim_ + 1) / 2, Flags_>;
    // using Element = Eigen::Matrix<Scalar, MatrixDim_, MatrixDim_, Flags_>;

    static constexpr int dim_threads = 2 * Element::dim_threads;
    static constexpr int n_threads = 4 * Element::n_threads;

};


template <typename Scalar_, int Flags_>
struct RecursiveMatrix<Scalar_, 2, Flags_> {
    using Scalar = Scalar_;

    static constexpr int matrix_dim = 2;
    static constexpr int matrix_size = matrix_dim * matrix_dim;

    static constexpr int dim_threads = 1;
    static constexpr int n_threads = 1;

    using Matrix = Eigen::Matrix<Scalar, 2, 2, Flags_>;
    using Element = Eigen::Matrix<Scalar, 2, 2, Flags_>;


    Matrix matrix;




};

#endif //_RECURSIVE_MATRIX_CUH