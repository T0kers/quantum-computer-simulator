use super::{*, complex::Complex};

pub type RowVector<const N: usize> = Matrix<1, N>;
pub type ColVector<const N: usize> = Matrix<N, 1>;

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Matrix<const M: usize, const N: usize> {
    elems: [[Complex; N]; M]
}

pub trait IsRowVector {}
pub trait IsColVector {}
impl<const N: usize> IsRowVector for Matrix<1, N> {}
impl<const M: usize> IsColVector for Matrix<M, 1> {}
pub trait IsMatrix {}

impl<const M: usize, const N: usize> Matrix<M, N> {
    pub fn new(elems: [[Complex; N]; M]) -> Matrix<M, N> {
        Matrix::<M, N> {
            elems
        }
    }
    pub fn new_zero() -> Matrix<M, N> {
        Matrix::<M, N> {
            elems: [[Complex::ZERO; N]; M]
        }
    }
    pub fn from_real(elems: [[Real; N]; M]) -> Matrix<M, N> {
        Matrix {
            elems: elems.map(|row| row.map(|e| Complex::from_real(e)))
        }
    }
    pub fn new_row(elems: [Complex; N]) -> RowVector::<N> {
        RowVector::<N> {
            elems: [elems; 1]
        }
    }
    pub fn row_from_real(elems: [Real; N]) -> RowVector::<N> {
        RowVector::<N> {
            elems: [elems.map(|e| Complex::from_real(e)); 1]
        }
    }
    pub fn new_col(elems: [Complex; M]) -> ColVector<M> {
        ColVector::<M> {
            elems: elems.map(|e| [e; 1])
        }
    }
    pub fn col_from_real(elems: [Real; M]) -> ColVector<M> {
        ColVector::<M> {
            elems: elems.map(|e| [Complex::from_real(e); 1])
        }
    }
    // pub fn tensor<const O: usize, const P: usize>(&self, rhs: &Matrix<O, P>) -> Matrix<{ mul_dims(M, O) }, { mul_dims(N, P) }> {
    //     let mut result = Matrix::new_zero();
        
    //     for i in 0..M {
    //         for j in 0..N {
    //             for k in 0..O {
    //                 for l in 0..P {
    //                     result.elems[i * O + k][j * P + l] = self.elems[i][j] * rhs.elems[k][l];
    //                 }
    //             }
    //         }
    //     }
        
    //     result
    // }
}

pub struct Not<T>(std::marker::PhantomData<T>);

// Implement IsMatrix for matrices that are neither row nor column vectors
impl<const M: usize, const N: usize> IsMatrix for Matrix<M, N> 
where 
    Self: Sized,
    Not<Self>: IsRowVector,
    Not<Self>: IsColVector,
{}

pub trait RowVectorOperations<const N: usize> {
    fn normalize(self) -> Self;
    fn magnitude(self) -> Real;
    fn magnitude_squared(self) -> Real;
}

pub trait ColVectorOperations<const N: usize> {
    fn normalize(self) -> Self;
    fn magnitude(self) -> Real;
    fn magnitude_squared(self) -> Real;
}

impl<const M: usize, const N: usize> RowVectorOperations<N> for Matrix<M, N> 
where Self: IsRowVector {
    fn normalize(self) -> Self {
        self / self.magnitude()
    }
    fn magnitude(self) -> Real {
        self.magnitude_squared().sqrt()
    }
    fn magnitude_squared(self) -> Real {
        let mut result = 0.0;
        for i in 0..N {
            result += self.elems[0][i].abs_squared();
        }
        result
    }
}

impl<const N: usize> std::ops::Mul<Matrix<N, 1>> for Matrix<1, N> 
where
    Self: IsRowVector,
    Matrix<N, 1>: IsColVector,
{
    type Output = Complex;
    fn mul(self, rhs: Matrix<N, 1>) -> Self::Output {
        let mut result = Complex::ZERO;
        for i in 0..N {
            result += self.elems[0][i] * rhs.elems[i][0];
        }
        result
    }
}

impl<const M: usize, const N: usize, const P: usize> std::ops::Mul<Matrix<N, P>> for Matrix<M, N> 
where
    Self: IsMatrix,
    Matrix<N, P>: IsMatrix,
{
    type Output = Matrix<M, P>;
    fn mul(self, rhs: Matrix<N, P>) -> Self::Output {
        let mut result = Matrix::<M, P>::new_zero();
        for m in 0..M {
            for p in 0..P {
                let field = &mut result.elems[m][p];
                for n in 0..N {
                    *field += self.elems[m][n] * rhs.elems[n][p];
                }
            }
        }
        result
    }
}

impl<const M: usize, const N: usize> ColVectorOperations<M> for Matrix<M, N> 
where Self: IsColVector {
    fn normalize(self) -> Self {
        self / self.magnitude()
    }
    fn magnitude(self) -> Real {
        self.magnitude_squared().sqrt()
    }
    fn magnitude_squared(self) -> Real {
        let mut result = 0.0;
        for i in 0..N {
            result += self.elems[i][0].abs_squared();
        }
        result
    }
}


impl<const M: usize, const N: usize> std::ops::Add<Matrix<M, N>> for Matrix<M, N> {
    type Output = Matrix<M, N>;
    fn add(mut self, rhs: Matrix<M, N>) -> Self::Output {
        for m in 0..M {
            for n in 0..N {
                self.elems[m][n] += rhs.elems[m][n];
            }
        }
        self
    }
}

impl<const M: usize, const N: usize> std::ops::Sub<Matrix<M, N>> for Matrix<M, N> {
    type Output = Matrix<M, N>;
    fn sub(mut self, rhs: Matrix<M, N>) -> Self::Output {
        for m in 0..M {
            for n in 0..N {
                self.elems[m][n] -= rhs.elems[m][n];
            }
        }
        self
    }
}

impl<const M: usize, const N: usize> std::ops::Neg for Matrix<M, N> {
    type Output = Matrix<M, N>;
    fn neg(mut self) -> Self::Output {
        for m in 0..M {
            for n in 0..N {
                self.elems[m][n] *= -1.0;
            }
        }
        self
    }
}

impl<const M: usize, const N: usize> std::ops::Mul<Complex> for Matrix<M, N> {
    type Output = Matrix<M, N>;
    fn mul(mut self, rhs: Complex) -> Self::Output {
        for m in 0..M {
            for n in 0..N {
                self.elems[m][n] *= rhs;
            }
        }
        self
    }
}

impl<const M: usize, const N: usize> std::ops::Mul<Matrix<M, N>> for Complex {
    type Output = Matrix<M, N>;
    fn mul(self, mut rhs: Matrix<M, N>) -> Self::Output {
        for m in 0..M {
            for n in 0..N {
                rhs.elems[m][n] *= self;
            }
        }
        rhs
    }
}

impl<const M: usize, const N: usize> std::ops::Div<Complex> for Matrix<M, N> {
    type Output = Matrix<M, N>;
    fn div(mut self, rhs: Complex) -> Self::Output {
        for m in 0..M {
            for n in 0..N {
                self.elems[m][n] /= rhs;
            }
        }
        self
    }
}

impl<const M: usize, const N: usize> std::ops::Mul<Real> for Matrix<M, N> {
    type Output = Matrix<M, N>;
    fn mul(mut self, rhs: Real) -> Self::Output {
        for m in 0..M {
            for n in 0..N {
                self.elems[m][n] *= rhs;
            }
        }
        self
    }
}

impl<const M: usize, const N: usize> std::ops::Mul<Matrix<M, N>> for Real {
    type Output = Matrix<M, N>;
    fn mul(self, mut rhs: Matrix<M, N>) -> Self::Output {
        for m in 0..M {
            for n in 0..N {
                rhs.elems[m][n] *= self;
            }
        }
        rhs
    }
}

impl<const M: usize, const N: usize> std::ops::Div<Real> for Matrix<M, N> {
    type Output = Matrix<M, N>;
    fn div(mut self, rhs: Real) -> Self::Output {
        for m in 0..M {
            for n in 0..N {
                self.elems[m][n] /= rhs;
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn commutativity() {
        assert_eq!(ColVector::col_from_real([1., 2.]) + ColVector::col_from_real([3., 4.]), ColVector::col_from_real([3., 4.]) + ColVector::col_from_real([1., 2.]));
        assert_eq!(ColVector::col_from_real([1., 2.]) - ColVector::col_from_real([3., 4.]), -ColVector::col_from_real([3., 4.]) + ColVector::col_from_real([1., 2.]));
        assert_eq!(ColVector::col_from_real([1., 2.]) * 3., 3. * ColVector::col_from_real([1., 2.]));

    }
}