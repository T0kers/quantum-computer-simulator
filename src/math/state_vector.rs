use core::panic;

use crate::math::{complex::Complex, Real};

#[derive(Debug)]
pub struct StateVector<const N: usize> {
    elems: [Complex; N]
}

impl<const N: usize> StateVector<N> {
    pub fn new(elems: [Complex; N]) -> StateVector<N> {
        assert!(N.is_power_of_two() && N > 1, "Length of StateVector must be a power of 2, and greater than 1.");
        let mut state = StateVector {
            elems
        };
        state.normalize_mut();
        state
    }
    pub fn from_real(elems: [Real; N]) -> StateVector<N> {
        StateVector::new(elems.map(|e| Complex::from_real(e)))
    }
    fn normalize_mut(&mut self) {
        let rhs = self.magnitude();
        for n in 0..N {
            self.elems[n] /= rhs;
        }
    }
    fn magnitude(&self) -> Real {
        self.magnitude_squared().sqrt()
    }
    fn magnitude_squared(&self) -> Real {
        let mut result = 0.0;
        for i in 0..N {
            result += self.elems[i].abs_squared();
        }
        result
    }

    pub fn apply_gate(&self, gate: UGate<2>, index: usize) -> Self {
        todo!()
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
struct UGate<const N: usize> {
    elems: [[Complex; N]; N]
}

impl<const N: usize> UGate<N> {
    pub fn new(elems: [[Complex; N]; N]) -> Self {
        let matrix = UGate { elems };
        assert!((matrix * matrix.complex_conj()).is_approx_equal(&UGate::identity()));
        assert!((matrix.complex_conj() * matrix).is_approx_equal(&UGate::<N>::identity()));
        matrix
    }
    pub fn from_real(elems: [[Real; N]; N]) -> Self {
        UGate::new(elems.map(|row| row.map(|e| Complex::from_real(e))))
    }
    fn new_zero() -> Self {
        UGate {
            elems: [[Complex::ZERO; N]; N]
        }
    }
    pub fn identity() -> Self {
        let mut elems = [[Complex::ZERO; N]; N];
        for n in 0..N {
            elems[n][n] = Complex::ONE;
        }
        UGate { elems }
    }
    pub fn complex_conj(&self) -> Self {
        let mut result = UGate::new_zero();
        for i in 0..N {
            for j in 0..N {
                result.elems[j][i] = self.elems[i][j].conj();
            }
        }
        result
    }
    pub fn is_approx_equal(&self, other: &Self) -> bool {
        for i in 0..N {
            for j in 0..N {
                if !self.elems[i][j].is_approx_equal(&other.elems[i][j]) {
                    return false;
                }
            }
        }
        true
    }
}

impl<const N: usize> std::ops::Mul<UGate<N>> for UGate<N> {
    type Output = UGate<N>;
    fn mul(self, rhs: UGate<N>) -> Self::Output {
        let mut result = UGate::new_zero();
        for m in 0..N {
            for p in 0..N {
                let field = &mut result.elems[m][p];
                for n in 0..N {
                    *field += self.elems[m][n] * rhs.elems[n][p];
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn multiplication() {
        let a = UGate::from_real([
            [1. / (2.0 as Real).sqrt(), 1. / (2.0 as Real).sqrt()],
            [1. / (2.0 as Real).sqrt(), -1. / (2.0 as Real).sqrt()],
        ]);
        assert!((a * a).is_approx_equal(&UGate::identity()));

        let b = UGate::from_real([
            [0., 1.],
            [1., 0.],
        ]);
        let c = UGate::new([
            [Complex::ZERO, Complex::new(0.0, -1.0)],
            [Complex::new(0.0, 1.0), Complex::ZERO],
        ]);
        let d = UGate::new([
            [Complex::new(0.0, 1.0), Complex::ZERO],
            [Complex::ZERO, Complex::new(0.0, -1.0)],
        ]);
        assert!((b * c).is_approx_equal(&d))
    }
}