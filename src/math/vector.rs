use super::{*, complex::Complex};

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vector<const N: usize> {
    elems: [Complex; N]
}

impl<const N: usize> Vector<N> {
    pub fn new(elems: [Complex; N]) -> Vector<N> {
        Vector::<N> {
            elems
        }
    }
    pub fn from_real(real_elems: [Real; N]) -> Vector<N> {
        let mut elems = [Complex::ZERO; N];

        for i in 0..N {
            elems[i] = Complex::from_real(real_elems[i]);
        }

        Vector { elems }
    }
    pub fn normalize(self) -> Vector<N> {
        self / self.magnitude()
    }
    pub fn magnitude(self) -> Real {
        self.magnitude_squared().sqrt()
    }
    pub fn magnitude_squared(self) -> Real {
        let mut result = 0.0;
        for i in 0..N {
            result += self.elems[i].abs_squared();
        }
        result
    }
}

impl<const N: usize> std::ops::Add<Vector<N>> for Vector<N> {
    type Output = Vector<N>;
    fn add(mut self, rhs: Vector<N>) -> Self::Output {
        for i in 0..N {
            self.elems[i] += rhs.elems[i];
        }
        self
    }
}

impl<const N: usize> std::ops::Sub<Vector<N>> for Vector<N> {
    type Output = Vector<N>;
    fn sub(mut self, rhs: Vector<N>) -> Self::Output {
        for i in 0..N {
            self.elems[i] -= rhs.elems[i];
        }
        self
    }
}

impl<const N: usize> std::ops::Mul<Real> for Vector<N> {
    type Output = Vector<N>;
    fn mul(mut self, rhs: Real) -> Self::Output {
        for i in 0..N {
            self.elems[i] *= rhs;
        }
        self
    }
}

impl<const N: usize> std::ops::Div<Real> for Vector<N> {
    type Output = Vector<N>;
    fn div(mut self, rhs: Real) -> Self::Output {
        for i in 0..N {
            self.elems[i] /= rhs;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bruh() {

    }
}