use super::*;

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Complex {
    pub re: Real,
    pub im: Real,
}

impl Complex {
    const EPSILON: f64 = 1e-10;
    pub const ZERO: Self = Complex::new(0., 0.);
    pub const ONE: Self = Complex::new(1., 0.);
    pub const fn new(re: Real, im: Real) -> Self {
        Complex {
            re, im
        }
    }
    pub fn from_real(re: Real) -> Self {
        Self::new(re, 0.)
    }
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }
    pub fn abs(self) -> Real {
        Real::sqrt(self.re * self.re + self.im * self.im)
    }
    pub fn abs_squared(self) -> Real {
        self.re * self.re + self.im * self.im
    }
    pub fn is_approx_equal(&self, other: &Self) -> bool {
        (self.re - other.re).abs() < Self::EPSILON && (self.im - other.im).abs() < Self::EPSILON
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl std::ops::AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl std::ops::Sub for Complex {
    type Output = Complex;
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl std::ops::SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl std::ops::Neg for Complex {
    type Output = Complex;
    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.re * rhs.re - self.im * rhs.im, self.re * rhs.im + self.im * rhs.re)
    }
}

impl std::ops::MulAssign for Complex {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl std::ops::Div for Complex {
    type Output = Complex;
    fn div(self, rhs: Self) -> Self::Output {
        let denominator = rhs.re * rhs.re + rhs.im * rhs.im;
        Self::new((self.re * rhs.re + self.im * rhs.im) / denominator, (self.im * rhs.re - self.re * rhs.im) / denominator)
    }
}

impl std::ops::DivAssign for Complex {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl std::ops::Add<Real> for Complex {
    type Output = Complex;
    fn add(self, rhs: Real) -> Self::Output {
        Self::new(self.re + rhs, self.im)
    }
}

impl std::ops::AddAssign<Real> for Complex {
    fn add_assign(&mut self, rhs: Real) {
        self.re = self.re + rhs;
    }
}

impl std::ops::Sub<Real> for Complex {
    type Output = Complex;
    fn sub(self, rhs: Real) -> Self::Output {
        Self::new(self.re - rhs, self.im)
    }
}

impl std::ops::SubAssign<Real> for Complex {
    fn sub_assign(&mut self, rhs: Real) {
        self.re = self.re - rhs;
    }
}

impl std::ops::Mul<Real> for Complex {
    type Output = Complex;
    fn mul(self, rhs: Real) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl std::ops::MulAssign<Real> for Complex {
    fn mul_assign(&mut self, rhs: Real) {
        self.re = self.re * rhs;
    }
}

impl std::ops::Div<Real> for Complex {
    type Output = Complex;
    fn div(self, rhs: Real) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

impl std::ops::DivAssign<Real> for Complex {
    fn div_assign(&mut self, rhs: Real) {
        self.re = self.re / rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn add_test() {
        let a = Complex::new(2., 4.);
        let b = Complex::new(3., 5.);
        assert_eq!(a + b, Complex::new(a.re + b.re, a.im + b.im))
    }
}