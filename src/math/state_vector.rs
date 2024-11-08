use crate::math::{complex::Complex, Real, RealConsts};

#[derive(Debug, Clone, PartialEq)]
pub struct StateVector {
    elems: Vec<Complex>,
}

impl StateVector {
    pub fn new(elems: Vec<Complex>) -> Self {
        assert!(elems.len().is_power_of_two() && elems.len() > 1, "Vector length must be a power of 2 and greater than 1.");
        let mut state = StateVector { elems };
        state.normalize_mut();
        state
    }
    pub fn zero_state(qubit_count: usize) -> Self {
        assert!(qubit_count > 0, "Amount of qubits must be positive.");
        let size = 1 << qubit_count;
        let mut elems = vec![Complex::ZERO; size];
        elems[0] = Complex::ONE;
        
        StateVector { elems }
    }
    pub fn from_real(elems: Vec<Real>) -> Self {
        StateVector::new(elems.iter().map(|e| Complex::from_real(*e)).collect())
    }
    fn normalize_mut(&mut self) {
        let rhs = self.magnitude();
        for n in 0..self.elems.len() {
            self.elems[n] /= rhs;
        }
    }
    fn magnitude(&self) -> Real {
        self.magnitude_squared().sqrt()
    }
    fn magnitude_squared(&self) -> Real {
        let mut result = 0.0;
        for i in 0..self.elems.len() {
            result += self.elems[i].abs_squared();
        }
        result
    }
    pub fn apply_1q_gate(&mut self, gate: &UGate<2>, qubit_index: usize) -> &mut Self {
        let qubit_mask = 1 << qubit_index;
        let upper_mask = !qubit_mask ^ (qubit_mask - 1);
        let lower_mask = qubit_mask - 1;
        for i in 0..self.elems.len() / 2 {
            let j0 = (i & lower_mask) | ((i << 1) & upper_mask);
            let j1 = j0 | qubit_mask;

            let elem0 = self.elems[j0];
            let elem1 = self.elems[j1];

            self.elems[j0] = gate.elems[0][0] * elem0 + gate.elems[0][1] * elem1;
            self.elems[j1] = gate.elems[1][0] * elem0 + gate.elems[1][1] * elem1;
        }
        self
    }
    pub fn apply_2q_gate(&mut self, gate: &UGate<4>, qubit0: usize, qubit1: usize) -> &mut Self {
        assert_ne!(qubit0, qubit1);
        let qubit0_mask = 1 << qubit0;
        let qubit1_mask = 1 << qubit1;
        let (ms_mask, ls_mask) = if qubit0 > qubit1 {
            (qubit0_mask, qubit1_mask)
        }
        else {
            (qubit1_mask, qubit0_mask)
        };
        let upper_mask = !ms_mask ^ (ms_mask - 1);
        let lower_mask = ls_mask - 1;
        let middle_mask = (!ls_mask ^ (ls_mask - 1)) & (ms_mask - 1);
        // println!("Masks:\n{:064b}\n{:064b}\n{:064b}\n", upper_mask, middle_mask, lower_mask);

        for i in 0..self.elems.len() / 4 {
            let j00 = (i & lower_mask) | ((i << 1) & middle_mask) | ((i << 2) & upper_mask);
            let j01 = j00 | qubit0_mask;
            let j10 = j00 | qubit1_mask;
            let j11 = j01 | j10;
            // println!("Indexes:\n{:064b}\n{:064b}\n{:064b}\n{:064b}\n", j00, j01, j10, j11);
            let elem00 = self.elems[j00];
            let elem01 = self.elems[j01];
            let elem10 = self.elems[j10];
            let elem11 = self.elems[j11];
            self.elems[j00] = gate.elems[0][0] * elem00 + gate.elems[0][1] * elem01 + gate.elems[0][2] * elem10 + gate.elems[0][3] * elem11;
            self.elems[j01] = gate.elems[1][0] * elem00 + gate.elems[1][1] * elem01 + gate.elems[1][2] * elem10 + gate.elems[1][3] * elem11;
            self.elems[j10] = gate.elems[2][0] * elem00 + gate.elems[2][1] * elem01 + gate.elems[2][2] * elem10 + gate.elems[2][3] * elem11;
            self.elems[j11] = gate.elems[3][0] * elem00 + gate.elems[3][1] * elem01 + gate.elems[3][2] * elem10 + gate.elems[3][3] * elem11;
        }
        self
    }
    pub fn qubit_count(&self) -> usize {
        self.elems.len().ilog2() as usize
    }
    pub fn dirac(&self) {
        let qubit_count = self.qubit_count();
        let output = self.elems.iter()
            .enumerate()
            .filter_map(|(i, e)| {
                if !e.is_approx_equal(&Complex::ZERO) {
                    Some(format!("{}|{:0qubit_count$b}>", e, i))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" + ");
        
        println!("{}", output);
    }    
    pub fn is_approx_equal(&self, other: &Self) -> bool {
        // todo: Handle invarience of global phase
        if self.elems.len() != other.elems.len() {
            false
        }
        else {
            for i in 0..self.elems.len() {
                if !self.elems[i].is_approx_equal(&other.elems[i]) {
                    return false;
                }
            }
            true
        }
    }
    pub fn tensor(&self, other: &Self) -> Self {
        todo!()
    }
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub struct UGate<const N: usize> {
    elems: [[Complex; N]; N]
}

impl<const N: usize> UGate<N> {
    pub fn new(elems: [[Complex; N]; N]) -> Self {
        let matrix = UGate { elems };
        assert!((matrix * matrix.complex_conj()).is_approx_equal(&UGate::identity()));
        assert!((matrix.complex_conj() * matrix).is_approx_equal(&UGate::<N>::identity()));
        matrix
    }
    const fn const_new(elems: [[Complex; N]; N]) -> Self {
        UGate { elems }
    }
    pub const fn const_new_from_real(elems: [[Real; N]; N]) -> Self {
        let mut new_elems: [[Complex; N]; N] = [[Complex::from_real(0.0); N]; N];
        
        let mut i = 0;
        while i < N {
            let mut j = 0;
            while j < N {
                new_elems[i][j] = Complex::from_real(elems[i][j]);
                j += 1;
            }
            i += 1;
        }

        UGate::const_new(new_elems)
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

pub const H: UGate<2> = UGate::const_new_from_real([
    [1.0 / RealConsts::SQRT_2, 1.0 / RealConsts::SQRT_2],
    [1.0 / RealConsts::SQRT_2, -1.0 / RealConsts::SQRT_2],
]);

pub const CNOT: UGate<4> = UGate::const_new_from_real([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
]);

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
    #[test]
    fn complex_conjugate() {
        let a = UGate::new([
            [Complex::ZERO, Complex::new(0.0, -1.0)],
            [Complex::new(0.0, 1.0), Complex::ZERO],
        ]);
        assert!((a * a.complex_conj()).is_approx_equal(&UGate::identity()))
    }
    #[test]
    fn apply_identity() {
        let qubits = 5;
        let state = StateVector::zero_state(qubits);
        let mut modified = state.clone();
        let identity = UGate::<2>::identity();
        for i in 0..qubits {
            modified.apply_1q_gate(&identity, i);
        }
        assert!(state.is_approx_equal(&modified))
    }
}