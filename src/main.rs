pub mod math;
use math::state_vector::{StateVector, CNOT, H};

fn main() {
    let mut a = StateVector::zero_state(2);
    
    a.apply_1q_gate(&H, 1);
    a.apply_2q_gate(&CNOT, 0, 1);
    a.dirac();
}
