pub mod math;
use math::vector::{Matrix, ColVector, RowVector, ColVectorOperations};
use math::state_vector::StateVector;

fn main() {
    let a = StateVector::from_real([1.0, 1.0]);
    println!("{:?}", a);
}
