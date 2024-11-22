pub mod numbers;
pub mod state_vector;
pub mod circuit;
use circuit::{CBit, Circuit, Qubit};
use numbers::RealConsts::PI;

fn main() {
    let mut qc = Circuit::new(2, 2);
    qc.h(Qubit(0));
    qc.cx(Qubit(0), Qubit(1));
    qc.measure(Qubit(0), CBit(0));
    qc.measure(Qubit(1), CBit(1));
    println!("{qc}");
    let stats = qc.run(1000);
    println!("{stats}");
}
