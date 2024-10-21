pub mod math;
use math::vector::Vector;


fn main() {
    let mut a = Vector::from_real([1., 1.]);
    a = a.normalize();
    println!("{:?}", a * 4.)
}
