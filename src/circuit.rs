use crate::{numbers::Real, state_vector::{rx, ry, rz, StateVector, UGate, CNOT, HADAMARD, PAULI_X, PAULI_Y, PAULI_Z}};


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Qubit(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CBit(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CustomOperation(pub usize);

#[derive(Debug, Clone)]
pub enum Operation {
    Hadamard(Qubit),
    PauliX(Qubit),
    PauliY(Qubit),
    PauliZ(Qubit),
    CNOT(Qubit, Qubit),

    RX(Qubit, Real),
    RY(Qubit, Real),
    RZ(Qubit, Real),

    Measure(Qubit, CBit),

    Custom(Vec<Qubit>, CustomOperation),
    Barrier,
}

pub struct Statistics {
    stats: HashMap<usize, usize>,
    bit_count: usize
}

impl Statistics {
    fn new(bit_count: usize) -> Self {
        Statistics {
            stats: HashMap::new(),
            bit_count
        }
    }
    fn add_result(&mut self, measurement: usize) {
        match self.stats.get(&measurement) {
            Some(count) => { self.stats.insert(measurement, count + 1); }
            None => { self.stats.insert(measurement, 1); }
        }
    }
}

impl std::fmt::Display for Statistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = self.bit_count;
        let formatted_stats: Vec<String> = self.stats.iter().map(|(key, value)| format!("{:0width$b}: {}", key, value)).collect();
        write!(f, "{{{}}}", formatted_stats.join(", "))
    }
}

#[derive(Debug, Clone)]
pub struct Circuit {
    qubit_count: usize,
    bit_count: usize,
    operations: Vec<Operation>,
}

impl Circuit {
    pub fn new(qubit_count: usize, bit_count: usize) -> Self {
        Circuit {
            qubit_count,
            bit_count,
            operations: Vec::new(),
        }
    }
    pub fn run(&self, shots: usize) -> Statistics {
        let mut statistics = Statistics::new(self.qubit_count);
        for _ in 0..shots {
            let mut state = StateVector::zero_state(self.qubit_count);
            let mut measurement: usize = 0;
            for gate in &self.operations {
                match gate {
                    Operation::Hadamard(qubit) => {
                        state.apply_1q_gate(&HADAMARD, qubit.0);
                    },
                    Operation::PauliX(qubit) => {
                        state.apply_1q_gate(&PAULI_X, qubit.0);
                    },
                    Operation::PauliY(qubit) => {
                        state.apply_1q_gate(&PAULI_Y, qubit.0);
                    },
                    Operation::PauliZ(qubit) => {
                        state.apply_1q_gate(&PAULI_Z, qubit.0);
                    },
                    Operation::CNOT(control, target) => {
                        state.apply_2q_gate(&CNOT, control.0, target.0);
                    },
                    Operation::RX(qubit, angle) => {
                        state.apply_1q_gate(&&rx(*angle), qubit.0);
                    },
                    Operation::RY(qubit, angle) => {
                        state.apply_1q_gate(&ry(*angle), qubit.0);
                    },
                    Operation::RZ(qubit, angle) => {
                        state.apply_1q_gate(&&rz(*angle), qubit.0);
                    },
                    Operation::Measure(qubit, bit) => {
                        measurement = state.measure(qubit.0) << bit.0 | !(1 << bit.0) & measurement;
                    }
                    Operation::Barrier => { /* Just skips */ }
                    Operation::Custom(qubits, operation) => {
                        todo!()
                    }
                }
            }
            statistics.add_result(measurement);
        }
        statistics
    }
    pub fn add_gate(&mut self, gate: Operation) {
        self.operations.push(gate);
    }
    pub fn h(&mut self, qubit: Qubit) {
        self.add_gate(Operation::Hadamard(qubit));
    }
    pub fn x(&mut self, qubit: Qubit) {
        self.add_gate(Operation::PauliX(qubit));
    }
    pub fn y(&mut self, qubit: Qubit) {
        self.add_gate(Operation::PauliY(qubit));
    }
    pub fn z(&mut self, qubit: Qubit) {
        self.add_gate(Operation::PauliZ(qubit));
    }
    pub fn cx(&mut self, qubit0: Qubit, qubit1: Qubit) {
        self.add_gate(Operation::CNOT(qubit0, qubit1));
    }
    pub fn rx(&mut self, qubit: Qubit, angle: Real) {
        self.add_gate(Operation::RX(qubit, angle));
    }
    pub fn ry(&mut self, qubit: Qubit, angle: Real) {
        self.add_gate(Operation::RY(qubit, angle));
    }
    pub fn rz(&mut self, qubit: Qubit, angle: Real) {
        self.add_gate(Operation::RZ(qubit, angle));
    }
    pub fn measure(&mut self, qubit: Qubit, bit: CBit) {
        self.add_gate(Operation::Measure(qubit, bit));
    }
    pub fn barrier(&mut self) {
        self.add_gate(Operation::Barrier);
    }
}

use std::{collections::{btree_map::Values, HashMap}, fmt::{self, write}};

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Circuit with {} qubits and {} classical bits:", self.qubit_count, self.bit_count)?;
        let mut strings = CircuitString::new(self.qubit_count, self.bit_count);

        for gate in &self.operations {
            match gate {
                Operation::Hadamard(qubit) => {
                    strings.push_to_qubit("H", *qubit);
                },
                Operation::PauliX(qubit) => {
                    strings.push_to_qubit("X", *qubit);
                },
                Operation::PauliY(qubit) => {
                    strings.push_to_qubit("Y", *qubit);
                },
                Operation::PauliZ(qubit) => {
                    strings.push_to_qubit("Z", *qubit);
                },
                Operation::CNOT(control, target) => {
                    strings.pad_qubit_interval(*control, *target);
                    strings.push_to_qubit("●", *control);
                    strings.push_to_qubit("X", *target);
                    strings.push_line(*control, *target);
                },
                Operation::RX(qubit, angle) => {
                    strings.push_to_qubit(&format!("RX({:.2})", angle), *qubit);
                },
                Operation::RY(qubit, angle) => {
                    strings.push_to_qubit(&format!("RY({:.2})", angle), *qubit);
                },
                Operation::RZ(qubit, angle) => {
                    strings.push_to_qubit(&format!("RZ({:.2})", angle), *qubit);
                },
                Operation::Measure(qubit, bit) => {
                    strings.pad_strings(); // TODO: pad only needed lines
                    strings.push_bit_line(*qubit, *bit);
                    strings.push_to_qubit("M", *qubit);
                    strings.push_to_bit("╚", *bit);
                }
                Operation::Barrier => {
                    strings.pad_strings();
                    strings.push_for_each("|");
                }
                _ => todo!(),
            }
            
        }
        strings.pad_strings();
        write!(f, "{}", strings)
    }
}

struct CircuitString {
    qubits: Vec<String>,
    bits: Vec<String>,
}

impl CircuitString {
    const QWIRE: char = '─';
    const CWIRE: char = '═';
    const QQCONNECTOR: char = '┼';
    const CQCONNECTOR: char = '╫';
    const CCCONNECTOR: char = '║';
    fn new(qubit_count: usize, bit_count: usize) -> Self {
        let mut qubits = vec![String::new(); qubit_count];
        qubits.iter_mut().enumerate().for_each(|(i, line)| line.push_str(&format!("q{i}{}", Self::QWIRE)));
        let mut bits = vec![String::new(); bit_count];
        bits.iter_mut().enumerate().for_each(|(i, line)| line.push_str(&format!("c{i}{}", Self::CWIRE)));

        Self {
            qubits, bits
        }
    }
    fn push_to_qubit(&mut self, string: &str, qubit: Qubit) {
        self.qubits[qubit.0].push_str(&format!("{string}{}", Self::QWIRE));
    }
    fn push_line(&mut self, qubit0: Qubit, qubit1: Qubit) {
        let (high, low) = if qubit0.0 > qubit1.0 {(qubit0.0, qubit1.0)} else {(qubit1.0, qubit0.0)};
        for i in low + 1..high {
            self.push_to_qubit(&format!("{}", Self::QQCONNECTOR), Qubit(i));
        }
    }
    fn push_bit_line(&mut self, qubit: Qubit, bit: CBit) {
        for i in qubit.0 + 1..self.qubits.len() {
            self.push_to_qubit(&format!("{}", Self::CQCONNECTOR), Qubit(i));
        }
        for i in 0..bit.0 {
            self.push_to_bit(&format!("{}", Self::CCCONNECTOR), CBit(i));
        }
    }
    fn push_to_bit(&mut self, string: &str, bit: CBit) {
        self.bits[bit.0].push_str(&format!("{string}{}", Self::CWIRE));
    }
    fn pad_qubit_interval(&mut self, qubit0: Qubit, qubit1: Qubit) {
        let (high, low) = if qubit0.0 > qubit1.0 {(qubit0.0 + 1, qubit1.0)} else {(qubit1.0 + 1, qubit0.0)};
        let max_len = self.qubits[low..high].iter().map(|s| s.chars().count()).max().unwrap_or(0);
        for string in &mut self.qubits[low..high] {
            let char_count = string.chars().count();
            if max_len > char_count {
                let padding: String = std::iter::repeat(Self::QWIRE).take(max_len - char_count).collect();
                string.push_str(&padding);
            }
        }
    }
    fn pad_strings(&mut self) {
        let max_len = std::cmp::max(self.qubits.iter().map(|s| s.chars().count()).max().unwrap_or(0), self.bits.iter().map(|s| s.chars().count()).max().unwrap_or(0));
        for string in &mut self.qubits {
            let char_count = string.chars().count();
            if max_len > char_count {
                let padding: String = std::iter::repeat(Self::QWIRE).take(max_len - char_count).collect();
                string.push_str(&padding);
            }
        }
        for string in &mut self.bits {
            let char_count = string.chars().count();
            if max_len > char_count {
                let padding: String = std::iter::repeat(Self::CWIRE).take(max_len - char_count).collect();
                string.push_str(&padding);
            }
        }
    }
    fn push_for_each(&mut self, string: &str) {
        self.qubits.iter_mut().for_each(|line| line.push_str(&format!("{string}{}", Self::QWIRE)));
        self.bits.iter_mut().for_each(|line| line.push_str(&format!("{string}{}", Self::CWIRE)));
    }
}

impl std::fmt::Display for CircuitString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for line in &self.qubits {
            writeln!(f, "{line}")?;
        }
        for line in &self.bits {
            writeln!(f, "{line}")?;
        }
        Ok(())
    }
}

