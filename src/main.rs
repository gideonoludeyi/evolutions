use std::collections::HashMap;

use rand::{seq::SliceRandom, Rng};
mod ga {
    use rand::Rng;
    pub trait PopulationInitializer<I> {
        fn initialize(&self, rng: &mut impl Rng) -> Vec<I>;
    }

    pub trait Crossover<I> {
        fn cross(&self, p1: &I, p2: &I, rng: &mut impl Rng) -> (I, I);
    }

    pub trait Mutation<I> {
        fn mutate(&self, individual: &I, rng: &mut impl Rng) -> I;
    }

    pub trait Selection<I> {
        fn select(&self, pop: &Vec<I>, rng: &mut impl Rng) -> Vec<I>;
    }

    pub fn genetic_algorithm<I>(
        initializer: impl PopulationInitializer<I>,
        selection: impl Selection<I>,
        crossover: impl Crossover<I>,
        mutation: impl Mutation<I>,
        mut rng: impl Rng,
    ) -> Vec<I> {
        let mut pop = initializer.initialize(&mut rng);
        for _ in 1..=100 {
            pop = selection.select(&pop, &mut rng);
            pop = {
                let mut offspring: Vec<I> = Vec::with_capacity(pop.len());
                let pop_iter = pop.chunks_exact(2);
                let remainder = pop_iter.remainder();
                for individuals in pop_iter {
                    let (child1, child2) =
                        crossover.cross(&individuals[0], &individuals[1], &mut rng);
                    offspring.push(child1);
                    offspring.push(child2);
                }
                // because the chunk size is 2, the remainder must contain only 1 element if not empty
                if !remainder.is_empty() {
                    let (child1, child2) =
                        crossover.cross(&remainder[0], &pop.get(0).unwrap(), &mut rng);
                    offspring.push(child1);
                    offspring.push(child2);
                }
                offspring
            };
            pop = pop
                .iter()
                .map(|individual| mutation.mutate(individual, &mut rng))
                .collect();
        }
        pop
    }
}

#[derive(Debug)]
struct VigenerePopulationInitializer {
    population_size: usize,
    chromosome_length: usize,
    alleles: String,
}
impl VigenerePopulationInitializer {
    fn new(population_size: usize, chromosome_length: usize, alleles: String) -> Self {
        Self {
            population_size,
            chromosome_length,
            alleles,
        }
    }
}
impl ga::PopulationInitializer<String> for VigenerePopulationInitializer {
    fn initialize(&self, rng: &mut impl Rng) -> Vec<String> {
        let mut pop = Vec::with_capacity(self.population_size);
        for _ in 0..self.population_size {
            let s = String::from_utf8(
                self.alleles
                    .as_bytes()
                    .choose_multiple(rng, self.chromosome_length)
                    .map(|x| *x)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            pop.push(s);
        }
        pop
    }
}

#[derive(Debug)]
struct UniformCrossover {
    crossover_rate: f64,
}
impl UniformCrossover {
    fn new(crossover_rate: f64) -> Self {
        Self { crossover_rate }
    }
}
impl ga::Crossover<String> for UniformCrossover {
    fn cross(&self, p1: &String, p2: &String, rng: &mut impl Rng) -> (String, String) {
        if rng.gen_bool(self.crossover_rate) {
            let mut c1 = p1.as_bytes().to_owned();
            let mut c2 = p2.as_bytes().to_owned();
            for i in 0..c1.len() {
                if rng.gen_bool(
                    0.5, /* 50/50 probability of swapping -- equivalent to using a bit mask */
                ) {
                    (c1[i], c2[i]) = (c2[i], c1[i]);
                }
            }
            (
                String::from_utf8(c1).unwrap(),
                String::from_utf8(c2).unwrap(),
            )
        } else {
            (p1.to_owned(), p2.to_owned())
        }
    }
}

#[derive(Debug)]
struct RandomCharacterMutation {
    mutation_rate: f64,
    alleles: String,
}
impl RandomCharacterMutation {
    fn new(mutation_rate: f64, alleles: String) -> Self {
        Self {
            mutation_rate,
            alleles,
        }
    }
}
impl ga::Mutation<String> for RandomCharacterMutation {
    fn mutate(&self, individual: &String, rng: &mut impl Rng) -> String {
        let codes = self.alleles.as_bytes();
        let vec = individual
            .as_bytes()
            .iter()
            .map(|code| {
                if rng.gen_bool(self.mutation_rate) {
                    *codes.choose(rng).unwrap()
                } else {
                    *code
                }
            })
            .collect();
        String::from_utf8(vec).unwrap()
    }
}

trait FitnessEvaluator<I> {
    fn evaluate(&self, individual: &I) -> f64;
}

struct TournamentSelection<I, F: FitnessEvaluator<I>> {
    _i: std::marker::PhantomData<I>,
    k: usize,
    evaluator: Box<F>,
}
impl<I, F: FitnessEvaluator<I>> TournamentSelection<I, F> {
    fn new(k: usize, evaluator: F) -> Self {
        Self {
            _i: std::marker::PhantomData,
            k,
            evaluator: Box::new(evaluator),
        }
    }
}
impl<F: FitnessEvaluator<String>> ga::Selection<String> for TournamentSelection<String, F> {
    fn select(&self, pop: &Vec<String>, rng: &mut impl Rng) -> Vec<String> {
        let mut newpop = Vec::with_capacity(pop.len());
        for _ in 0..pop.len() {
            let winner = pop
                .choose_multiple(rng, self.k)
                .max_by(|ind1, ind2| {
                    self.evaluator
                        .evaluate(ind1)
                        .partial_cmp(&self.evaluator.evaluate(ind2))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            newpop.push(winner.to_owned());
        }
        newpop
    }
}

#[derive(Clone, Debug)]
struct Cipher(String);
impl Cipher {
    fn decrypt(&self, key: &String) -> Vec<u8> {
        let cipher: Vec<u8> = self
            .0
            .to_lowercase()
            .replace(|c: char| !c.is_alphabetic(), "")
            .into();
        let key: Vec<u8> = key
            .to_lowercase()
            .replace(|c: char| !c.is_alphabetic(), "")
            .into();
        let mut plain = vec![];
        let mut key_ptr = 0_usize;
        for c in cipher {
            let mut key_char = 0;
            if !key.is_empty() {
                while key[key_ptr] < 97 || key[key_ptr] > (25 + 97) {
                    key_ptr = (key_ptr + 1) % key.len();
                }
                key_char = key[key_ptr];
                key_ptr = (key_ptr + 1) % key.len();
            }
            plain.push((26 + c - key_char) % 26 + 97);
        }
        plain
    }
}

#[derive(Clone, Debug)]
struct ExpectedCharFrequencyEvaluator {
    cipher: Cipher,
    expected_freqdist: HashMap<String, f64>,
}
impl ExpectedCharFrequencyEvaluator {
    fn new(cipher: String, corpus: &str) -> Self {
        Self {
            cipher: Cipher(cipher),
            expected_freqdist: Self::ngram_freq(corpus, 2),
        }
    }
    fn ngram_freq(text: &str, n: usize) -> HashMap<String, f64> {
        let length = text.len();
        let mut freq: HashMap<String, f64> = HashMap::new();
        for i in 0..=length - n {
            let s = &text[i..i + n];
            freq.insert(s.to_owned(), freq.get(s).unwrap_or(&0_f64) + 1.0);
        }
        for (_, count) in freq.iter_mut() {
            *count /= length as f64;
        }
        freq
    }
    fn cmpfreqdists(
        expected_freqdist: &HashMap<String, f64>,
        actual_freqdist: &HashMap<String, f64>,
    ) -> f64 {
        actual_freqdist
            .iter()
            .map(|(token, freq)| (freq - expected_freqdist.get(token).unwrap_or(&0.0)).abs())
            .sum()
    }
}
impl FitnessEvaluator<String> for ExpectedCharFrequencyEvaluator {
    fn evaluate(&self, individual: &String) -> f64 {
        let plain = self.cipher.decrypt(individual);
        let freq = Self::ngram_freq(&String::from_utf8(plain).unwrap(), 2);
        let fit = Self::cmpfreqdists(&self.expected_freqdist, &freq);
        1.0 - fit
    }
}

fn main() {
    let rng = rand::thread_rng();

    let alleles = "abcdefghijklmnopqrstuvwxyz-".to_owned();
    let cipher_text = "xbwdesmhihslwhkktefvktkktcwfpiibihwmosfilojvooegvefwnochsuuspsureifakbnlalzsrsroiejwzgfpjczldokrceoahzshpbdwpcjstacgbarfwifwohylckafckzwwomlalghrtafchfetcgfpfrgxclwzocdctmjebx".to_owned();
    let corpus = &std::fs::read_to_string("tmp/corpus.txt").unwrap(); // text obtained from http://corpus-db.org/docs
    let fitfn = ExpectedCharFrequencyEvaluator::new(cipher_text, corpus);
    let results = ga::genetic_algorithm(
        VigenerePopulationInitializer::new(100, 8, alleles.clone()),
        TournamentSelection::new(3, fitfn.clone()),
        UniformCrossover::new(0.9),
        RandomCharacterMutation::new(0.1, alleles.clone()),
        rng,
    );

    println!("{results:#?}");
    let fitnesses = results
        .iter()
        .map(|ind| fitfn.evaluate(ind))
        .collect::<Vec<_>>();
    println!("{fitnesses:#?}");
}
