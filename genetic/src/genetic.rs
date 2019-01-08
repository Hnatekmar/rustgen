extern crate rand;
use rand::Rng;

pub trait Genome: Sized + Clone {
    fn cross(&self, other: &Self) -> Self;
    fn mutate(&self) -> Self;
}

 #[derive(Copy, Clone)] 
pub struct GenomeResult<T: Genome> {
    genome: T,
    fitness: f64
}

impl<T: Genome> GenomeResult<T> {
    fn from_genome<F: Fn(&T) -> f64>(gen: T, fit_fn: F) -> Self {
        let fitness_value = fit_fn(&gen);
        GenomeResult {
            genome: gen,
            fitness: fitness_value
        }
    }
}

pub fn genetic<T: Genome, F: Fn(&T) -> f64, Fc: Fn() -> T>(fitness: F, create: Fc, size: usize, iterations: usize) -> T {
    let mut population = Vec::new();
    let mut rng = rand::thread_rng();
    for _ in 0 .. size {
        let created = create();
        population.push(GenomeResult::from_genome(created, &fitness));
    }
    fn roulette_selection<T: Genome>(pop: &Vec<GenomeResult<T>>, fitness_sum: f64) -> &GenomeResult<T> {
        let mut rng = rand::thread_rng();
        let res = rng.gen_range(0.0, fitness_sum + 0.00000001);
        let mut acc: f64 = 0.0;
        let mut i: usize = 0;
        while i < pop.len() {
            acc += pop[i].fitness;
            if acc > res {
                break;
            }
            i += 1;
        }
        &pop[i]
    }
    for i in 1 .. iterations + 1 { 
        population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        println!("{}, {}", i, &population[population.len() - 1].fitness);
        let fitness_sum = population.iter().map(|genome| genome.fitness).sum::<f64>();
        let mut new_population = Vec::new();
        for _ in 0 .. size {
            let a = roulette_selection(&population, fitness_sum);
            let b = roulette_selection(&population, fitness_sum);
            let mut result = a.genome.cross(&b.genome);
            if rng.gen_range(0.0, 1.0) > 0.95 {
                result = result.mutate();
            }
            new_population.push(GenomeResult::from_genome(result, &fitness));
        }
        population = new_population;
    }
    population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    population[population.len() - 1].genome.clone()
}