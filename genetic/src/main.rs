extern crate num_traits;
extern crate rand;
use rand::Rng;
mod genetic;
use genetic::{Genome, genetic};
use std::f64::consts::E;

#[derive(Clone)] 
enum Activation {
    Linear,
    Sigmoid,
    RELU
}

#[derive(Clone)] 
struct Neuron {
    bias: f64,
    weights: Vec<f64>,
    activation: Activation
}

impl Neuron {
    fn propagate(&self, x: &Vec<f64>) -> Option<f64> {
        if x.len() != self.weights.len() {
            None
        } else {
            Some(x.iter().zip(&self.weights).map(|xw| xw.0 * xw.1).sum::<f64>() + self.bias)
        }
    }
    fn activate(&self, x: &Vec<f64>) -> f64 {
        let propagate = self.propagate(x).unwrap();
        match self.activation {
            Activation::Linear => propagate,
            Activation::Sigmoid => 1.0 / (1.0 + E.powf(-propagate)),
            Activation::RELU => if propagate > 0.0f64 { propagate } else { 0.0 }
        }
    }

    fn random_activation() -> Activation {
        let mut rng = rand::thread_rng();
        match rng.gen_range(0, 3) {
            0 => Activation::Linear,
            1 => Activation::Sigmoid,
            _ => Activation::RELU
        }
    }
    
    fn mutate(&mut self) -> () {
        let mut rng = rand::thread_rng();
        match rng.gen_range(0, 3) {
            0 => {
                let i = rng.gen_range(0, self.weights.len());
                self.weights[i] += rng.gen_range(-1.0, 1.0);
            },
            1 => {
                self.bias += rng.gen_range(-1.0, 1.0);
            },
            _ => {
                self.activation = Neuron::random_activation();
            }
        }
    }

    fn create_random(input_size: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let hidden_layer:Vec<f64> = (0 .. input_size).map(|_| rng.gen_range(-1.0, 1.0)).collect();
        Neuron {
            bias: rng.gen_range(-1.0, 1.0),
            weights: hidden_layer,
            activation: Neuron::random_activation()
        }
    }
}

type Layer = Vec<Neuron>;
#[derive(Clone)] 
struct NeuralNetwork {
    layers: Vec<Layer>
}

impl NeuralNetwork {
    fn new(input_size: usize, hidden_layer_size: usize, number_of_hidden_layers: usize, output_size: usize) -> NeuralNetwork {
        let mut network = NeuralNetwork {
            layers: Vec::new()
        };
        let mut first_layer: Layer = Vec::new();
        for _ in 0 .. hidden_layer_size {
            first_layer.push(Neuron::create_random(input_size));
        }
        network.layers.push(first_layer);
        for _ in 0 .. number_of_hidden_layers - 1 {
            let mut layer: Layer = Vec::new();
            for _ in 0 .. hidden_layer_size {
                layer.push(Neuron::create_random(hidden_layer_size));
            }
            network.layers.push(layer);
        }
        let mut output_layer: Layer = Vec::new();
        for _ in 0 .. output_size {
            output_layer.push(Neuron::create_random(hidden_layer_size));
        }
        network.layers.push(output_layer);
        network
    }

    fn propagate(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut it = x.clone();
        for layer in &self.layers {
            it = layer.iter().map(|neuron| neuron.activate(&it)).collect::<Vec<f64>>();
        }
        it
    }
}

impl Genome for NeuralNetwork {
    fn cross(&self, other: &Self) -> Self {
        let mut res: Vec<Layer> = Vec::new();
        for i in 0 .. self.layers.len() {
            if i % 2 == 0 {
                res.push(self.layers[i].clone());
            } else {
                res.push(other.layers[i].clone());
            }
        } 
        NeuralNetwork {
            layers: res
        }
    }
    fn mutate(&self) -> Self {
        let mut rng = rand::thread_rng();
        let mut copy = self.clone();
        let i = rng.gen_range(0, copy.layers.len());
        let genome_index = rng.gen_range(0, copy.layers[i].len());
        copy.layers[i][genome_index].mutate();
        copy
    }
}

fn create_random_network() -> NeuralNetwork {
    NeuralNetwork::new(2, 6, 2, 1)
}

fn fitness(neural_network: &NeuralNetwork) -> f64 {
    let inputs: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0]
    ];
    let outputs = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0]
    ];
    let mut acc = 0.0f64;
    for i in 0 .. inputs.len() {
        let prediction = neural_network.propagate(&inputs[i]);
        acc += (prediction[0] - outputs[i][0]).powf(2.0f64);
    }
    100000f64 - acc / (outputs.len() as f64)
}

fn main() {
    genetic(fitness, create_random_network, 10000, 20000);
}