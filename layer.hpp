// Copyright (c) 2022 Perets Dmytro
// Author: Perets Dmytro <imperator999mcpe@gmail.com>
//
// Personal usage is allowed only if this comment was not changed or deleted.
// Commercial usage must be agreed with the author of this comment.


#ifndef PERCEPTRON_LAYER_HPP
#define PERCEPTRON_LAYER_HPP

#include "matrix.hpp"

#include <vector>
#include <cstddef>
#include <random>
#include <ctime>
#include <climits>
#include <cassert>


namespace neural
{
	struct core_data
	{
		std::vector<neuron_t> biases;
		std::vector<std::vector<neuron_t >> weights;
	};
	
	class layer
	{
		friend class perceptron;
		
		std::vector<neuron_t> biases;
		std::vector<std::vector<neuron_t >> weights;
		
		/// cache
		std::vector<neuron_t> last_inputs;
		std::vector<neuron_t> last_outputs;
	
	public:
		layer(decltype(biases) biases, decltype(weights) weights);
		
		layer(size_t prev_layer, size_t layer, neuron_t min, neuron_t max);
		
		std::vector<neuron_t> feed_forward(const std::vector<neuron_t>& inputs, neural_function activation_function);
		
		std::vector<neuron_t> back_propagate(const std::vector<neuron_t>& errors, neural_function deactivation_function, double alfa);
		
		std::vector<neuron_t> compute_errors(const std::vector<neuron_t>& samples);
		
		core_data get_core_data();
	};
}

#endif //PERCEPTRON_LAYER_HPP
