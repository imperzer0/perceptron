// Copyright (c) 2022 Perets Dmytro
// Author: Perets Dmytro <imperator999mcpe@gmail.com>
//
// Personal usage is allowed only if this comment was not changed or deleted.
// Commercial usage must be agreed with the author of this comment.


#include "layer.hpp"


namespace neural
{
	layer::layer(decltype(biases) biases, decltype(weights) weights)
			: biases(std::move(biases)), weights(std::move(weights)) { }
	
	layer::layer(size_t prev_layer, size_t layer, neuron_t min, neuron_t max)
			: biases(layer, 0), weights(prev_layer, std::vector<neuron_t>(layer))
	{
		assert(min < max);
		
		srandom(time(nullptr) + clock());
		
		neuron_t K = UINT32_MAX / (max - min);
		
		for (auto& b : biases)
			b = ((neuron_t)random() / K) + min;
		
		for (auto& wp : weights)
			for (auto& wc : wp)
				wc = ((neuron_t)random() / K) + min;
	}
	
	std::vector<neuron_t> layer::feed_forward(const std::vector<neuron_t>& inputs, neural_function activation_function)
	{
		last_inputs = inputs;
		last_outputs = std::move(apply_function(std::move(matrix_multiply_forward(weights, inputs, biases)), activation_function));
		return last_outputs;
	}
	
	std::vector<neuron_t> layer::back_propagate(const std::vector<neuron_t>& errors, neural_function deactivation_function, double alfa)
	{
		auto err_next = matrix_multiply_backward(weights, errors);
		auto err_biases = matrix_multiply_backward(biases, errors);
		
		correct_weights(weights, last_inputs, last_outputs, errors, deactivation_function, alfa);
		correct_biases(biases, last_outputs, errors, deactivation_function, alfa);
		
		return std::move(err_next);
	}
	
	std::vector<neuron_t> layer::compute_errors(const std::vector<neuron_t>& samples)
	{
		assert(samples.size() == last_outputs.size());
		
		std::vector<neuron_t> res = last_outputs;
		
		for (size_t i = 0; i < res.size(); ++i)
			res[i] = samples[i] - res[i];
		
		return std::move(res);
	}
	
	core_data layer::get_core_data() { return core_data{ .biases = biases, .weights = weights }; }
}