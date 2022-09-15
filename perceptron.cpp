// Copyright (c) 2022 Perets Dmytro
// Author: Perets Dmytro <imperator999mcpe@gmail.com>
//
// Personal usage is allowed only if this comment was not changed or deleted.
// Commercial usage must be agreed with the author of this comment.


#include "perceptron.hpp"


namespace neural
{
	perceptron::perceptron(
			const std::vector<size_t>& sizes, neuron_t min, neuron_t max,
			neural_function activation_function, neural_function deactivation_function
	) : activation_function(activation_function), deactivation_function(deactivation_function)
	{
		assert(sizes.size() > 0);
		
		inputs_size = *sizes.begin();
		
		size_t prev = inputs_size;
		for (auto it = sizes.begin() + 1; it != sizes.end(); ++it)
		{
			layers.emplace_back(prev, *it, min, max);
			prev = *it;
		}
	}
	
	perceptron::perceptron(const std::vector<struct core_data>& data, neural_function activation_function, neural_function deactivation_function)
			: activation_function(activation_function), deactivation_function(deactivation_function)
	{
		assert(!data.empty());
		
		inputs_size = data[0].weights.size();
		
		for (auto& d : data)
			layers.emplace_back(d.biases, d.weights);
	}
	
	std::vector<neuron_t> perceptron::use(std::vector<neuron_t>&& inputs)
	{
		assert(inputs.size() == inputs_size);
		
		auto&& inp = inputs;
		for (auto& l : layers)
			inp = l.feed_forward(inp, activation_function);
		last_output = inp;
		return last_output;
	}
	
	void perceptron::teach(const std::vector<neuron_t>& samples, double learning_rate)
	{
		assert(samples.size() == layers.back().biases.size());
		
		auto it = layers.end() - 1;
		auto&& errors = it->compute_errors(samples);
		errors = (it--)->back_propagate(errors, deactivation_function, learning_rate);
		for (; it != layers.begin() - 1; --it)
			errors = it->back_propagate(errors, deactivation_function, learning_rate);
	}
	
	std::vector<struct core_data> perceptron::get_core_data()
	{
		std::vector<struct core_data> data;
		
		for (auto& l : layers)
			data.emplace_back(std::move(l.get_core_data()));
		
		return std::move(data);
	}
}