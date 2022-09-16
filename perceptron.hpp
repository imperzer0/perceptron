// Copyright (c) 2022 Perets Dmytro
// Author: Perets Dmytro <imperator999mcpe@gmail.com>
//
// Personal usage is allowed only if this comment was not changed or deleted.
// Commercial usage must be agreed with the author of this comment.


#ifndef PERCEPTRON_PERCEPTRON_HPP
#define PERCEPTRON_PERCEPTRON_HPP

#include "layer.hpp"


namespace neural
{
	static constexpr neuron_t sigm_activation(neuron_t x)
	{
		return 1 / (1 + std::exp(-x));
	}
	
	static constexpr neuron_t sigm_deactivation(neuron_t x)
	{
		return x * (1 - x);
	}
	
	static constexpr neuron_t sigm_scaled_activation(neuron_t x)
	{
		return 3 / (2 * (1 + std::exp(-x))) - 0.5;
	}
	
	static constexpr neuron_t sigm_scaled_deactivation(neuron_t x)
	{
		return sigm_scaled_activation(x) * (1 - sigm_scaled_activation(x));
	}
	
	static constexpr neuron_t tanh_activation(neuron_t x)
	{
		return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
	}
	
	static constexpr neuron_t tanh_deactivation(neuron_t x)
	{
		neuron_t tmp = tanh_activation(x);
		return 1 - tmp * tmp;
	}
	
	static constexpr neuron_t swish_activation(neuron_t x)
	{
		return x * sigm_activation(x);
	}
	
	static constexpr neuron_t swish_deactivation(neuron_t x)
	{
		return sigm_activation(x) + x * sigm_activation(x) * (1 - sigm_activation(x));
	}
	
	class perceptron
	{
		std::vector<layer> layers;
		size_t inputs_size;
		
		neural_function activation_function;
		neural_function deactivation_function;
		
		/// cache
		std::vector<neuron_t> last_output;
	
	public:
		perceptron(
				const std::vector<size_t>& sizes, neuron_t min, neuron_t max,
				neural_function activation_function, neural_function deactivation_function
		);
		
		perceptron(const std::vector<struct core_data>& data, neural_function activation_function, neural_function deactivation_function);
		
		std::vector<neuron_t> use(std::vector<neuron_t>&& inputs);
		
		void teach(const std::vector<neuron_t>& samples, double learning_rate);
		
		std::vector<struct core_data> get_core_data();
	};
}

#endif //PERCEPTRON_PERCEPTRON_HPP
