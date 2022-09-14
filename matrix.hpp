// Copyright (c) 2022 Perets Dmytro
// Author: Perets Dmytro <imperator999mcpe@gmail.com>
//
// Personal usage is allowed only if this comment was not changed or deleted.
// Commercial usage must be agreed with the author of this comment.


#ifndef PERCEPTRON_MATRIX_HPP
#define PERCEPTRON_MATRIX_HPP

#include <vector>
#include <cassert>


namespace neural
{
	typedef long double neuron_t;
	
	typedef neuron_t (* neural_function)(neuron_t x);
	
	extern std::vector<neuron_t> apply_function(std::vector<neuron_t> vec, neural_function func);
	
	extern std::vector<neuron_t> matrix_multiply_forward(
			const std::vector<std::vector<neuron_t>>& W, const std::vector<neuron_t>& I, const std::vector<neuron_t>& B
	);
	
	extern std::vector<neuron_t> matrix_multiply_backward(const std::vector<std::vector<neuron_t>>& W, const std::vector<neuron_t>& E);
	
	extern std::vector<neuron_t> matrix_multiply_backward(const std::vector<neuron_t>& B, const std::vector<neuron_t>& E);
	
	extern std::vector<std::vector<neuron_t>>& correct_weights(
			std::vector<std::vector<neuron_t>>& W,
			const std::vector<neuron_t>& I, const std::vector<neuron_t>& O, const std::vector<neuron_t>& E,
			neural_function deactivation_function, double alfa
	);
	
	extern std::vector<neuron_t>& correct_biases(
			std::vector<neuron_t>& B, const std::vector<neuron_t>& O, const std::vector<neuron_t>& E,
			neural_function deactivation_function, double alfa
	);
}

#endif //PERCEPTRON_MATRIX_HPP
