// Copyright (c) 2022 Perets Dmytro
// Author: Perets Dmytro <imperator999mcpe@gmail.com>
//
// Personal usage is allowed only if this comment was not changed or deleted.
// Commercial usage must be agreed with the author of this comment.


#include <cstddef>
#include "matrix.hpp"


namespace neural
{
	std::vector<neuron_t> apply_function(std::vector<neuron_t> vec, neural_function func)
	{
		for (auto& v : vec)
			v = func(v);
		return std::move(vec);
	}
	
	/// W * I
	/// 2x3 * 3x1 = 2x1
	///  ij  ij  ij    i         j
	/// w11 w21 w31   i1   b1   r1: w11*i1 + w21*i2 + w31*i3 + b1
	/// w12 w22 w32 x i2 + b2 = r2: w12*i1 + w22*i2 + w32*i3 + b2
	///               i3
	std::vector<neuron_t> matrix_multiply_forward(
			const std::vector<std::vector<neuron_t>>& W, const std::vector<neuron_t>& I, const std::vector<neuron_t>& B
	)
	{
		assert(W.size() == I.size());
		assert(!W.empty());
		
		std::vector<neuron_t> res(W[0].size(), 0);
		
		for (size_t i = 0; i < W.size(); ++i)
			for (size_t j = 0; j < W[i].size(); ++j)
				res[j] += I[i] * W[i][j];
		
		assert(res.size() == B.size());
		
		for (size_t j = 0; j < res.size(); ++j)
			res[j] += B[j];
		
		return std::move(res);
	}
	
	
	/// W(T) * I
	/// 3x2 * 2x1 = 3x1
	///  ij  ij    j    i
	/// w11 w12   e1   r1: w11*e1 + w12*e2
	/// w21 w22 x e2 = r2: w21*e1 + w22*e2
	/// w31 w32        r3: w31*e1 + w32*e2
	std::vector<neuron_t> matrix_multiply_backward(const std::vector<std::vector<neuron_t>>& W, const std::vector<neuron_t>& E)
	{
		std::vector<neuron_t> res(W.size(), 0);
		
		for (size_t i = 0; i < W.size(); ++i)
		{
			assert(W[i].size() == E.size());
			for (size_t j = 0; j < W[i].size(); ++j)
				res[i] += E[j] * W[i][j];
		}
		
		return std::move(res);
	}
	
	/// W(T) * I
	/// 3x2 * 2x1 = 3x1
	///  i          i
	///            e1
	/// b1 b2 b3 x e2 = r1: b1*e1 + b2*e2 + b3*e3
	///            e3
	std::vector<neuron_t> matrix_multiply_backward(const std::vector<neuron_t>& B, const std::vector<neuron_t>& E)
	{
		assert(B.size() == E.size());
		std::vector<neuron_t> res(B.size(), 0);
		
		for (size_t j = 0; j < B.size(); ++j)
			res[j] += E[j] * B[j];
		
		return std::move(res);
	}
	
	std::vector<std::vector<neuron_t>>& correct_weights(
			std::vector<std::vector<neuron_t>>& W,
			const std::vector<neuron_t>& I, const std::vector<neuron_t>& O, const std::vector<neuron_t>& E,
			neural_function deactivation_function, double alfa
	)
	{
		assert(!W.empty());
		assert(I.size() == W.size());
		assert(O.size() == W[0].size());
		assert(E.size() == W[0].size());
		
		for (size_t i = 0; i < W.size(); ++i)
			for (size_t j = 0; j < W[i].size(); ++j)
				W[i][j] += alfa * E[j] * deactivation_function(O[j]) * I[i];
		
		return W;
	}
	
	std::vector<neuron_t>& correct_biases(
			std::vector<neuron_t>& B, const std::vector<neuron_t>& O, const std::vector<neuron_t>& E,
			neural_function deactivation_function, double alfa
	)
	{
		assert(O.size() == B.size());
		assert(E.size() == B.size());
		
		for (size_t j = 0; j < B.size(); ++j)
			B[j] += alfa * E[j] * deactivation_function(O[j]);
		
		return B;
	}
}