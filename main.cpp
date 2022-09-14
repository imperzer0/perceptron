#include <iostream>
#include "perceptron.hpp"


void print_vector(const std::vector<neural::neuron_t>& vec, std::ostream& ostream = std::cout)
{
	ostream << "( ";
	for (auto& v : vec)
		ostream << v << " ";
	ostream << ")\n";
}

std::vector<neural::neuron_t> read_vector(size_t size, std::istream& istream = std::cin)
{
	decltype(read_vector(size)) vec;
	neural::neuron_t x;
	for (size_t i = 0; i < size; ++i)
	{
		istream >> x;
		vec.push_back(x);
	}
	return std::move(vec);
}


int main()
{
	neural::perceptron perc({ 2, 3, 1 }, -1.0, 1.0, neural::sigm_activation, neural::sigm_deactivation);
	
	while (true)
	{
		for (size_t i = 0; i < 1000; ++i)
		{
			perc.use({ 1, 0 });
			perc.teach({ 1 }, 0.1);
			
			perc.use({ 1, 1 });
			perc.teach({ 0 }, 0.1);
			
			perc.use({ 0, 1 });
			perc.teach({ 1 }, 0.1);
			
			perc.use({ 0, 0 });
			perc.teach({ 0 }, 0.1);
		}
		
		int c = getchar();
		
		print_vector(perc.use({ 1, 0 }));
		print_vector(perc.use({ 1, 1 }));
		print_vector(perc.use({ 0, 1 }));
		print_vector(perc.use({ 0, 0 }));
		//		std::cout << " c = " << c << "\n\n";
	}
	
	return 0;
}
