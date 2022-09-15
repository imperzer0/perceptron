#include <iostream>
#include "perceptron.hpp"


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
	neural::perceptron perc({ 3, 3, 2 }, -1.0, 0.0, neural::sigm_activation, neural::sigm_deactivation);
	
	const double lr = 0.1;
	
	while (true)
	{
		for (size_t i = 0; i < 1000; ++i)
		{
			perc.use({ 0.5, 0, 0.5 });
			perc.teach({ 0, 1 }, lr);
			
			perc.use({ 0, 1, 0 });
			perc.teach({ 1, 1 }, lr);
			
			perc.use({ 0.5, 0, 0 });
			perc.teach({ 0, 0.5 }, lr);
			
			perc.use({ 0, 0, 1 });
			perc.teach({ 0, 1 }, lr);
		}
		
		neural::print_vector(perc.use({ 0.5, 0, 0.5 }));
		neural::print_vector(perc.use({ 0, 1, 0 }));
		neural::print_vector(perc.use({ 0.5, 0, 0 }));
		neural::print_vector(perc.use({ 0, 0, 1 }));
		
		std::cout << "\n";
		
		{
			neural::perceptron perc1(perc.get_core_data(), neural::sigm_activation, neural::sigm_deactivation);
			
			neural::print_vector(perc1.use({ 0.5, 0, 0.5 }));
			neural::print_vector(perc1.use({ 0, 1, 0 }));
			neural::print_vector(perc1.use({ 0.5, 0, 0 }));
			neural::print_vector(perc1.use({ 0, 0, 1 }));
		}
		
		std::cout << "\n";
		
		int c = getchar();
		//		std::cout << " c = " << c << "\n\n";
	}
	
	return 0;
}
