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

#define _ACTIVATION(name) neural::name##_activation
#define ACTIVATION(name) _ACTIVATION(name)
#define _DEACTIVATION(name) neural::name##_deactivation
#define DEACTIVATION(name) _DEACTIVATION(name)

#define FUN sigm

int main()
{
	neural::perceptron perc({ 3, 23, 1 }, -2.0, 1.0, ACTIVATION(FUN), DEACTIVATION(FUN));
	
	const double lr = 0.1;
	
	while (true)
	{
		for (size_t i = 0; i < 1000; ++i)
		{
			perc.use({ 1, 0, 0 });
			perc.teach({ 1 }, lr);
			
			perc.use({ 0, 1, 0 });
			perc.teach({ 1 }, lr);
			
			perc.use({ 1, 1, 0 });
			perc.teach({ 1 }, lr);
			
			perc.use({ 0, 0, 0 });
			perc.teach({ 0 }, lr);
			
			perc.use({ 1, 0, 1 });
			perc.teach({ 1 }, lr);
			
			perc.use({ 0, 1, 1 });
			perc.teach({ 1 }, lr);
			
			perc.use({ 1, 1, 1 });
			perc.teach({ 0 }, lr);
			
			perc.use({ 0, 0, 1 });
			perc.teach({ 1 }, lr);
		}
		
		neural::print_vector(perc.use({ 1, 0, 0 }));
		neural::print_vector(perc.use({ 0, 1, 0 }));
		neural::print_vector(perc.use({ 1, 1, 0 }));
		neural::print_vector(perc.use({ 0, 0, 0 }));
		neural::print_vector(perc.use({ 1, 0, 1 }));
		neural::print_vector(perc.use({ 0, 1, 1 }));
		neural::print_vector(perc.use({ 1, 1, 1 }));
		neural::print_vector(perc.use({ 0, 0, 1 }));
		
		std::cout << "\n";
		
		//		{
		//			neural::perceptron perc1(perc.get_core_data(), neural::sigm_activation, neural::sigm_deactivation);
		//
		//			neural::print_vector(perc1.use({ 0.5, 0, 0.5 }));
		//			neural::print_vector(perc1.use({ 0, 1, 0 }));
		//			neural::print_vector(perc1.use({ 0.5, 0, 0 }));
		//			neural::print_vector(perc1.use({ 0, 0, 1 }));
		//		}
		//
		//		std::cout << "\n";
		
		int c = getchar();
		//		std::cout << " c = " << c << "\n\n";
	}
	
	return 0;
}
