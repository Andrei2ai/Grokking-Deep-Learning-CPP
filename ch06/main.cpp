#include <NumCpp.hpp>
#include "main.hpp"

static void learning_the_whole_dataset()
{
	nc::NdArray<double> weights {0.5, 0.48, -0.7};

	const double alpha = 0.1;

	const nc::NdArray<double> streetlights {{ 1, 0, 1 },
		                                    { 0, 1, 1 },
		                                    { 0, 0, 1 },
		                                    { 1, 1, 1 },
		                                    { 0, 1, 1 },
		                                    { 1, 0, 1 }};

	const nc::NdArray<double> walk_vs_stop  { 0,
		                                      1,
		                                      0,
		                                      1,
		                                      1,
		                                      0};

	for( int i = 0; i < 40; ++i)
	{
		double error_for_all_lights = 0;

		std::cout << "----\n\n";

		for( int row_index  = 0; row_index  < walk_vs_stop.size(); ++row_index )
		{
			auto input = streetlights(row_index, streetlights.cSlice());
			auto goal_prediction = walk_vs_stop[row_index];

			auto prediction = input.dot(weights);
			auto error = (goal_prediction - prediction) * (goal_prediction - prediction);

			error_for_all_lights += error.item();

			auto delta = (prediction - goal_prediction).item();
			weights = weights - (alpha * (input * delta));
			PRINT(prediction);
		}

		PRINT(error_for_all_lights);
	}
}

int main(int argc, char *argv[])
{
	learning_the_whole_dataset();

	return 0;
}
