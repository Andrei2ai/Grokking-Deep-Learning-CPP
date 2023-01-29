#include <NumCpp.hpp>
#include "main.hpp"

static void learning_the_whole_dataset()
{
	nc::NdArray<double> weights {0.5, 0.48, -0.7};

	const double alpha = 0.1;

	const nc::NdArray<double> streetlights {
		{ 1, 0, 1 },
		{ 0, 1, 1 },
		{ 0, 0, 1 },
		{ 1, 1, 1 },
		{ 0, 1, 1 },
		{ 1, 0, 1 }
	};

	const nc::NdArray<double> walk_vs_stop {0, 1, 0, 1, 1, 0};

	for( int iteration = 0; iteration < 40; ++iteration)
	{
		double error_for_all_lights = 0;

		std::cout << "----\n\n";

		for( int row_index  = 0; row_index < walk_vs_stop.size(); ++row_index )
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

static void backpropagation_in_code()
{
	auto relu = [](const nc::NdArray<double>& x){
		return (x > 0.).astype<double>() * x;
	};

	auto relu2deriv = [](const nc::NdArray<double>& output){
		return (output > 0.).astype<double>();
	};

	const double alpha       = 0.2;
	const int    hidden_size = 4;

	const nc::NdArray<double> streetlights {
		{ 1, 0, 1 },
		{ 0, 1, 1 },
		{ 0, 0, 1 },
		{ 1, 1, 1 }
	};

	const nc::NdArray<double> walk_vs_stop = nc::NdArray<double>({1, 1, 0, 0}).transpose();

	nc::random::seed(1);
	static std::mt19937 generator;
	generator.seed(1);

	auto weights_0_1 = 2. * nc::random::rand<double>(nc::Shape(3, hidden_size)) - 1.;
	auto weights_1_2 = 2. * nc::random::rand<double>(nc::Shape(hidden_size, 1)) - 1.;

	PRINT(weights_0_1);
	PRINT(weights_1_2);

	// Random from python np.random.seed(1)
//	nc::NdArray<double> weights_0_1  {
//		{-0.16595599,  0.44064899, -0.99977125, -0.39533485},
//		{-0.70648822, -0.81532281, -0.62747958, -0.30887855},
//		{-0.20646505,  0.07763347, -0.16161097,  0.370439  }
//	};

	// Random from python np.random.seed(1)
//	nc::NdArray<double> weights_1_2 {-0.5910955 , 0.75623487, -0.94522481, 0.34093502};
//	weights_1_2 = weights_1_2.transpose();

	for( int iteration = 0; iteration < 60; ++iteration)
	{
		double layer_2_error = 0;

		for( int i  = 0; i < streetlights.shape().rows; ++i )
		{
			auto layer_0 = streetlights(i, streetlights.cSlice());
			auto layer_1 = relu(nc::dot(layer_0, weights_0_1));
			auto layer_2 = nc::dot(layer_1, weights_1_2);

			layer_2_error += nc::sum((layer_2 - walk_vs_stop[i]) * (layer_2 - walk_vs_stop[i])).item();

			auto layer_2_delta = (walk_vs_stop[i] - layer_2);
			auto layer_1_delta = layer_2_delta.dot(weights_1_2.transpose()) * relu2deriv(layer_1);

			weights_1_2 += alpha * layer_1.transpose().dot(layer_2_delta);
			weights_0_1 += alpha * layer_0.transpose().dot(layer_1_delta);
		}

		if((iteration % 10) == 9)
		{
			PRINT(layer_2_error);
		}
	}
}

int main(int argc, char *argv[])
{
	//learning_the_whole_dataset();

	backpropagation_in_code();

	return 0;
}
