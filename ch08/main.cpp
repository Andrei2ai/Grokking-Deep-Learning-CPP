#include <NumCpp.hpp>
#include "mnist_read.hpp"
#include "main.hpp"

static constexpr int train_count = 1000;
static constexpr int test_count  = 10'000;

static void layer_network_on_MNIST(
        const nc::NdArray<double>& images,
        const nc::NdArray<double>& labels,
        const nc::NdArray<double>& test_images,
        const nc::NdArray<double>& test_labels)
{
	nc::random::seed(1);

	auto relu = [](const nc::NdArray<double>& x){
		return (x > 0.).astype<double>() * x;
	};

	auto relu2deriv = [](const nc::NdArray<double>& output){
		return (output > 0.).astype<double>();
	};

	const double alpha = 0.005;
	const int iterations = 350;
	const int hidden_size = 40;

	auto weights_0_1 = 0.2 * nc::random::rand<double>(nc::Shape(pixels_per_image,hidden_size)) - 0.1;
	auto weights_1_2 = 0.2 * nc::random::rand<double>(nc::Shape(hidden_size,num_labels)) - 0.1;


	for( int j = 0; j < iterations; ++j)
	{
		double error = 0.;
		int correct_cnt = 0.;

		for( int i = 0; i < images.shape().rows; ++i)
		{
			auto layer_0 = images(i, images.cSlice());
			auto layer_1 = relu(nc::dot(layer_0, weights_0_1));
			auto layer_2 = nc::dot(layer_1, weights_1_2);

			error += nc::sum((labels(i, labels.cSlice()) - layer_2) * (labels(i, labels.cSlice()) - layer_2)).item();
			correct_cnt += (layer_2.argmax() == labels(i, labels.cSlice()).argmax()).astype<int>().item();


			auto layer_2_delta = (labels(i, labels.cSlice()) - layer_2);
			auto layer_1_delta = layer_2_delta.dot(weights_1_2.transpose()) * relu2deriv(layer_1);

			weights_1_2 += alpha * layer_1.transpose().dot(layer_2_delta);
			weights_0_1 += alpha * layer_0.transpose().dot(layer_1_delta);
		}


		if(((j % 10) == 0) || (j == (iterations-1)))
		{
			std::cout << "I: " << j
			          << ", Train-Err: " << error / images.shape().rows
			          << ", Train-Acc: " << (double)correct_cnt / images.shape().rows;//  << std::endl;

			error = 0.;
			correct_cnt = 0.;

			for( int i = 0; i < test_images.shape().rows; ++i)
			{
				auto layer_0 = test_images(i, test_images.cSlice());
				auto layer_1 = relu(nc::dot(layer_0, weights_0_1));
				auto layer_2 = nc::dot(layer_1, weights_1_2);

				error += nc::sum((test_labels(i, test_labels.cSlice()) - layer_2) * (test_labels(i, test_labels.cSlice()) - layer_2)).item();
				correct_cnt += (layer_2.argmax() == test_labels(i, test_labels.cSlice()).argmax()).astype<int>().item();
			}

			std::cout << ", Test-Err: " << error / test_images.shape().rows
			          << ", Test-Acc: " << (double)correct_cnt / test_images.shape().rows  << std::endl;
		}
	}

}


int main(int argc, char *argv[])
{
	auto images = nc::zeros<double>(train_count, pixels_per_image);
	{
		auto x_train = readTrainImg();
		for( int i = 0; i < train_count; ++i)
		{
			for( int j = 0; j < x_train[i].size(); ++j)
			{
				images(i, j) = x_train[i].at(j) / 255.0;
			}
		}
	}

	auto labels = nc::zeros<double>(train_count, num_labels);
	{
		auto y_train = readTrainLabel();
		for(int i = 0 ; i < train_count; ++i)
		{
			labels(i, y_train[i]) = 1;
		}
	}


	auto test_images = nc::zeros<double>(test_count, pixels_per_image);
	{
		auto x_test = readTestImg();
		for( int i = 0; i < test_count; ++i)
		{
			for( int j = 0; j < x_test[i].size(); ++j)
			{
				test_images(i, j) = x_test[i].at(j) / 255.0;
			}
		}
	}

	auto test_labels = nc::zeros<double>(test_count, num_labels);
	{
		auto y_test = readTestLabel();
		for(int i = 0 ; i < test_count; ++i)
		{
			test_labels(i, y_test[i]) = 1;
		}
	}

	layer_network_on_MNIST(images, labels, test_images, test_labels);

	return 0;
}
