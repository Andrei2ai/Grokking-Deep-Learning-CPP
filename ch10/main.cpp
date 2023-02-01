#include <cassert>
#include <vector>
#include <NumCpp.hpp>
#include "../mnist/mnist_read.hpp"
#include "main.hpp"

static constexpr int train_count = 1000;
static constexpr int test_count  = 10'000;


static void upgrading_our_MNIST_network(
        const nc::NdArray<double>& images,
        const nc::NdArray<double>& labels,
        const nc::NdArray<double>& test_images,
        const nc::NdArray<double>& test_labels)
{
	nc::random::seed(1);

	auto tanh2deriv = [](const nc::NdArray<double>& output){
		return 1.0 - (output * output);
	};

	const int batch_size = 128;
	const double alpha = 2;
	const int iterations = 300;

	const int input_rows = 28;
	const int input_cols = 28;

	const int kernel_rows = 3;
	const int kernel_cols = 3;
	const int num_kernels = 16;

	const int hidden_size = ((input_rows - kernel_rows) * (input_cols - kernel_cols)) * num_kernels;

	//auto weights_0_1 = 0.02 * nc::random::rand<double>(nc::Shape(pixels_per_image,hidden_size)) - 0.01;
	auto kernels = 0.02 * nc::random::rand<double>(nc::Shape(kernel_rows*kernel_cols, num_kernels)) - 0.01;
	auto weights_1_2 = 0.2 * nc::random::rand<double>(nc::Shape(hidden_size,num_labels)) - 0.1;

	auto get_image_section = [](nc::NdArray<double>& layer, int row_from, int row_to, int col_from, int col_to) {
		for(int i = 0; i < layer.shape().rows; ++i)
		{
			auto tmpImg = layer(i, layer.cSlice()).reshape(28, 28);
			auto tmp = tmpImg(nc::Slice(row_from, row_to), nc::Slice(col_from, col_to));

		}
		return ;
	};

	for( int j = 0; j < iterations; ++j)
	{
		int correct_cnt = 0.;

		for( int i = 0; i < images.shape().rows / batch_size; ++i)
		{
			auto batch_start = i * batch_size;
			auto batch_end = (i+1) * batch_size;

			auto layer_0_tmp = images(nc::Slice(batch_start, batch_end), images.cSlice());
			std::vector<nc::NdArray<double>> layer_0;
			for(int v = 0; v < layer_0_tmp.shape().rows; ++v)
			{
				layer_0.push_back(layer_0_tmp(v, layer_0_tmp.cSlice()).reshape(input_rows, input_cols));
			}
			std::vector<nc::NdArray<double>> sects;
			for(int row_start  = 0; row_start < (input_rows - kernel_rows); ++row_start )
			{
				for(int col_start = 0; col_start  < (input_cols - kernel_cols); ++col_start  )
				{
					auto sect = get_image_section(layer_0, row_start, row_start+kernel_rows, col_start, col_start+kernel_cols);
				}
			}


		}

		int test_correct_cnt = 0;

		for( int i = 0; i < test_images.shape().rows; ++i)
		{
			auto layer_0 = test_images(i, test_images.cSlice());
			auto layer_1 = nc::tanh(nc::dot(layer_0, weights_0_1));
			auto layer_2 = nc::dot(layer_1, weights_1_2);

			test_correct_cnt += (layer_2.argmax() == test_labels(i, test_labels.cSlice()).argmax()).astype<int>().item();
		}

		if((j % 10) == 0)
		{
			std::cout << "I: " << j
			          << ", Test-Acc: " << (double)test_correct_cnt / test_images.shape().rows
			          << ", Train-Acc: " << (double)correct_cnt / images.shape().rows << std::endl;
		}
	}
}

int main(int argc, char *argv[])
{
	auto images = nc::zeros<double>(train_count, pixels_per_image);
	{
		auto x_train = readTrainImg();
		assert(train_count <= x_train.size());
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
		assert(train_count <= y_train.size());
		for(int i = 0 ; i < train_count; ++i)
		{
			labels(i, y_train[i]) = 1;
		}
	}

	auto test_images = nc::zeros<double>(test_count, pixels_per_image);
	{
		auto x_test = readTestImg();
		assert(train_count <= x_test.size());
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
		assert(train_count <= y_test.size());
		for(int i = 0 ; i < test_count; ++i)
		{
			test_labels(i, y_test[i]) = 1;
		}
	}

	upgrading_our_MNIST_network(images, labels, test_images, test_labels);


	return 0;
}
