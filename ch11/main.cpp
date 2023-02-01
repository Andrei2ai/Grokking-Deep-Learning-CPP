#include <cmath>
#include <NumCpp.hpp>
#include "../imdb/imdb_read.hpp"
#include "main.hpp"

static auto raw_reviews = read_lines("reviews.txt");

static auto raw_labels = read_lines("labels.txt");

static auto tokens = get_tokens(raw_reviews);

static auto vocab = get_vocab(tokens);

static auto word2index = get_word2index(vocab);

static nc::NdArray<double> weights_0_1 {};

static nc::NdArray<double> weights_1_2 {};



static void mtx(const std::vector<nc::NdArray<int>>& input_dataset, const nc::NdArray<int>& target_dataset)
{
	nc::random::seed(1);

	auto sigmoid = [](const nc::NdArray<double>& x) {
		return 1.0/(1.0 + nc::exp(-x));
	};

	const double alpha = 0.01;
	const int iterations = 2;
	const int hidden_size = 100;

	weights_0_1 = 0.2 * nc::random::rand<double>(nc::Shape(vocab.size(),hidden_size)) - 0.1;
	weights_1_2 = 0.2 * nc::random::rand<double>(nc::Shape(hidden_size,1)) - 0.1;

	double correct = 0;
	double total = 0;

	for( int iter = 0; iter < iterations; ++iter)
	{
		for(int i = 0; i < (input_dataset.size() - 1000); ++i)
		{
			auto x = input_dataset[i];
			auto y = target_dataset[i];
			auto layer_1_tmp = nc::zeros<double>(x.size(), hidden_size);
			for( int r = 0; r < x.size(); ++r)
			{
				for( int c = 0; c < hidden_size; ++c)
				{
					layer_1_tmp(r, c) = weights_0_1(x[r], c);
				}
			}
			auto layer_1 = sigmoid(nc::sum(layer_1_tmp, nc::Axis::ROW));
			auto layer_2 = sigmoid(nc::dot(layer_1, weights_1_2));

			auto layer_2_delta = layer_2 - static_cast<double>(y);
			auto layer_1_delta = layer_2_delta.dot(weights_1_2.transpose());

			for( int r = 0; r < x.size(); ++r)
			{
				for( int c = 0; c < hidden_size; ++c)
				{
					weights_0_1(x[r], c) -= layer_1_delta(0, c) * alpha;
				}
			}
			weights_1_2 -= (layer_1 * layer_2_delta.item()).transpose() * alpha;

			if(nc::abs(layer_2_delta).item() < 0.5)
			{
				correct +=1;
			}

			total += 1;

			if( (i % 10) == 9 )
			{
				auto progress = (100.0 * i)/(input_dataset.size() - 1000);
				std::cout << "Iter: " << iter
				          << "Progress: " << progress
				          << "% Training Accuracy: " << correct/total << std::endl;
			}
		}
	}

	correct = 0;
	total = 0;
	for(int i = 0; i < (input_dataset.size() - 1000); ++i)
	{
		auto x = input_dataset[i];
		auto y = target_dataset[i];
		auto layer_1_tmp = nc::zeros<double>(x.size(), hidden_size);
		for( int r = 0; r < x.size(); ++r)
		{
			for( int c = 0; c < hidden_size; ++c)
			{
				layer_1_tmp(r, c) = weights_0_1(x[r], c);
			}
		}
		auto layer_1 = sigmoid(nc::sum(layer_1_tmp, nc::Axis::ROW));
		auto layer_2 = sigmoid(nc::dot(layer_1, weights_1_2));

		if(nc::abs(layer_2 - static_cast<double>(y)).item() < 0.5)
		{
			correct +=1;
		}

		total += 1;
	}
	std::cout << "Test Accuracy: " << correct / total << std::endl;
}

static void comparing_word_embeddings()
{
	auto similar = [](const std::string& target) {
		auto target_index = word2index[target];
		std::map<std::string, double> scores {};
		for( auto&& word: word2index)
		{
			auto raw_difference = weights_0_1(word.second, weights_0_1.cSlice()) - weights_0_1(target_index, weights_0_1.cSlice());
			auto squared_difference = raw_difference * raw_difference;
			scores[word.first] = -std::sqrt(nc::sum(squared_difference).item());
		}

		std::set<std::pair<double, std::string>> s;  // The new (temporary) container.

		for (auto const &kv : scores)
		{
			s.emplace(kv.second, kv.first);  // Flip the pairs.
		}

		auto it = s.crbegin();
		for (int j = 0; j < 15; ++j)
		{
			std::cout << it->first << " " << it->second << std::endl;
			++it;
		}

		std::cout << "...." << std::endl;
	};

	similar("beautiful");

	similar("terrible");
}


int main(int argc, char *argv[])
{
	std::vector<nc::NdArray<int>> input_dataset {};
	for(auto &&t: tokens)
	{
		std::set<int> sent_indices {};
		for(auto &&s: t)
		{
			if( word2index.find(s) != word2index.end() )
			{
				sent_indices.insert(word2index[s]);
			}
		}
		auto item = nc::zeros<int>(1, sent_indices.size());
		std::copy(sent_indices.cbegin(), sent_indices.cend(), item.begin());
		input_dataset.push_back(item);
	}

	auto target_dataset = nc::NdArray<int>(1, raw_labels.size());
	for(int i = 0; i < raw_labels.size(); ++i)
	{
		target_dataset(0, i) = static_cast<int>((raw_labels[i].find("positive") != std::string::npos));
	}

	mtx(input_dataset, target_dataset);

	std::cout << "========" << std::endl;

	comparing_word_embeddings();

	return 0;
}
