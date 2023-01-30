#include <cassert>
#include "mnist_read.hpp"

uint8_t read8(std::ifstream &f)
{
	char ch = 0;
	f.read(&ch, sizeof(ch));
	return ch;
}

uint32_t read32(std::ifstream &f) // MSB first
{
	uint32_t res = 0;

	res |= read8(f) << 24;
	res |= read8(f) << 16;
	res |= read8(f) << 8;
	res |= read8(f);

	return res;
}

std::vector<imgArray> readImg(std::ifstream &f)
{
	const uint32_t mn = read32(f);
	assert(mn == 2051);
	const uint32_t count = read32(f);
	const uint32_t rows = read32(f);
	const uint32_t cols = read32(f);

	std::vector<imgArray> v;
	v.reserve(count);

	for( int i = 0; i < count; ++i)
	{
		imgArray img {};
		auto it = img.begin();

		for( int r = 0; r < rows; ++r)
		{
			for( int c = 0; c < cols; ++c)
			{
				*it = read8(f);
				++it;
			}
		}
		v.push_back(img);
	}

	return v;
}

std::vector<uint8_t> readLabel(std::ifstream &f)
{
	const uint32_t mn = read32(f);
	assert(mn == 2049);
	const uint32_t count = read32(f);

	std::vector<uint8_t> v;
	v.reserve(count);

	for(int i = 0; i < count; ++i)
	{
		v.push_back(read8(f));
	}

	return v;
}

std::ifstream openFile(const std::string &fileName)
{
	std::ifstream f(fileName, std::ios::binary);
	if (f.is_open() == false)
	{
		throw std::runtime_error(fileName + " cannot open");
	}
	return f;
}

std::vector<imgArray> readTrainImg()
{
	std::ifstream x = openFile("train-images.idx3-ubyte");
	return readImg(x);
}

std::vector<imgArray> readTestImg()
{
	std::ifstream x  = openFile("t10k-images.idx3-ubyte");
	return readImg(x);
}

std::vector<uint8_t> readTrainLabel()
{
	std::ifstream y = openFile("train-labels.idx1-ubyte");
	return readLabel(y);
}

std::vector<uint8_t> readTestLabel()
{
	std::ifstream y = openFile("t10k-labels.idx1-ubyte");
	return readLabel(y);
}

