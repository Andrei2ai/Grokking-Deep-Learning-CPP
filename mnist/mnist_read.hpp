#ifndef MNIST_READ_HPP
#define MNIST_READ_HPP

#include <cstdint>
#include <fstream>
#include <array>
#include <vector>


static constexpr int num_labels       {10};
static constexpr int pixels_per_image {28 * 28};


using imgArray = std::array<uint8_t, pixels_per_image>;

uint8_t read8(std::ifstream &f);

uint32_t read32(std::ifstream &f); // MSB first

std::vector<imgArray> readImg(std::ifstream &f);

std::vector<uint8_t> readLabel(std::ifstream &f);

std::ifstream openFile(const std::string& fileName);

std::vector<imgArray> readTrainImg();

std::vector<imgArray> readTestImg();

std::vector<uint8_t> readTrainLabel();

std::vector<uint8_t> readTestLabel();


#endif // MNIST_READ_HPP
