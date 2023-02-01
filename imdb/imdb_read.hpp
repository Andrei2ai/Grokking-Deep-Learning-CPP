#ifndef IMDB_READ_HPP
#define IMDB_READ_HPP

#include <vector>
#include <string>
#include <map>

using vec_str = std::vector<std::string>;
using vec_int = std::vector<int>;

vec_str read_lines(const std::string& fileName);

std::vector<vec_str> get_tokens(const vec_str& vs);

vec_str get_vocab(const std::vector<vec_str>& vvs);

std::map<std::string, int> get_word2index(const vec_str& vs);


#endif // IMDB_READ_HPP
