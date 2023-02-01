#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include "imdb_read.hpp"


std::vector<std::string> read_lines(const std::string& fileName)
{
	auto p = std::filesystem::current_path();
	p.append(fileName);
	std::cout << "Current path is " << p <<std::endl;

	std::ifstream r(p);
	if (r.is_open() == false)
	{
		std::perror("ifstream");
		throw std::runtime_error(fileName + " cannot open");
	}

	std::vector<std::string> raw {};
	std::string temp;
	while (std::getline(r, temp))
	{
		raw.push_back(temp);
	}

	return raw;
}

std::vector<vec_str> get_tokens(const vec_str &vs)
{
	std::vector<vec_str> tokens {};
	for(auto &&str: vs)
	{
		std::set<std::string> tokenset {};
		std::stringstream ss {str};
		std::string s {};
		while (std::getline(ss, s, ' '))
		{
			tokenset.insert(s);
		}
		vec_str item(tokenset.size());
		std::copy(tokenset.cbegin(), tokenset.cend(), item.begin());
		tokens.push_back(item);
	}
	return tokens;
}

vec_str get_vocab(const std::vector<vec_str> &vvs)
{
	std::set<std::string> vocabset {};
	for(auto &&vs: vvs)
	{
		for(auto &&s: vs)
		{
			if(s.length() > 0 )
			{
				vocabset.insert(s);
			}
		}
	}

	vec_str vocab(vocabset.size());
	std::copy(vocabset.cbegin(), vocabset.cend(), vocab.begin());

	return vocab;
}

std::map<std::string, int> get_word2index(const vec_str &vs)
{
	std::map<std::string, int> word2index {};
	for(int i = 0; i < vs.size(); ++i)
	{
		word2index[vs[i]] = i;
	}

	return word2index;
}
