#include <vector>
#include <cmath>
#include <cassert>
#include <iterator>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <stdexcept>
#include <stdint.h>
#include <time.h>

typedef float value_t;
typedef unsigned char cluster_t;

struct point
{
	point(std::vector<value_t>& coords) : coords(coords), cluster(0) {}
	std::vector<value_t> coords;
	cluster_t cluster;
};

typedef std::vector<point> data_t;
typedef std::vector<point> means_t;

typedef double distance_t;

/*void generate(std::size_t count, unsigned int seed, data_t& data)
{
	srand(seed);
	data.reserve(count);
	while (count--)
	{
		data.push_back(point(rand(), rand()));

	}
}*/

inline value_t power2(value_t x)
{
	return x*x;
}

inline value_t distance(const point& a, const point& b)
{
	value_t value = 0;;
	for (size_t i = 0; i < a.coords.size(); i++)
	{
		value += power2(b.coords[i] - a.coords[i]);
	}
	return value;
}

void assign_to_clusters(data_t& data, const means_t& means)
{
	assert(means.size()>0);
	for (data_t::iterator it = data.begin(); it != data.end(); ++it)
	{
		cluster_t closest_cluster(0);
		value_t mindist(distance(*it, means[0]));
		for (means_t::const_iterator mit = means.begin() + 1; mit != means.end(); ++mit)
		{
			value_t dist = distance(*it, *mit);
			if (dist<mindist) { closest_cluster = static_cast<cluster_t>(std::distance(means.begin(), mit)); mindist = dist; }
		}
		it->cluster = closest_cluster;
	}
}

void compute_means(const data_t& data, means_t& means)
{
	std::vector<std::size_t> counts(means.size(), 0);
	for (means_t::iterator mit = means.begin(); mit != means.end(); ++mit)
	{
		for (size_t i = 0; i < mit->coords.size(); i++)
		{
			mit->coords = std::vector<value_t>(mit->coords.size(),0);
		}
	}
	for (data_t::const_iterator it = data.begin(); it != data.end(); ++it)
	{
		++counts[it->cluster];
		for (size_t i = 0; i < it->coords.size(); i++)
		{
			means[it->cluster].coords[i] += it->coords[i];
		}
	}
	for (means_t::iterator mit = means.begin(); mit != means.end(); ++mit)
	{
		for (size_t i = 0; i < mit->coords.size(); i++)
		{
			mit->coords[i] /= counts[std::distance(means.begin(), mit)];
		}
	}
}

void save(const std::string& file_name, const data_t& data)
{
	FILE* f = fopen(file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	for (data_t::const_iterator it = data.begin(); it != data.end(); ++it)
	{
		if (!fwrite(&it->coords, sizeof(value_t), it->coords.size(), f)) throw std::runtime_error("value cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");
}

void load(const std::string& file_name, data_t& data)
{
	FILE* f = fopen(file_name.c_str(), "rb");
	if (!f) throw std::runtime_error("cannot open file for reading");
	//if (fseek(f, 0, SEEK_END)) throw std::runtime_error("seeking failed");
	uint64_t count = 0, dimension = 0;
	if (!fread(&count, sizeof(uint64_t), 1, f))  throw std::runtime_error("size cannot be read");
	if (!fread(&dimension, sizeof(uint64_t), 1, f))  throw std::runtime_error("dimension cannot be read");
	while (count--)
	{
		std::vector<value_t> coords(dimension);
		if (!fread(&coords[0], sizeof(value_t), dimension, f))  throw std::runtime_error("value cannot be read");
		data.push_back(point(coords));
	}
}

template<typename T>
T lexical_cast(const std::string& x)
{
	std::istringstream stream(x);
	T res;
	stream >> res;
	return res;
}

void usage()
{
	std::cout << "Usage:" << std::endl << "kmeans <data_file> <means_file> <clusters_file> <k> <iterations>" << std::endl << "kmeans --generate <data_file> <size> <seed>" << std::endl;
}

void save_results(const std::string& means_file_name, const std::string& clusters_file_name, const means_t& means, const data_t& data)
{
	FILE* f = fopen(means_file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	for (means_t::const_iterator it = means.begin(); it != means.end(); ++it)
	{
		if (!fwrite(&it->coords, sizeof(value_t), it->coords.size(), f)) throw std::runtime_error("value cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");


	f = fopen(clusters_file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	for (data_t::const_iterator it = data.begin(); it != data.end(); ++it)
	{
		if (!fwrite(&it->cluster, sizeof(cluster_t), 1, f)) throw std::runtime_error("value cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");
}

int main(int argc, const char* argv[])
{
	if (argc == 5)
	{
		std::string option(argv[1]);
		if (option != "--generate") { usage(); return 1; }
		std::string file_name(argv[2]);
		std::string s_size(argv[3]);
		std::string s_seed(argv[4]);
		std::size_t size = lexical_cast<std::size_t>(s_size);
		std::size_t seed = lexical_cast<std::size_t>(s_seed);
		data_t data;
		//generate(size, seed, data);
		save(file_name, data);
		return 0;
	}

	if (argc == 6)
	{
		std::string file_name(argv[1]);
		std::string means_file_name(argv[2]);
		std::string clusters_file_name(argv[3]);
		std::string s_k(argv[4]);
		std::string s_iterations(argv[5]);
		std::size_t k = lexical_cast<std::size_t>(s_k);
		std::size_t iterations = lexical_cast<std::size_t>(s_iterations);

		data_t data;
		load(file_name, data);

		assert(data.size() >= k);

		data_t means(data.begin(), data.begin() + k);

		clock_t start, end;
		start = clock();

		while (iterations--)
		{
			assign_to_clusters(data, means);
			compute_means(data, means);
		}

		end = clock();
		std::cout << "Time required for execution: "
		<< (double)(end-start)/CLOCKS_PER_SEC
		<< " seconds." << "\n\n";

		save_results(means_file_name, clusters_file_name, means, data);
		return 0;
	}
	usage();
	return 1;
}