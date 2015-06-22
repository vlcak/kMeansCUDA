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
uint64_t dimension;

struct point
{
	point() : cluster(0)
	{
		coords = new value_t[dimension];//(value_t*) malloc(dimension * sizeof(value_t));
	}

	~point()
	{
		delete[] coords;
	}
	//std::vector<value_t> coords;
	value_t* coords;
	//uint16_t dimensions;
	cluster_t cluster;
	value_t distanceFromCluster;
};

struct mean
{
	mean(point* p) : count(0)
	{
		coords = new value_t[dimension]; //(value_t*) _aligned_malloc(dimension * sizeof(value_t), 16);
		memcpy(coords, p->coords, dimension * sizeof(value_t));
	}
	mean() : count(0)
	{
		coords = new value_t[dimension]; //(value_t*) _aligned_malloc(dimension * sizeof(value_t), 16);
		memset(coords, 0, dimension * sizeof(value_t));
	}

	~mean()
	{
		delete[] coords;
	}

	value_t* coords;
	value_t count;
};

typedef std::vector<point*> data_t;
typedef std::vector<mean*> means_t;
typedef std::vector<std::size_t> counts_t;
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

inline value_t distance(const point* a, const mean* b)
{
	value_t sub;
	value_t totalSum = 0;
	for (size_t i = 0; i < dimension; ++i)
	{
		sub = b->coords[i] - a->coords[i];
		totalSum += sub * sub;
	}
	return totalSum;
}

inline value_t distance(means_t::const_iterator a, const point* b)
{
	
	value_t sub;
	value_t totalSum = 0;
	for (size_t i = 0; i < dimension; ++i)
	{
		sub = b->coords[i] - (*a)->coords[i];
		totalSum += sub * sub;
	}
	return totalSum;
}

void assign_to_clusters(data_t& data, const means_t& means)
{
	assert(means.size()>0);

	for (data_t::iterator di = data.begin(); di != data.end(); ++di)
	{
		cluster_t closest_cluster(0);
		value_t mindist(distance((*di), means[0]));
		for (means_t::const_iterator mi = means.cbegin(); mi != means.cend() ; ++mi)
		{
			value_t dist = distance((*di), (*mi));
			if (dist < mindist)
			{
				closest_cluster = std::distance(means.cbegin(), mi);
				mindist = dist;
			}
		}
		(*di)->distanceFromCluster = mindist;
		(*di)->cluster = closest_cluster;
	}
}

void compute_means(const data_t& data, means_t& means)
{
	std::vector<std::size_t> counts(means.size(), 0);
	for (means_t::iterator mi = means.begin(); mi != means.end() ; ++mi)
	{
		memset((*mi)->coords, 0, dimension * sizeof(value_t));
		(*mi)->count = 0;
	}
	for (data_t::const_iterator di = data.cbegin(); di != data.cend(); ++di)
	{
		++means[(*di)->cluster]->count;
		value_t *pm = means[(*di)->cluster]->coords;
		value_t *pd = (*di)->coords;
		for (size_t j = 0; j < dimension; ++j)
		{
			*pm += *pd;
			++pm; ++pd;
		}
	}
	for (means_t::iterator mi = means.begin(); mi != means.end() ; ++mi)
	{
		value_t *pm = (*mi)->coords;
		for (int j = 0; j < dimension; ++j)
		{
			*pm /= (*mi)->count;
			++pm;
		}
	}
}

void save(const std::string& file_name, const data_t& data)
{
	FILE* f = fopen(file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	for (data_t::const_iterator it = data.begin(); it != data.end(); ++it)
	{
		if (!fwrite(&((*it)->coords), sizeof(value_t), dimension, f)) throw std::runtime_error("value cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");
}

void load(const std::string& file_name, data_t& data)
{
	FILE* f = fopen(file_name.c_str(), "rb");
	if (!f) throw std::runtime_error("cannot open file for reading");
	//if (fseek(f, 0, SEEK_END)) throw std::runtime_error("seeking failed");
	uint64_t count = 0;
	if (!fread(&count, sizeof(uint64_t), 1, f))  throw std::runtime_error("size cannot be read");
	if (!fread(&dimension, sizeof(uint64_t), 1, f))  throw std::runtime_error("dimension cannot be read");
	do
	{
		point* p = new point();
		if (!fread(&p->coords[0], sizeof(value_t), dimension, f))  throw std::runtime_error("value cannot be read");
		data.push_back(p);
	}
	while (--count);
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
	if (!fwrite(&dimension, sizeof(uint64_t), 1, f)) throw std::runtime_error("dimension cannot be written");
	uint64_t i = 0;
	for (means_t::const_iterator it = means.begin(); it != means.end(); ++it)
	{
		if (!fwrite(&(*it)->coords[0], sizeof(value_t), dimension, f)) throw std::runtime_error("value cannot be written");
		if (!fwrite(&i, sizeof(cluster_t), 1, f)) throw std::runtime_error("value cannot be written");
		++i;
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");


	f = fopen(clusters_file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	if (!fwrite(&dimension, sizeof(uint64_t), 1, f)) throw std::runtime_error("dimension cannot be written");
	for (data_t::const_iterator it = data.begin(); it != data.end(); ++it)
	{
		if (!fwrite(&(*it)->coords[0], sizeof(value_t), dimension, f)) throw std::runtime_error("value cannot be written");
		if (!fwrite(&(*it)->cluster, sizeof(cluster_t), 1, f)) throw std::runtime_error("value cannot be written");
		//if (!fwrite(&(*it)->distanceFromCluster, sizeof(value_t), 1, f)) throw std::runtime_error("distance cannot be written");
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
		//std::size_t size = lexical_cast<std::size_t>(s_size);
		//std::size_t seed = lexical_cast<std::size_t>(s_seed);
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

		means_t means;
		for (int i = 0; i < k; ++i)
		{
			means.push_back(new mean(data[i]));
		}

		clock_t start, end;
		start = clock();

		while (iterations--)
		{
			assign_to_clusters(data, means);
			compute_means(data, means);
		}

		end = clock();
		std::cout << static_cast<double>(end-start)/CLOCKS_PER_SEC << "\n";

		save_results(means_file_name, clusters_file_name, means, data);
		return 0;
	}
	usage();
	return 1;
}