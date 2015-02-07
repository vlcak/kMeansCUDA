#include <stdio.h>
#include <vector>
#include <cmath>
#include <cassert>
#include <iterator>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <stdexcept>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

typedef float value_t;
typedef unsigned char cluster_t;


struct point
{
	point(std::vector<value_t>& coords) : coords(coords), cluster(0) {}
	std::vector<value_t> coords;
	cluster_t cluster;
};

std::size_t dimension;

struct mean
{
	mean() : coords(dimension, 0), count(0) {}
	std::vector<value_t> coords;
	value_t count;
};

typedef std::vector<point> data_t;
typedef std::vector<point> means_t;
typedef std::vector<std::size_t> counts_t;
typedef double distance_t;

inline value_t power2(value_t x)
{
	return x*x;
}

inline value_t distance(const point* a, const point* b)
{
	value_t value = 0;;
	for (size_t i = 0; i < a->coords.size(); i++)
	{
		value += power2(b->coords[i] - a->coords[i]);
	}
	return value;
}

inline value_t distance(std::vector<point>::const_iterator a, const point* b)
{
	value_t value = 0;;
	for (size_t i = 0; i < a->coords.size(); i++)
	{
		value += power2(b->coords[i] - a->coords[i]);
	}
	return value;
}

struct CountMinDistanceTask
{
	data_t& data;
	const means_t& means;
	std::vector<mean> newMeans;
	CountMinDistanceTask(data_t& da, const means_t& m) :data(da), means(m)
	{
		newMeans = std::vector<mean>(means.size());
	}

	CountMinDistanceTask(CountMinDistanceTask& cmdt, tbb::split) :data(cmdt.data), means(cmdt.means)
	{
		newMeans = std::vector<mean>(cmdt.means.size());
	}

	void join(CountMinDistanceTask& cmdt)
	{
		std::vector<mean>::const_iterator cm = cmdt.newMeans.cbegin();
		for (std::vector<mean>::iterator nm = newMeans.begin(); nm != newMeans.end(); ++nm,++cm)
		{
			nm->count += cm->count;
			for (size_t i = 0; i < nm->coords.size(); i++)
			{
				nm->coords[i] += cm->coords[i];
			}
		}
	}

	void operator()(const tbb::blocked_range<size_t>& range) {
		value_t min_distance(LLONG_MAX);
		value_t dist(0);
		cluster_t cluster(0);
		std::vector<point>::const_iterator m;
		point* d = &data[range.begin()];
		for (tbb::blocked_range<size_t>::const_iterator r = range.begin(); r != range.end(); r++)
		{
			min_distance = LLONG_MAX;
			
			m = means.cbegin();
			for (cluster_t i = 0; i < means.size(); i++)
			{
				dist = distance(m++, d);
				if (dist < min_distance)
				{
					min_distance = dist;
					cluster = i;
				}
			}

			for (size_t i = 0; i < newMeans[cluster].coords.size(); i++)
			{
				newMeans[cluster].coords[i] += d->coords[i];
			}
			++newMeans[cluster].count;
			d++->cluster = cluster;
		}
	}

};

void generate(std::size_t count, unsigned int seed, data_t& data, size_t dimension)
{
	srand(seed);
	data.reserve(count);
	while (count--)
	{
		std::vector<value_t> coords(dimension);
		for (size_t i = 0; i < dimension; i++)
		{
			coords[i] = rand();
		}
		data.push_back(point(coords));
	}
}


void assign_to_clusters(data_t& data, means_t& means, size_t granularity)
{
	CountMinDistanceTask cmdt = CountMinDistanceTask(data, means);
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, data.size(), granularity), cmdt);
	std::vector<mean>::const_iterator nm = cmdt.newMeans.cbegin();
	for (std::vector<point>::iterator m = means.begin(); m!=means.end(); m++,nm++)
	{
		for (size_t i = 0; i < m->coords.size(); i++)
		{
			m->coords[i] = nm->coords[i] / nm->count;
		}
	}
	/*mean* nm = &cmdt.newMeans[0];
	point* m = &means[0];
	for (int i = 0; i < cmdt.newMeans.size(); i++)
	{
		m->x = nm->x / nm->count;
		m++->y = nm++->y / nm->count;
	}*/
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
	uint64_t count = 0;
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
		std::string s_dimension(argv[4]);
		std::string s_seed(argv[5]);
		std::size_t size = lexical_cast<std::size_t>(s_size);
		dimension = lexical_cast<std::size_t>(s_dimension);
		unsigned int seed = lexical_cast<unsigned int>(s_seed);
		data_t data;
		generate(size, seed, data, dimension);
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
		tbb::task_scheduler_init init;


		data_t data;
		load(file_name, data);

		assert(data.size() >= k);
		assert(k > 0);
		data_t means(data.begin(), data.begin() + k);

		size_t granularity = (2048 * 16384) / (k * data.size()); //ideální jednotka granularity
		if (granularity == 0)
		{
			++granularity;
		}

		while (iterations--)
		{
			assign_to_clusters(data, means, granularity);
		}

		save_results(means_file_name, clusters_file_name, means, data);
		return 0;
	}
	usage();
	return 1;
}