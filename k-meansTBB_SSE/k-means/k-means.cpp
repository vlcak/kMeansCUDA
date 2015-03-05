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

#include <xmmintrin.h>

#include <time.h>

typedef float value_t;
typedef __m128 value128_t;
typedef unsigned char cluster_t;

uint64_t dimension;
uint64_t realDimension;

struct point
{
	point() : cluster(0)
	{
		coords = (value_t*) _aligned_malloc(dimension * sizeof(value_t), 16);
	}

	~point()
	{
		_aligned_free(coords);
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
		coords = (value_t*) _aligned_malloc(dimension * sizeof(value_t), 16);
		memcpy(coords, p->coords, dimension * sizeof(value_t));
	}
	mean() : count(0)
	{
		coords = (value_t*) _aligned_malloc(dimension * sizeof(value_t), 16);
		memset(coords, 0, dimension * sizeof(value_t));
	}

	~mean()
	{
		_aligned_free(coords);
	}

	value_t* coords;
	value_t count;
};

typedef std::vector<point*> data_t;
typedef std::vector<mean*> means_t;
typedef std::vector<std::size_t> counts_t;
typedef double distance_t;

inline value_t power2(value_t x)
{
	return x*x;
}

inline value_t distance(const point* a, const point* b)
{
	value128_t sub;
	value128_t mul;
	value128_t totalSum = _mm_setzero_ps();
	for (size_t i = 0; i < dimension; i += 4)
	{
		value128_t* aCoords = (value128_t*)&a->coords[i];
		value128_t* bCoords = (value128_t*)&b->coords[i];
		sub = _mm_sub_ps(*aCoords, *bCoords);
		mul = _mm_mul_ps(sub, sub);
		totalSum = _mm_add_ps(totalSum, mul);
		//value += power2(b->coords[i] - a->coords[i]);
	}
	totalSum = _mm_hadd_ps(totalSum, totalSum);
	totalSum = _mm_hadd_ps(totalSum, totalSum);
	return (float)_mm_extract_ps(totalSum, 0);
}

inline value_t distance(means_t::const_iterator a, const point* b)
{
	value128_t sub;
	value128_t mul;
	value128_t totalSum = _mm_setzero_ps();
	for (size_t i = 0; i < dimension; i += 4)
	{
		value128_t* aCoords = (value128_t*)&(*a)->coords[i];
		value128_t* bCoords = (value128_t*)&b->coords[i];
		sub = _mm_sub_ps(*aCoords, *bCoords);
		mul = _mm_mul_ps(sub, sub);
		totalSum = _mm_add_ps(totalSum, mul);
		//value += power2(b->coords[i] - a->coords[i]);
	}
	totalSum = _mm_hadd_ps(totalSum, totalSum);
	totalSum = _mm_hadd_ps(totalSum, totalSum);
	float value = (float)_mm_extract_ps(totalSum, 0);
	return value;
}

struct CountMinDistanceTask
{
	data_t& data;
	const means_t& means;
	means_t newMeans;
	CountMinDistanceTask(data_t& da, const means_t& m) :data(da), means(m)
	{
		newMeans = means_t(means.size());
		for (int i = 0; i < means.size(); ++i)
		{
			newMeans[i] = new mean();
		}
	}

	CountMinDistanceTask(CountMinDistanceTask& cmdt, tbb::split) :data(cmdt.data), means(cmdt.means)
	{
		newMeans = means_t(cmdt.means.size());
		for (int i = 0; i < cmdt.means.size(); ++i)
		{
			newMeans[i] = new mean();
		}
	}

	void join(CountMinDistanceTask& cmdt)
	{
		means_t::const_iterator cm = cmdt.newMeans.cbegin();
		for (means_t::iterator nm = newMeans.begin(); nm != newMeans.end(); ++nm,++cm)
		{
			(*nm)->count += (*cm)->count;
			value128_t* aCoords = (value128_t*)(*nm)->coords;
			value128_t* bCoords = (value128_t*)(*cm)->coords;
			for (size_t i = 0; i < dimension; i += 4)
			{
				*aCoords = _mm_add_ps(*aCoords, *bCoords);
				++aCoords;
				++bCoords;
			}

			_aligned_free((*cm)->coords);
			//delete(*cm);
		}
	}

	void operator()(const tbb::blocked_range<size_t>& range) {
		value_t min_distance(LLONG_MAX);
		value_t dist(0);
		cluster_t cluster(0);
		means_t::const_iterator m;
		data_t::iterator d = data.begin() + range.begin();
		value128_t sub, mul, totalSum;
		value128_t* aCoords,* bCoords;
		for (tbb::blocked_range<size_t>::const_iterator r = range.begin(); r != range.end(); ++r)
		{
			min_distance = LLONG_MAX;
			
			m = means.cbegin();
			for (cluster_t c = 0; c < means.size(); ++c)
			{
				//dist = distance(m, (*d));

				totalSum = _mm_setzero_ps();
				aCoords = (value128_t*)(*m)->coords;
				bCoords = (value128_t*)(*d)->coords;
				for (size_t i = 0; i < dimension; i += 4)
				{
					sub = _mm_sub_ps(*aCoords, *bCoords);
					mul = _mm_mul_ps(sub, sub);
					totalSum = _mm_add_ps(totalSum, mul);
					++aCoords;
					++bCoords;
					//value += power2(b->coords[i] - a->coords[i]);
				}
				totalSum = _mm_hadd_ps(totalSum, totalSum);
				totalSum = _mm_hadd_ps(totalSum, totalSum);
				dist = totalSum.m128_f32[0];

				++m;
				if (dist < min_distance)
				{
					min_distance = dist;
					cluster = c;
				}
			}
			aCoords = (value128_t*)newMeans[cluster]->coords;
			bCoords = (value128_t*)(*d)->coords;
			for (size_t i = 0; i < dimension; i+=4)
			{
				*aCoords = _mm_add_ps(*aCoords, *bCoords);
				++aCoords;
				++bCoords;
				//newMeans[cluster]->coords[i] += (*d)->coords[i];
			}
			++newMeans[cluster]->count;
			(*d)->cluster = cluster;
			(*d)->distanceFromCluster = min_distance;
			++d;
		}
	}

};

void generate(std::size_t count, unsigned int seed, data_t& data)
{
	srand(seed);
	data.reserve(count);
	while (count--)
	{
		std::vector<value_t> coords(dimension);
		point* p = new point();
		for (size_t i = 0; i < dimension; i++)
		{
			p->coords[i] = (value_t)rand();
		}
		data.push_back(p);
	}
}


void assign_to_clusters(data_t& data, means_t& means, size_t granularity)
{
	CountMinDistanceTask cmdt = CountMinDistanceTask(data, means);
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, data.size(), granularity), cmdt);
	means_t::const_iterator nm = cmdt.newMeans.cbegin();
	for (means_t::iterator m = means.begin(); m != means.end(); ++m, ++nm)
	{
		value_t invCount = (1 / (*nm)->count);
		value128_t count = _mm_load1_ps(&invCount);
		value128_t* aCoords = (value128_t*)(*m)->coords;
		value128_t* bCoords = (value128_t*)(*nm)->coords;
		for (size_t i = 0; i < dimension; i += 4)
		{		
			*aCoords = _mm_mul_ps(*bCoords, count);
			++aCoords;
			++bCoords;
			//(*m)->coords[i] = (*nm)->coords[i] / (*nm)->count;
		}
		_aligned_free ((*nm)->coords);
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
		if (!fwrite(&((*it)->coords), sizeof(value_t), realDimension, f)) throw std::runtime_error("value cannot be written");
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
	if (!fread(&realDimension, sizeof(uint64_t), 1, f))  throw std::runtime_error("dimension cannot be read");
	dimension = 4 * (uint16_t)ceil(realDimension / 4.0);
	do
	{
		point* p = new point();
		if (!fread(&p->coords[0], sizeof(value_t), realDimension, f))  throw std::runtime_error("value cannot be read");
		for (int i = realDimension; i < dimension; i++)
		{
			p->coords[i] = 0.0f;
		}
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
	if (!fwrite(&realDimension, sizeof(uint64_t), 1, f)) throw std::runtime_error("dimension cannot be written");
	uint64_t i = 0;
	for (means_t::const_iterator it = means.begin(); it != means.end(); ++it)
	{
		if (!fwrite(&(*it)->coords[0], sizeof(value_t), realDimension, f)) throw std::runtime_error("value cannot be written");
		if (!fwrite(&i, sizeof(cluster_t), 1, f)) throw std::runtime_error("value cannot be written");
		++i;
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");


	f = fopen(clusters_file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	if (!fwrite(&realDimension, sizeof(uint64_t), 1, f)) throw std::runtime_error("dimension cannot be written");
	for (data_t::const_iterator it = data.begin(); it != data.end(); ++it)
	{
		if (!fwrite(&(*it)->coords[0], sizeof(value_t), realDimension, f)) throw std::runtime_error("value cannot be written");
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
		std::string s_dimension(argv[4]);
		std::string s_seed(argv[5]);
		std::size_t size = lexical_cast<std::size_t>(s_size);
		dimension = lexical_cast<uint16_t>(s_dimension);
		unsigned int seed = lexical_cast<unsigned int>(s_seed);
		data_t data;
		generate(size, seed, data);
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
		means_t means;
		for (int i = 0; i < k; ++i)
		{
			means.push_back(new mean(data[i]));
		}

		size_t granularity = (2048 * 16384) / (k * data.size()); //ideální jednotka granularity
		if (granularity == 0)
		{
			++granularity;
		}

		clock_t start, end;
		start = clock();

		while (iterations--)
		{
			assign_to_clusters(data, means, granularity);
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