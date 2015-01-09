#include <vector>
#include <cmath>
#include <cassert>
#include <iterator>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <stdexcept>


typedef unsigned long long value_t;
typedef unsigned char cluster_t;

struct point
{
	point(value_t x, value_t y) : x(x), y(y), cluster(0) {}
	value_t x;
	value_t y;
	cluster_t cluster;
};

typedef std::vector<point> data_t;
typedef std::vector<point> means_t;

typedef double distance_t;

void generate(std::size_t count,unsigned int seed, data_t& data)
{
	srand(seed);
	data.reserve(count);
	while(count--)
	{
		data.push_back(point(rand(),rand()));

	}
}

inline distance_t distance(const point& a, const point& b)
{
	return std::sqrt((double)((b.x-a.x)*(b.x-a.x)) + ((b.y-a.y)*(b.y-a.y)));
}

void assign_to_clusters(data_t& data, const means_t& means)
{
	assert(means.size()>0);
	for(data_t::iterator it=data.begin();it!=data.end();++it)
	{
		cluster_t closest_cluster(0);
		distance_t mindist(distance(*it,means[0]));
		for(means_t::const_iterator mit=means.begin()+1;mit!=means.end();++mit)
		{
			distance_t dist=distance(*it,*mit);
			if (dist<mindist) { closest_cluster=static_cast<cluster_t>(std::distance(means.begin(),mit)); mindist=dist; }
		}
		it->cluster=closest_cluster;
	}
}

void compute_means(const data_t& data, means_t& means)
{
	std::vector<std::size_t> counts(means.size(),0);
	for(means_t::iterator mit=means.begin();mit!=means.end();++mit)
	{
		mit->x=0;
		mit->y=0;
	}
	for(data_t::const_iterator it=data.begin();it!=data.end();++it)
	{
		++counts[it->cluster];
		means[it->cluster].x+=it->x;
		means[it->cluster].y+=it->y;
	}
	for(means_t::iterator mit=means.begin();mit!=means.end();++mit)
	{
		mit->x/=counts[std::distance(means.begin(),mit)];
		mit->y/=counts[std::distance(means.begin(),mit)];
	}
}

void save(const std::string& file_name, const data_t& data)
{
	FILE* f = fopen(file_name.c_str(),"wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	for(data_t::const_iterator it=data.begin();it!=data.end();++it)
	{
		if (!fwrite(&it->x,sizeof(value_t),1,f)) throw std::runtime_error("value cannot be written");
		if (!fwrite(&it->y,sizeof(value_t),1,f)) throw std::runtime_error("value cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");
}

void load(const std::string& file_name, data_t& data)
{
	FILE* f = fopen(file_name.c_str(),"rb");
	if (!f) throw std::runtime_error("cannot open file for reading");
	if (fseek(f,0,SEEK_END)) throw std::runtime_error("seeking failed");
	std::size_t count=ftell(f)/(2*sizeof(value_t));
	if (fseek(f,0,SEEK_SET)) throw std::runtime_error("seeking failed");
	while(count--)
	{
		value_t x,y;
		if (!fread(&x,sizeof(value_t),1,f))  throw std::runtime_error("value cannot be read");
		if (!fread(&y,sizeof(value_t),1,f))  throw std::runtime_error("value cannot be read");
		data.push_back(point(x,y));
	}
}

template<typename T>
T lexical_cast(const std::string& x)
{
	std::istringstream stream(x);
	T res;
	stream>>res;
	return res;
}

void usage()
{
		std::cout<<"Usage:"<<std::endl<<"kmeans <data_file> <means_file> <clusters_file> <k> <iterations>"<<std::endl<<"kmeans --generate <data_file> <size> <seed>"<<std::endl;
}

void save_results(const std::string& means_file_name, const std::string& clusters_file_name, const means_t& means, const data_t& data)
{
	FILE* f = fopen(means_file_name.c_str(),"wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	for(means_t::const_iterator it=means.begin();it!=means.end();++it)
	{
		if (!fwrite(&it->x,sizeof(value_t),1,f)) throw std::runtime_error("value cannot be written");
		if (!fwrite(&it->y,sizeof(value_t),1,f)) throw std::runtime_error("value cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");


	f = fopen(clusters_file_name.c_str(),"wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	for(data_t::const_iterator it=data.begin();it!=data.end();++it)
	{
		if (!fwrite(&it->cluster,sizeof(cluster_t),1,f)) throw std::runtime_error("value cannot be written");
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");
}

int main(int argc, const char* argv[])
{
	if (argc==5)
	{
		std::string option(argv[1]);
		if (option!="--generate") { usage(); return 1; }
		std::string file_name(argv[2]);
		std::string s_size(argv[3]);
		std::string s_seed(argv[4]);
		std::size_t size=lexical_cast<std::size_t>(s_size);
		std::size_t seed=lexical_cast<std::size_t>(s_seed);
		data_t data;
		generate(size,seed,data);
		save(file_name,data);
		return 0;
	}

	if (argc==6)
	{
		std::string file_name(argv[1]);
		std::string means_file_name(argv[2]);
		std::string clusters_file_name(argv[3]);
		std::string s_k(argv[4]);
		std::string s_iterations(argv[5]);
		std::size_t k=lexical_cast<std::size_t>(s_k);
		std::size_t iterations=lexical_cast<std::size_t>(s_iterations);

		data_t data;
		load(file_name,data);

		assert(data.size()>=k);

		data_t means(data.begin(),data.begin()+k);

		while(iterations--)
		{
			assign_to_clusters(data,means);
			compute_means(data,means);
		}

		save_results(means_file_name,clusters_file_name,means,data);
		return 0;
	}
	usage();
	return 1;
}
