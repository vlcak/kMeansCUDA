#include <string>
#include <vector>
#include <random>
#include <stdint.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <iomanip>

void showHelp()
{}

template<typename T>
void save(const std::string& file_name, const std::vector<std::vector<T>>& data)
{
	FILE* f = fopen(file_name.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	//
	uint64_t size[2] = { data.size(), data[0].size() };
	if (!fwrite(&size, sizeof(uint64_t), 2, f)) throw std::runtime_error("value cannot be written");
	/*if (data.size() > 0)
	{
		if (!fwrite(*data[0].size(), sizeof(size_t), 1, f)) throw std::runtime_error("value cannot be written");
	}*/
	for (std::vector<T> point : data)
	{
		for (T coord : point)
		{
			if (!fwrite(&coord, sizeof(T), 1, f)) throw std::runtime_error("value cannot be written");
		}
	}
	/*
	for (data_t::const_iterator it = data.begin(); it != data.end(); ++it)
	{
		for (it::const_iterator itp = itp->begin(); itp != itp->end(); ++itp)
		{
			if (!fwrite(&it, sizeof(T), 1, f)) throw std::runtime_error("value cannot be written");
		}
	}*/
	if (fclose(f)) throw std::runtime_error("closing the file failed");
}

template<typename T>
void saveText(const std::string& file_name, const std::vector<std::vector<T>>& data)
{
	std::ofstream outFile;
	outFile.open(file_name);
	if (outFile.fail()) throw std::runtime_error("cannot open file for writing");
	for (std::vector<T> point : data)
	{
		for (T coord : point)
		{
			outFile << std::setprecision(8) << coord << ";";
		}
		outFile << std::endl;
	}
	outFile.close();
}

template<typename T>
T vectLength(std::vector<T> v)
{
	T lengthSquered = 0;
	for (T coord : v)
	{
		lengthSquered += coord * coord;
	}
	return std::sqrt(lengthSquered);
}

float vectLength(std::vector<float> v)
{
	float lengthSquered = 0;
	for (float coord : v)
	{
		lengthSquered += coord * coord;
	}
	return std::sqrtf(lengthSquered);
}

template<typename T>
std::vector<std::vector<T>> generateUniformPoints(const unsigned long long pointsCount, const unsigned int dimensions, T lowerBound, T upperBound)
{
	std::vector<std::vector<T> > points(pointsCount, std::vector<T>(dimensions));
	std::uniform_real_distribution<T> uniformDist(lowerBound, upperBound);
	std::default_random_engine re;
	for (size_t i = 0; i < pointsCount; i++)
	{
		for (size_t j = 0; j < dimensions; j++)
		{
			points[i][j] = uniformDist(re);
		}
	}
	return points;
}

template<typename T>
std::vector<std::vector<T>> generateUniformPoints(const unsigned int clustersCount, const float clusterRadius, const unsigned long long pointsCount, const unsigned int dimensions, T lowerBound, T upperBound, std::string fileName)
{
	FILE* f = fopen(fileName.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	//
	uint64_t size[2] = { pointsCount, dimensions };
	if (!fwrite(&size, sizeof(uint64_t), 2, f)) throw std::runtime_error("value cannot be written");
	std::vector<std::vector<T> > points = generateUniformPoints(clustersCount, dimensions, lowerBound, upperBound);
	for (std::vector<T> point : points)
	{
		for (T coord : point)
		{
			if (!fwrite(&coord, sizeof(T), 1, f)) throw std::runtime_error("value cannot be written");
		}
	}
	std::normal_distribution<T> vectorRandom(0.0, 1.0);
	std::uniform_real_distribution<T> lengthRandom(0, clusterRadius);
	std::default_random_engine re;
	std::vector<T> point(dimensions);
	for (size_t i = clustersCount; i < pointsCount; i++)
	{
		for (size_t j = 0; j < dimensions; j++)
		{
			point[j] = vectorRandom(re);// +points[i % clustersCount][j];
		}
		T length = lengthRandom(re) / vectLength(point);
		std::transform(point.begin(), point.end(), point.begin(),
			std::bind2nd(std::multiplies<T>(), length));
		std::transform(points[i % clustersCount].begin(), points[i % clustersCount].end(), point.begin(), point.begin(),
			std::plus<T>());
		for (T coord : point)
		{
			if (!fwrite(&coord, sizeof(T), 1, f)) throw std::runtime_error("value cannot be written");
		}
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");
	return points;
}

template<typename T>
std::vector<std::vector<T>> generateNormalPoints(const unsigned int clustersCount, const float clusterRadius, const unsigned long long pointsCount, const unsigned int dimensions, T lowerBound, T upperBound, std::string fileName)
{
	FILE* f = fopen(fileName.c_str(), "wb");
	if (!f) throw std::runtime_error("cannot open file for writing");
	//
	uint64_t size[2] = { pointsCount, dimensions };
	if (!fwrite(&size, sizeof(uint64_t), 2, f)) throw std::runtime_error("value cannot be written");
	std::vector<std::vector<T> > points = generateUniformPoints(clustersCount, dimensions, lowerBound, upperBound);
	for (std::vector<T> point : points)
	{
		for (T coord : point)
		{
			if (!fwrite(&coord, sizeof(T), 1, f)) throw std::runtime_error("value cannot be written");
		}
	}
	std::normal_distribution<T> vectorRandom(0.0, 1.0);
	std::default_random_engine re;
	std::vector<T> point(dimensions);
	for (size_t i = clustersCount; i < pointsCount; i++)
	{
		for (size_t j = 0; j < dimensions; j++)
		{
			point[j] = vectorRandom(re);
		}
		T length = vectorRandom(re) * clusterRadius / vectLength(point);
		std::transform(point.begin(), point.end(), point.begin(),
			std::bind2nd(std::multiplies<T>(), length));
		std::transform(points[i % clustersCount].begin(), points[i % clustersCount].end(), point.begin(), point.begin(),
			std::plus<T>());
		for (T coord : point)
		{
			if (!fwrite(&coord, sizeof(T), 1, f)) throw std::runtime_error("value cannot be written");
		}
	}
	if (fclose(f)) throw std::runtime_error("closing the file failed");
	return points;
}

int main(int argc, const char* argv[])
{
	unsigned int clustersCount = 0, dimensions = 0;
	unsigned long long pointsCount = 0;
	float clusterRadius = 0, lowerBound = 0.0f, upperBound = 10.0f;
	std::string outputFileName = "";
	bool normalDistribution = false, useDouble = false, textOutput = false;
	for (int i = 1; i < argc; ++i)
	{
		if ((argv[i][0] == '-') && (std::strlen(argv[i]) == 2))
		{
			switch (argv[i][1])
			{
			case 'c':
				if (argc > i + 1)
				{
					clustersCount = std::strtoul(argv[i + 1], NULL, 10);
					++i;
				}
				else
				{
					showHelp();
				}
				break;
			case 'p':
				if (argc > i + 1)
				{
					pointsCount = std::strtoull(argv[i + 1], NULL, 10);
					++i;
				}
				else
				{
					showHelp();
				}
				break;
			case 'd':
				if (argc > i + 1)
				{
					dimensions = std::strtoul(argv[i + 1], NULL, 10);
					++i;
				}
				else
				{
					showHelp();
				}
				break;
			case 'r':
				if (argc > i + 1)
				{
					clusterRadius = std::strtof(argv[i + 1], NULL);
					++i;
				}
				else
				{
					showHelp();
				}
				break;

			case 'o':
				if (argc > i + 1)
				{
					outputFileName = std::string(argv[i + 1]);
					++i;
				}
				else
				{
					showHelp();
				}
				break;
			}
		}
		else
		{
			if (0 == strcmp(argv[i], "--normal"))
			{
				normalDistribution = true;
			}
			else if (0 == strcmp(argv[i], "--double"))
			{
				useDouble = true;
			} else if(0 == strcmp(argv[i], "--text"))
			{
				textOutput = true;
			}
			else
			{
				showHelp();
			}
		}
	}

    std::vector< std::vector<double> > pointsD;
    std::vector< std::vector<float> > pointsF;

    lowerBound = -128;
    upperBound = 128;
    std::string fileName;
	size_t cluster = 32;

	for (size_t size = 8192; size <= 1048576; size *= 2)
    {
        for (size_t dimension = 3; dimension <= 3; dimension = dimension < 32 ? dimension * 2 : dimension + 32)
        {
            //for (size_t cluster = 2; cluster <= 32; cluster *= 2)
            {
                std::cout << "Generating data: size:" + std::to_string(size / 1000) + "K, dimension: " + std::to_string(dimension) + ", clusters count = " + std::to_string(cluster) << std::endl;
                fileName = "../../data/dataDN" + std::to_string(dimension) + "D" + std::to_string(size / 1000) + "K" + std::to_string(cluster) + "C.dat";
				pointsD = generateNormalPoints<double>(cluster, clusterRadius, size, dimension, lowerBound, upperBound, fileName);
                //save(fileName, pointsD);
                fileName = "../../data/dataDU" + std::to_string(dimension) + "D" + std::to_string(size / 1000) + "K" + std::to_string(cluster) + "C.dat";
				pointsD = generateUniformPoints<double>(cluster, clusterRadius, size, dimension, lowerBound, upperBound, fileName);
                //save(fileName, pointsD);
                fileName = "../../data/dataFN" + std::to_string(dimension) + "D" + std::to_string(size / 1000) + "K" + std::to_string(cluster) + "C.dat";
				pointsF = generateNormalPoints<float>(cluster, clusterRadius, size, dimension, lowerBound, upperBound, fileName);
                //save(fileName, pointsF);
                fileName = "../../data/dataFU" + std::to_string(dimension) + "D" + std::to_string(size / 1000) + "K" + std::to_string(cluster) + "C.dat";
				pointsF = generateUniformPoints<float>(cluster, clusterRadius, size, dimension, lowerBound, upperBound, fileName);
                //save(fileName, pointsF);
            }
        }
    }

	/*if (useDouble)
	{
		std::vector< std::vector<double> > points;
		if (clustersCount == 0)
		{
			points = generateUniformPoints<double>(pointsCount, dimensions, lowerBound, upperBound);
			}
		else
		{
			points = normalDistribution
				? generateNormalPoints<double>(clustersCount, clusterRadius, pointsCount, dimensions, lowerBound, upperBound)
				: generateUniformPoints<double>(clustersCount, clusterRadius, pointsCount, dimensions, lowerBound, upperBound);
		}
		textOutput ? saveText(outputFileName, points) : save(outputFileName, points);
	}
	else
	{
		std::vector< std::vector<float> > points;
		if (clustersCount == 0)
		{
			points = generateUniformPoints<float>(pointsCount, dimensions, lowerBound, upperBound);
		}
		else
		{
			points = normalDistribution
				? generateNormalPoints<float>(clustersCount, clusterRadius, pointsCount, dimensions, lowerBound, upperBound)
				: generateUniformPoints<float>(clustersCount, clusterRadius, pointsCount, dimensions, lowerBound, upperBound);
		}
		textOutput ? saveText(outputFileName, points) : save(outputFileName, points);
	}*/

}