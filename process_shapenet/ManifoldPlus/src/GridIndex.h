#ifndef MANIFOLD2_GRIDINDEX_H_
#define MANIFOLD2_GRIDINDEX_H_

#include "types.h"

struct GridIndex
{
public:
	GridIndex(){}
	GridIndex(int x, int y, int z)
	: id(x,y,z)
	{}
	bool operator<(const GridIndex& ind) const
	{
		int i = 0;
		while (i < 3 && id[i] == ind.id[i])
			i++;
		return (i < 3 && id[i] < ind.id[i]);
	}
	GridIndex operator+(const GridIndex& ind) const
	{
		GridIndex grid(*this);
		grid.id += ind.id;
		return grid;
	}
	GridIndex operator/(int x) const
	{
		return GridIndex(id[0]/x,id[1]/x,id[2]/x);
	}
	Vector3i id;
};

#endif