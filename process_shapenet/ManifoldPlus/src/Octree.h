#ifndef OCTREE_H_
#define OCTREE_H_

#include <list>
#include <map>
#include <set>
#include <vector>


#include "GridIndex.h"

class Octree
{
public:
	Octree();
	Octree(const Vector3 min_c, const Vector3 max_c, const MatrixI& faces);
	Octree(const Vector3& min_c, const Vector3& volume_size);
	~Octree();

	bool IsExterior(const Vector3 &p);

	bool Intersection(int face_index, const Vector3& min_corner,
		const Vector3& size, const MatrixD& V);


	void Split(const MatrixD& V);
	void BuildConnection();
	void ConnectTree(Octree* l, Octree* r, int dim);
	void ConnectEmptyTree(Octree* l, Octree* r, int dim);

	void ExpandEmpty(std::list<Octree*>& empty_list,
		std::set<Octree*>& empty_set, int dim);

	void BuildEmptyConnection();

	void ConstructFace(const Vector3i& start,
		std::map<GridIndex,int>* vcolor,
		std::vector<Vector3>* vertices,
		std::vector<Vector4i>* faces,
		std::vector<std::set<int> >* v_faces);

	Vector3 min_corner_, volume_size_;
	int level_;
	int number_;
	int occupied_;
	int exterior_;

	Octree* children_[8];
	Octree* connection_[6];
	Octree* empty_connection_[6];
	std::list<Octree*> empty_neighbors_;

	std::vector<Vector3i> F_;
	std::vector<int> Find_;
};


#endif
