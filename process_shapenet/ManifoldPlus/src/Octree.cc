#include "Octree.h"

#include "Intersection.h"

Octree::Octree()
{
	memset(children_, 0, sizeof(Octree*) * 8);
	memset(connection_, 0, sizeof(Octree*) * 6);
	memset(empty_connection_, 0, sizeof(Octree*) * 6);
	level_ = 0;
	number_ = 1;
	occupied_ = 1;
	exterior_ = 0;
}


Octree::Octree(const Vector3 min_c, const Vector3 max_c, const MatrixI& faces)
{
	memset(children_, 0, sizeof(Octree*) * 8);
	memset(connection_, 0, sizeof(Octree*) * 6);
	memset(empty_connection_, 0, sizeof(Octree*) * 6);
	level_ = 0;
	number_ = 1;
	occupied_ = 1;
	exterior_ = 0;

	min_corner_ = min_c;
	volume_size_ = max_c - min_c;

	int ind = 0;
	for (int i = 1; i < 3; ++i)
		if (volume_size_[i] > volume_size_[ind])
			ind = i;
	for (int i = 0; i < 3; ++i)
	{
		min_corner_[i] -= (volume_size_[ind] - volume_size_[i]) * 0.5;
	}
	volume_size_ = Vector3(1, 1, 1) * volume_size_[ind];
	F_.resize(faces.rows());
	Find_.resize(faces.size());
	for (int i = 0; i < (int)faces.rows(); ++i) {
		F_[i] = faces.row(i);
		Find_[i] = i;
	}
}

Octree::Octree(const Vector3& min_c, const Vector3& volume_size)
{
	memset(children_, 0, sizeof(Octree*) * 8);
	memset(connection_, 0, sizeof(Octree*) * 6);
	memset(empty_connection_, 0, sizeof(Octree*) * 6);
	level_ = 0;
	number_ = 1;
	occupied_ = 1;
	exterior_ = 0;

	min_corner_ = min_c;
	volume_size_ = volume_size;
}

Octree::~Octree()
{
	for (int i = 0; i < 8; ++i)
	{
		if (children_[i])
			delete children_[i];
	}
}


bool Octree::IsExterior(const Vector3& p)
{
	for (int i = 0; i < 3; ++i)
		if (p[i] < min_corner_[i] || p[i] > min_corner_[i] + volume_size_[i])
			return true;
	if (!occupied_)
		return exterior_;
	if (level_ == 0)
		return false;
	int index = 0;
	for (int i = 0; i < 3; ++i)
	{
		index *= 2;
		if (p[i] > min_corner_[i] + volume_size_[i] / 2)
			index += 1;
	}
	return children_[index]->IsExterior(p);
}

bool Octree::Intersection(int Find_ex, const Vector3& min_corner,
	const Vector3& size, const MatrixD& V)
{
	float boxcenter[3];
	float boxhalfsize[3];
	float triverts[3][3];
	for (int i = 0; i < 3; ++i)
	{
		boxhalfsize[i] = size[i] * 0.5;
		boxcenter[i] = min_corner[i] + boxhalfsize[i];
	}
	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			triverts[i][j] = V(F_[Find_ex][i],j);
		}
	}
	return TriBoxOverlap(boxcenter, boxhalfsize, triverts);
}

void Octree::Split(const MatrixD& V)
{
	level_ += 1;
	number_ = 0;
	if (level_ > 1) {
		for (int i = 0; i < 2; ++i) {
			for (int j = 0; j < 2; ++j) {
				for (int k = 0; k < 2; ++k) {
					int ind = i * 4 + j * 2 + k;
					if (children_[ind] && children_[ind]->occupied_) {
						children_[ind]->Split(V);
						number_ += children_[ind]->number_;
					}
				}
			}
		}
		F_.clear();
		Find_.clear();
		return;
	}
	Vector3 halfsize = volume_size_ * 0.5;
	for (int i = 0; i < 2; ++i) {
		for (int j = 0; j < 2; ++j) {
			for (int k = 0; k < 2; ++k) {
				int ind = i * 4 + j * 2 + k;

				Vector3 startpoint = min_corner_;
				startpoint[0] += i * halfsize[0];
				startpoint[1] += j * halfsize[1];
				startpoint[2] += k * halfsize[2];
									
				children_[ind] = new Octree(startpoint, halfsize);
				children_[ind]->occupied_ = 0;
				children_[ind]->number_ = 0;

				for (int face = 0; face < (int)F_.size(); ++face) {
					if (Intersection(face, startpoint, halfsize, V)) {
						children_[ind]->F_.push_back(F_[face]);
						children_[ind]->Find_.push_back(Find_[face]);
						if (children_[ind]->occupied_ == 0) {
							children_[ind]->occupied_ = 1;
							number_ += 1;
							children_[ind]->number_ = 1;
						}
					}
				}
			}
		}
	}
	F_.clear();
	Find_.clear();
}

void Octree::BuildConnection()
{
	if (level_ == 0)
		return;
	for (int i = 0; i < 8; ++i)
	{
		if (children_[i])
		{
			children_[i]->BuildConnection();
		}
	}
	int y_index[] = {0, 1, 4, 5};
	for (int i = 0; i < 4; ++i)
	{
		if (children_[i * 2] && children_[i * 2 + 1])
			ConnectTree(children_[i * 2], children_[i * 2 + 1], 2);
		if (children_[y_index[i]] && children_[y_index[i] + 2])
			ConnectTree(children_[y_index[i]], children_[y_index[i] + 2], 1);
		if (children_[i] && children_[i + 4])
			ConnectTree(children_[i], children_[i + 4], 0);
	}
}

void Octree::ConnectTree(Octree* l, Octree* r, int dim)
{
	int y_index[] = {0, 1, 4, 5};
	if (dim == 2)
	{
		l->connection_[2] = r;
		r->connection_[5] = l;
		for (int i = 0; i < 4; ++i) {
			if (l->children_[i * 2 + 1] && r->children_[i * 2]) {
				ConnectTree(l->children_[i * 2 + 1], r->children_[i * 2], dim);
			}
		}
	}
	else if (dim == 1)
	{
		l->connection_[1] = r;
		r->connection_[4] = l;
		for (int i = 0; i < 4; ++i) {
			if (l->children_[y_index[i] + 2] && r->children_[y_index[i]]) {
				ConnectTree(l->children_[y_index[i] + 2],
					r->children_[y_index[i]], dim);
			}
		}
	}
	else if (dim == 0)
	{
		l->connection_[0] = r;
		r->connection_[3] = l;
		for (int i = 0; i < 4; ++i) {
			if (l->children_[i + 4] && r->children_[i]) {
				ConnectTree(l->children_[i + 4], r->children_[i], dim);
			}
		}
	}
}

void Octree::ConnectEmptyTree(Octree* l, Octree* r, int dim)
{
	int y_index[] = {0, 1, 4, 5};
	if (l->occupied_ && r->occupied_)
	{
		if (l->level_ == 0)
			return;
		if (dim == 2)
		{
			for (int i = 0; i < 4; ++i) {
				ConnectEmptyTree(l->children_[i * 2 + 1],
					r->children_[i * 2], dim);
			}
		}
		else if (dim == 1)
		{
			for (int i = 0; i < 4; ++i) {
				ConnectEmptyTree(l->children_[y_index[i] + 2],
					r->children_[y_index[i]], dim);
			}
		}
		else if (dim == 0)
		{
			for (int i = 0; i < 4; ++i) {
				ConnectEmptyTree(l->children_[i + 4], r->children_[i], dim);
			}
		}			
		return;
	}
	if (!(l->occupied_ || r->occupied_))
	{
		l->empty_neighbors_.push_back(r);
		r->empty_neighbors_.push_back(l);
		return;
	}
	if (!l->occupied_)
	{
		if (dim == 2)
		{
			r->empty_connection_[5] = l;
			if (r->level_ > 0)
			{
				for (int i = 0; i < 4; ++i)
				{
					ConnectEmptyTree(l, r->children_[i * 2], dim);
				}
			}
		}
		else if (dim == 1)
		{
			r->empty_connection_[4] = l;
			if (r->level_ > 0)
			{
				for (int i = 0; i < 4; ++i)
				{
					ConnectEmptyTree(l, r->children_[y_index[i]], dim);
				}
			}
		}
		else if (dim == 0)
		{
			r->empty_connection_[3] = l;
			if (r->level_ > 0)
			{
				for (int i = 0; i < 4; ++i)
				{
					ConnectEmptyTree(l, r->children_[i], dim);
				}
			}
		}
		return;
	}
	if (!r->occupied_)
	{
		if (dim == 2)
		{
			l->empty_connection_[2] = r;
			if (l->level_ > 0)
			{
				for (int i = 0; i < 4; ++i)
				{
					ConnectEmptyTree(l->children_[i * 2 + 1], r, dim);
				}
			}
		}
		else if (dim == 1)
		{
			l->empty_connection_[1] = r;
			if (l->level_ > 0)
			{
				for (int i = 0; i < 4; ++i)
				{
					ConnectEmptyTree(l->children_[y_index[i] + 2], r, dim);
				}
			}
		}
		else if (dim == 0)
		{
			l->empty_connection_[0] = r;
			if (l->level_ > 0)
			{
				for (int i = 0; i < 4; ++i)
				{
					ConnectEmptyTree(l->children_[i + 4], r, dim);
				}
			}
		}
	}
}


void Octree::ExpandEmpty(std::list<Octree*>& empty_list,
	std::set<Octree*>& empty_set, int dim)
{
	if (!occupied_)
	{
		if (empty_set.find(this) == empty_set.end())
		{
			empty_set.insert(this);
			empty_list.push_back(this);
		}
		return;
	}
	if (level_ == 0)
		return;
	int y_index[] = {0, 1, 4, 5};
	if (dim == 2 || dim == 5)
	{
		for (int i = 0; i < 4; ++i)
		{
			children_[i * 2 + (dim == 5)]->ExpandEmpty(
				empty_list, empty_set, dim);
		}
		return;
	}
	if (dim == 1 || dim == 4)
	{
		for (int i = 0; i < 4; ++i)
		{
			children_[y_index[i] + 2 * (dim == 4)]->ExpandEmpty(
				empty_list, empty_set, dim);
		}
		return;
	}
	for (int i = 0; i < 4; ++i)
	{
		children_[i + 4 * (dim == 3)]->ExpandEmpty(
			empty_list, empty_set, dim);
	}
}

void Octree::BuildEmptyConnection()
{
	if (level_ == 0)
		return;

	for (int i = 0; i < 8; ++i)
	{
		if (children_[i]->occupied_)
		{
			children_[i]->BuildEmptyConnection();
		}
	}
	int pair_x[] = {0,2,4,6,0,1,4,5,0,1,2,3};
	int pair_y[] = {1,3,5,7,2,3,6,7,4,5,6,7};
	int dim[] = {2,2,2,2,1,1,1,1,0,0,0,0};
	for (int i = 0; i < 12; ++i)
	{
		ConnectEmptyTree(children_[pair_x[i]], children_[pair_y[i]], dim[i]);
	}
}

void Octree::ConstructFace(const Vector3i& start,
	std::map<GridIndex,int>* vcolor,
	std::vector<Vector3>* vertices,
	std::vector<Vector4i>* faces,
	std::vector<std::set<int> >* v_faces)
{
	if (level_ == 0)
	{
		if (!occupied_)
			return;
		Vector3i offset[6][4] = {
			{Vector3i(1,0,0),Vector3i(1,0,1),Vector3i(1,1,1),Vector3i(1,1,0)},
			{Vector3i(0,1,0),Vector3i(1,1,0),Vector3i(1,1,1),Vector3i(0,1,1)},
			{Vector3i(0,0,1),Vector3i(0,1,1),Vector3i(1,1,1),Vector3i(1,0,1)},
			{Vector3i(0,0,0),Vector3i(0,1,0),Vector3i(0,1,1),Vector3i(0,0,1)},
			{Vector3i(0,0,0),Vector3i(0,0,1),Vector3i(1,0,1),Vector3i(1,0,0)},
			{Vector3i(0,0,0),Vector3i(1,0,0),Vector3i(1,1,0),Vector3i(0,1,0)}};

		for (int i = 0; i < 6; ++i)
		{
			if (empty_connection_[i] && empty_connection_[i]->exterior_)
			{
				int id[4];
				for (int j = 0; j < 4; ++j)
				{
					Vector3i vind = start + offset[i][j];
					GridIndex v_id;
					v_id.id = vind * 2;
					std::map<GridIndex,int>::iterator it = vcolor->find(v_id);
					if (it == vcolor->end())
					{
						Vector3 d = min_corner_;
						for (int k = 0; k < 3; ++k)
							d[k] += offset[i][j][k] * volume_size_[k];
						vcolor->insert(std::make_pair(v_id, vertices->size()));
						id[j] = vertices->size();
						vertices->push_back(d);
						v_faces->push_back(std::set<int>());
						for (std::vector<int>::iterator it1 = Find_.begin();
							it1 != Find_.end(); ++it1)
							(*v_faces)[id[j]].insert(*it1);
					}
					else {
						id[j] = it->second;
						for (std::vector<int>::iterator it1 = Find_.begin();
							it1 != Find_.end(); ++it1)
							(*v_faces)[it->second].insert(*it1);
					}
				}
				faces->push_back(Vector4i(id[0],id[1],id[2],id[3]));
			}
		}
	}
	else {
		for (int i = 0; i < 8; ++i)
		{
			if (children_[i] && children_[i]->occupied_)
			{
				int x = i / 4;
				int y = (i - x * 4) / 2;
				int z = i - x * 4 - y * 2; 
				Vector3i nstart = start * 2 + Vector3i(x,y,z);
				children_[i]->ConstructFace(nstart, vcolor,
					vertices, faces, v_faces);
			}
		}
	}
}