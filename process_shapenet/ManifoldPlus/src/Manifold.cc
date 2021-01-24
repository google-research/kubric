#include "Manifold.h"

#include <Eigen/Dense>

#include "MeshProjector.h"

Manifold::Manifold()
	: tree_(0)
{
}

Manifold::~Manifold()
{
	if (tree_)
		delete tree_;
	tree_ = 0;
}

void Manifold::ProcessManifold(const MatrixD& V, const MatrixI& F,
	int depth, MatrixD* out_V, MatrixI* out_F)
{
	V_ = V;
	F_ = F;

	BuildTree(depth);
	ConstructManifold();

	*out_V = MatrixD(vertices_.size(), 3);
	*out_F = MatrixI(face_indices_.size(), 3);
	for (int i = 0; i < vertices_.size(); ++i)
		out_V->row(i) = vertices_[i];
	for (int i = 0; i < face_indices_.size(); ++i)
		out_F->row(i) = face_indices_[i];

	MeshProjector projector;
	projector.Project(V_, F_, out_V, out_F);
}

void Manifold::BuildTree(int depth)
{
	CalcBoundingBox();
	tree_ = new Octree(min_corner_, max_corner_, F_);

	for (int iter = 0; iter < depth; ++iter) {
		tree_->Split(V_);
	}

	tree_->BuildConnection();
	tree_->BuildEmptyConnection();

	std::list<Octree*> empty_list;
	std::set<Octree*> empty_set;
	for (int i = 0; i < 6; ++i)
	{
		tree_->ExpandEmpty(empty_list, empty_set, i);
	}

	while ((int)empty_list.size() > 0)
	{
		Octree* empty = empty_list.front();
		empty->exterior_ = 1;
		for (std::list<Octree*>::iterator it = empty->empty_neighbors_.begin();
			it != empty->empty_neighbors_.end(); ++it)
		{
			if (empty_set.find(*it) == empty_set.end())
			{
				empty_list.push_back(*it);
				empty_set.insert(*it);
			}
		}
		empty_list.pop_front();
	}
}

void Manifold::CalcBoundingBox()
{
	min_corner_ = Vector3(1e30,1e30,1e30);
	max_corner_ = -min_corner_;
	for (int i = 0; i < (int)V_.rows(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			if (V_(i, j) < min_corner_[j])
			{
				min_corner_[j] = V_(i, j);
			}
			if (V_(i, j) > max_corner_[j])
			{
				max_corner_[j] = V_(i, j);
			}
		}
	}
	Vector3 volume_size = max_corner_ - min_corner_;
	for (int i = 0; i < 3; ++i) {
		volume_size[i] = std::max(volume_size[i], 1e-3);
	}
	// make sure no precision problem when doing intersection test
	double weird_number[3][2] = {
		{0.1953725, 0.1947674},
		{0.1975733, 0.1936563},
		{0.1957376, 0.1958437}
	};
	for (int i = 0; i < 3; ++i) {
		min_corner_[i] -= volume_size[i] * weird_number[i][0];
		max_corner_[i] += volume_size[i] * weird_number[i][1];
	}
}

void Manifold::ConstructManifold()
{
	std::map<GridIndex,int> vcolor;
	std::vector<Vector3> nvertices;
	std::vector<Vector4i> nface_indices;
	std::vector<Vector3i> triangles;
	std::vector<std::set<int> > v_faces;
	tree_->ConstructFace(Vector3i(0,0,0), &vcolor, &nvertices,
		&nface_indices, &v_faces);

	SplitGrid(nface_indices, vcolor, nvertices, v_faces, triangles);
	std::vector<int> hash_v(nvertices.size(),0);
	for (int i = 0; i < (int)triangles.size(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			hash_v[triangles[i][j]] = 1;
		}
	}
	vertices_.clear();
	for (int i = 0; i < (int)hash_v.size(); ++i)
	{
		if (hash_v[i])
		{
			hash_v[i] = (int)vertices_.size();
			//v_faces[vertices_.size()] = v_faces[i];
			//v_info_[vertices_.size()] = v_info_[i];
			vertices_.push_back(nvertices[i]);
		}
	}
	for (int i = 0; i < (int)triangles.size(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			triangles[i][j] = hash_v[triangles[i][j]];
		}
	}
	face_indices_ = triangles;
}

bool Manifold::SplitGrid(
	const std::vector<Vector4i>& nface_indices,
	std::map<GridIndex,int>& vcolor,
	std::vector<Vector3>& nvertices,
	std::vector<std::set<int> >& v_faces,
	std::vector<Vector3i>& triangles)
{
	FT unit_len = 0;
	v_info_.resize(vcolor.size());
	for (auto it = vcolor.begin();
		it != vcolor.end(); ++it)
	{
		v_info_[it->second] = it->first;
	}
	std::set<int> marked_v;
	std::map<std::pair<int, int>, std::list<std::pair<int, int> > > edge_info;
	for (int i = 0; i < (int)nface_indices.size(); ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			int x = nface_indices[i][j];
			int y = nface_indices[i][(j + 1) % 4];
			if (x > y)
			{
				int temp = x;
				x = y;
				y = temp;
			}
			std::pair<int,int> edge = std::make_pair(x,y);
			auto it = edge_info.find(edge);
			if (it != edge_info.end())
			{
				it->second.push_back(std::make_pair(i,j));
			} else
			{
				std::list<std::pair<int,int> > buf;
				buf.push_back(std::make_pair(i,j));
				edge_info.insert(std::make_pair(edge, buf));
			}
		}
	}
	for (auto it = edge_info.begin();
		it != edge_info.end(); ++it)
	{
		if (it->second.size() > 2) {
			marked_v.insert(it->first.first);
			marked_v.insert(it->first.second);
		}
	}
	triangles.clear();
	FT half_len = (nvertices[nface_indices[0][1]]
		- nvertices[nface_indices[0][0]]).norm() * 0.5;
	for (int i = 0; i < (int)nface_indices.size(); ++i)
	{
		int t = 0;
		while (t < 4 && marked_v.find(nface_indices[i][t]) == marked_v.end())
			++t;
		if (t == 4)
		{
			triangles.push_back(Vector3i(nface_indices[i][0],
				nface_indices[i][2], nface_indices[i][1]));
			triangles.push_back(Vector3i(nface_indices[i][0],
				nface_indices[i][3], nface_indices[i][2]));
			continue;
		}
		int ind[4];
		for (int j = 0; j < 4; ++j)
			ind[j] = nface_indices[i][(t+j)%4];
		bool flag1 = marked_v.find(ind[1]) != marked_v.end();
		bool flag2 = marked_v.find(ind[2]) != marked_v.end();
		bool flag3 = marked_v.find(ind[3]) != marked_v.end();
		GridIndex pt1 = (v_info_[ind[0]] + v_info_[ind[1]]) / 2;
		GridIndex pt2 = (v_info_[ind[0]] + v_info_[ind[3]]) / 2;
		GridIndex pt3 = (v_info_[ind[2]] + v_info_[ind[3]]) / 2;
		GridIndex pt4 = (v_info_[ind[1]] + v_info_[ind[2]]) / 2;
		int ind1, ind2, ind3, ind4;
		auto it = vcolor.find(pt1);
		if (it == vcolor.end())
		{
			vcolor.insert(std::make_pair(pt1,nvertices.size()));
			v_info_.push_back(pt1);
			ind1 = (int)nvertices.size();
			nvertices.push_back((nvertices[ind[0]]+nvertices[ind[1]])*0.5);
			v_faces.push_back(v_faces[ind[0]]);
		}
		else {
			ind1 = it->second;
		}
		it = vcolor.find(pt2);
		if (it == vcolor.end())
		{
			vcolor.insert(std::make_pair(pt2,nvertices.size()));
			v_info_.push_back(pt2);
			ind2 = (int)nvertices.size();
			v_faces.push_back(v_faces[ind[0]]);
			nvertices.push_back((nvertices[ind[0]]+nvertices[ind[3]])*0.5);
		} else {
			ind2 = it->second;
		}
		if (flag1 || flag2)
		{
			it = vcolor.find(pt4);
			if (it == vcolor.end())
			{
				vcolor.insert(std::make_pair(pt4,nvertices.size()));
				v_info_.push_back(pt4);
				ind4 = (int)nvertices.size();
				nvertices.push_back((nvertices[ind[1]]+nvertices[ind[2]])*0.5);
				if (flag1)
					v_faces.push_back(v_faces[ind[1]]);
				else
					v_faces.push_back(v_faces[ind[2]]);
			} else
			ind4 = it->second;
		}
		if (flag2 || flag3)
		{
			it = vcolor.find(pt3);
			if (it == vcolor.end())
			{
				vcolor.insert(std::make_pair(pt3,nvertices.size()));
				v_info_.push_back(pt3);
				ind3 = (int)nvertices.size();
				nvertices.push_back((nvertices[ind[2]]+nvertices[ind[3]])*0.5);
				if (flag2)
					v_faces.push_back(v_faces[ind[2]]);
				else
					v_faces.push_back(v_faces[ind[3]]);
			} else
			ind3 = it->second;			
		}
		if (!flag1 && !flag2 && !flag3)
		{
			triangles.push_back(Vector3i(ind1,ind[2],ind[1]));
			triangles.push_back(Vector3i(ind2,ind[2],ind1));
			triangles.push_back(Vector3i(ind[3],ind[2],ind2));
		} else
		if (!flag1 && !flag2 && flag3)
		{
			triangles.push_back(Vector3i(ind1,ind2,ind3));
			triangles.push_back(Vector3i(ind1,ind3,ind[2]));
			triangles.push_back(Vector3i(ind1,ind[2],ind[1]));			
		} else
		if (!flag1 && flag2 && !flag3)
		{			
			triangles.push_back(Vector3i(ind1,ind4,ind[1]));
			triangles.push_back(Vector3i(ind1,ind2,ind4));
			triangles.push_back(Vector3i(ind2,ind[3],ind3));			
			triangles.push_back(Vector3i(ind2,ind3,ind4));			
		} else
		if (!flag1 && flag2 && flag3)
		{			
			triangles.push_back(Vector3i(ind1,ind4,ind[1]));
			triangles.push_back(Vector3i(ind1,ind2,ind4));
			triangles.push_back(Vector3i(ind2,ind3,ind4));			
		} else
		if (flag1 && !flag2 && !flag3)
		{			
			triangles.push_back(Vector3i(ind1,ind2,ind4));
			triangles.push_back(Vector3i(ind4,ind2,ind[3]));
			triangles.push_back(Vector3i(ind4,ind[3],ind[2]));
		} else
		if (flag1 && !flag2 && flag3)
		{			
			triangles.push_back(Vector3i(ind1,ind2,ind4));
			triangles.push_back(Vector3i(ind4,ind2,ind3));
			triangles.push_back(Vector3i(ind4,ind3,ind[2]));
		} else
		if (flag1 && flag2 && !flag3)
		{			
			triangles.push_back(Vector3i(ind1,ind2,ind4));
			triangles.push_back(Vector3i(ind2,ind3,ind4));
			triangles.push_back(Vector3i(ind2,ind[3],ind3));			
		} else
		if (flag1 && flag2 && flag3)
		{			
			triangles.push_back(Vector3i(ind1,ind2,ind3));
			triangles.push_back(Vector3i(ind1,ind3,ind4));
		}
	}
	for (auto it = marked_v.begin();
		it != marked_v.end(); ++it)
	{
		Vector3 p = nvertices[*it];
		for (int dimx = -1; dimx < 2; dimx += 2) {
			for (int dimy = -1; dimy < 2; dimy += 2) {
				for (int dimz = -1; dimz < 2; dimz += 2) {
					Vector3 p1 = p + Vector3(dimx * half_len,
						dimy * half_len, dimz * half_len);
					if (tree_->IsExterior(p1))
					{
						GridIndex ind = v_info_[*it];
						GridIndex ind1 = ind;
						GridIndex ind2 = ind;
						GridIndex ind3 = ind;
						ind1.id[0] += dimx;
						ind2.id[1] += dimy;
						ind3.id[2] += dimz;
						if (vcolor.find(ind1) == vcolor.end())
						{
							vcolor.insert(std::make_pair(ind1,
								nvertices.size()));
							v_info_.push_back(ind1);

							nvertices.push_back(Vector3(
								p[0]+half_len*dimx,p[1], p[2]));
							v_faces.push_back(v_faces[*it]);
						}
						if (vcolor.find(ind2) == vcolor.end())
						{
							vcolor.insert(std::make_pair(ind2,
								nvertices.size()));
							v_info_.push_back(ind2);

							nvertices.push_back(Vector3(
								p[0],p[1]+half_len*dimy,p[2]));
							v_faces.push_back(v_faces[*it]);
						}
						if (vcolor.find(ind3) == vcolor.end())
						{
							vcolor.insert(std::make_pair(
								ind3, nvertices.size()));
							v_info_.push_back(ind3);

							nvertices.push_back(Vector3(
								p[0],p[1],p[2]+half_len*dimz));
							v_faces.push_back(v_faces[*it]);
						}
						int id1 = vcolor[ind1];
						int id2 = vcolor[ind2];
						int id3 = vcolor[ind3];
						Vector3 norm = (nvertices[id2]-nvertices[id1]).cross(
							nvertices[id3]-nvertices[id1]);
						if (norm.dot(Vector3(dimx,dimy,dimz)) < 0)
							triangles.push_back(Vector3i(id1,id3,id2));
						else
							triangles.push_back(Vector3i(id1,id2,id3));
					}
				}
			}
		}
	}
	std::map<int,std::set<std::pair<int,int> > > ocs, ecs;
	std::set<int> odds;
	std::set<int> evens;
	for (int i = 0; i < (int)nvertices.size(); ++i)
	{
		bool flag = false;
		for (int k = 0; k < 3; ++k)
			if (v_info_[i].id[k] % 2 == 1)
				flag = true;
		if (flag) {
			odds.insert(i);
			ocs.insert(std::make_pair(i,std::set<std::pair<int,int> >()));
		}
	}
	for (int i = 0; i < (int)nvertices.size(); ++i)
	{
		GridIndex ind = v_info_[i];
		int flag = 0;
		while (flag < 3 && ind.id[flag] % 2 == 0)
		{
			flag++;
		}
		if (flag < 3)
			continue;
		for (int j = -2; j < 5; j += 4)
		{
			if (flag < 3)
				break;
			for (int k = 0; k < 3; ++k)
			{
				GridIndex ind1 = ind;
				ind1.id[k] += j;
				auto it = vcolor.find(ind1);
				if (it == vcolor.end())
				{
					flag = 0;
					break;
				}
				int y = it->second;
				unit_len = (nvertices[y] - nvertices[i]).norm();
				std::pair<int,int> edge_id;
				if (i < y)
					edge_id = std::make_pair(i,y);
				else
					edge_id = std::make_pair(y,i);
				if (edge_info.find(edge_id) == edge_info.end())
				{
					flag = 0;
					break;
				}
			}
		}
		if (flag < 3)
			continue;
		evens.insert(i);
		ecs.insert(std::make_pair(i,std::set<std::pair<int,int> >()));
	}
	for (int i = 0; i < (int)triangles.size(); ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			int x = triangles[i][j];
			if (odds.find(x) != odds.end())
			{
				ocs[x].insert(std::make_pair(i,j));
			}
			if (evens.find(x) != evens.end())
			{
				ecs[x].insert(std::make_pair(i,j));
			}
		}
	}
	for (auto it = evens.begin();
		it != evens.end(); ++it)
	{
		int i = *it;
		Vector3 dir;
		int count = 0;
		for (int j = 0; j < 8; ++j)
		{
			Vector3 d((j&0x04)>0,(j&0x02)>0,(j&0x01)>0);
			d = d * 2.0 - Vector3(1,1,1);
			d = d.normalized() * (unit_len * 0.5);
			if (!tree_->IsExterior(nvertices[i] + d))
			{
				dir = d.normalized();
				count += 1;
			}
		}
		if (count > 2)
			continue;
		std::set<std::pair<int,int> >& p = ecs[i];
		for (auto it1 = p.begin();
			it1 != p.end(); ++it1)
		{
			if (dir.dot(nvertices[triangles[it1->first][(it1->second+1)%3]]
				-nvertices[i])<0)
			{
				triangles[it1->first][it1->second] = (int)nvertices.size();
			}
		}
		nvertices[i] += dir * (0.5 * unit_len);
		v_faces.push_back(v_faces[i]);
		nvertices.push_back(nvertices[i]);
		nvertices.back() -= unit_len * dir;

	}
	for (auto it = odds.begin();
		it != odds.end(); ++it)
	{
		int i = *it;
		int k = 0;
		while (v_info_[i].id[k] % 2 == 0)
			k += 1;
		GridIndex id1, id2;
		id1 = v_info_[i];
		id2 = v_info_[i];
		id1.id[k] -= 1;
		id2.id[k] += 1;
		int x = vcolor[id1];
		int y = vcolor[id2];
		if (x > y)
		{
			int temp = x;
			x = y;
			y = temp;
		}
		if (edge_info[std::make_pair(x,y)].size() > 2)
		{
			Vector3 vert = nvertices[x] - nvertices[y];
			FT len = vert.norm();
			vert /= len;
			Vector3 dir(len*0.5,len*0.5,len*0.5);
			dir = dir - dir.dot(vert)*vert;
			if (!tree_->IsExterior(nvertices[i]+dir))
			{
				dir = vert.cross(dir);
			}
			dir = dir.normalized();
			std::set<std::pair<int,int> >& p = ocs[i];
			for (auto it1 = p.begin();
				it1 != p.end(); ++it1)
			{
				if (dir.dot(nvertices[triangles[it1->first][(it1->second+1)%3]]
					-nvertices[i])<0)
				{
					triangles[it1->first][it1->second] = (int)nvertices.size();
				}
			}
			nvertices[i] += dir * (0.5 * len);
			v_faces.push_back(v_faces[i]);
			nvertices.push_back(nvertices[i]);
			nvertices.back() -= len * dir;
		}
	}
	return true;
}