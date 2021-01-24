#ifndef MANIFOLD2_MESH_PROJECTOR_H_
#define MANIFOLD2_MESH_PROJECTOR_H_

#include <vector>
#include <igl/AABB.h>

#include "types.h"

class MeshProjector
{
public:
	MeshProjector();
	void ComputeHalfEdge();
	void ComputeIndependentSet();
	void UpdateFaceNormal(int i);
	void UpdateVertexNormal(int i, int conservative);
	void UpdateVertexNormals(int conservative);
	void IterativeOptimize(FT len, bool initialized = false);
	void AdaptiveRefine(FT len, FT ratio = 0.1);
	void EdgeFlipRefine(std::vector<int>& candidates);
	void Project(const MatrixD& V, const MatrixI& F,
		MatrixD* out_V, MatrixI* out_F);
	void UpdateNearestDistance();
	int BoundaryCheck();
	void SplitVertices();
	void OptimizePosition(int v, const Vector3& target_p, FT len, bool debug=false);
	void OptimizeNormal(int i, const Vector3& vn, const Vector3& target_vn);
	void OptimizeNormals();
	void PreserveSharpFeatures(FT len_thres);
	void Highlight(int id, FT len);
	void Sanity(const char* log);

	bool IsNeighbor(int v1, int v2);
private:
	std::vector<std::vector<int> > vertex_groups_;

	igl::AABB<MatrixD,3> tree_;
	MatrixD V_, out_V_, target_V_, out_N_, out_FN_;
	MatrixI F_, out_F_;
	VectorXi V2E_, E2E_;

	VectorX sqrD_;
	VectorXi I_;

	std::vector<int> sharp_vertices_;
	std::vector<Vector3> sharp_positions_;

	int num_V_, num_F_;

	std::vector<int> active_vertices_, active_vertices_temp_;
	std::vector<std::pair<FT, int> > indices_;
	int num_active_;
};
#endif