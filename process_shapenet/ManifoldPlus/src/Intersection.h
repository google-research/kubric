#ifndef MANIFOLD2_INTERSECTION_H_
#define MANIFOLD2_INTERSECTION_H_

#include "types.h"
int TriBoxOverlap(float boxcenter[3],float boxhalfsize[3],float triverts[3][3]);

int PlaneIntersect(const Vector3& p0, const Vector3& n0,
  const Vector3& p1, const Vector3& n1,
  Vector3* o, Vector3* t);

#endif