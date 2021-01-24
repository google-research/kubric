#ifndef MANIFOLD2_IO_H_
#define MANIFOLD2_IO_H_

#include "types.h"

void ReadOBJ(const char* filename, MatrixD* V, MatrixI* F);
void WriteOBJ(const char* filename, const MatrixD& V, const MatrixI& F);

#endif