"""Build a lower-triangle drawing copy of the motor MuJoCo model."""

from copy import copy

import numpy as np

VISUAL_GRID = 32
MIN_ORIGINAL_FACES = 1200


def representatives(vertices, resolution):
    low = vertices.min(axis=0)
    span = float(np.max(vertices.max(axis=0) - low))
    if span <= 0:
        raise ValueError("A visual mesh has no spatial extent")
    quantized = np.floor((vertices - low) * (resolution / span)).astype(np.int32)
    quantized = np.clip(quantized, 0, resolution)
    width = resolution + 1
    cells = np.ravel_multi_index(quantized.T, (width,) * 3)
    count = np.bincount(cells)
    mean = np.stack([np.bincount(cells, weights=vertices[:, axis]) for axis in range(3)], axis=1)
    mean /= np.maximum(count[:, None], 1)
    error = np.sum((vertices - mean[cells]) ** 2, axis=1)
    index = np.lexsort((np.arange(len(vertices)), error, cells))
    ordered = cells[index]
    unique = np.r_[True, ordered[1:] != ordered[:-1]]
    by_cell = np.empty(len(count), dtype=np.int32)
    by_cell[ordered[unique]] = index[unique]
    return by_cell[cells]


def reduced_faces(vertices, faces):
    replacements = representatives(vertices, VISUAL_GRID)
    updated = replacements[faces]
    valid = np.flatnonzero(np.all(updated != np.roll(updated, 1, axis=1), axis=1))
    triangles = updated[valid]
    start = np.argmin(triangles, axis=1)
    order = (start[:, None] + np.arange(3)) % 3
    canonical = np.take_along_axis(triangles, order, axis=1)
    _, first = np.unique(canonical, axis=0, return_index=True)
    retained = np.sort(valid[first])
    return updated[retained], retained


def visual_copy(source):
    drawing = copy(source)
    reductions = []
    for mesh in range(source.nmesh):
        length = int(source.mesh_facenum[mesh])
        if length >= MIN_ORIGINAL_FACES:
            row = compress_mesh(source, drawing, mesh, length)
            if row is not None:
                reductions.append(row)
    return drawing, reductions


def compress_mesh(source, drawing, mesh, length):
    vertex = int(source.mesh_vertadr[mesh])
    face = int(source.mesh_faceadr[mesh])
    vertices = source.mesh_vert[vertex:vertex + int(source.mesh_vertnum[mesh])]
    old = slice(face, face + length)
    faces, indices = reduced_faces(vertices, source.mesh_face[old])
    size = len(faces)
    if size < 4 or size >= length:
        return None
    reduced = slice(face, face + size)
    drawing.mesh_face[reduced] = faces
    drawing.mesh_facenormal[reduced] = source.mesh_facenormal[old][indices]
    drawing.mesh_facetexcoord[reduced] = source.mesh_facetexcoord[old][indices]
    drawing.mesh_facenum[mesh] = size
    return {"mesh": source.mesh(mesh).name, "source_faces": length, "draw_faces": size}
