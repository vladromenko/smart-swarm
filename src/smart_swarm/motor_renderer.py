"""Conservative per-camera drawing on a separate motor-only model."""

import numpy as np

from .motor_visual_mesh import visual_copy
from .motor_visual_profile import COMPACT, STOCK, visual_binding
from .physics import World


def mesh_bounds(model):
    import mujoco

    identity = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    centers, extents, faces = [], [], []
    for geom in identity:
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        centers.append((low + high) / 2)
        extents.append((high - low) / 2)
        faces.append(int(model.mesh_facenum[mesh]))
    return {"geom": identity, "center": np.asarray(centers), "half_extent": np.asarray(extents),
            "face_count": np.asarray(faces)}


def camera_normals(camera, aspect):
    if camera.orthographic or camera.frustum_near <= 0:
        raise ValueError("This diagnostic requires the original perspective camera profiles")
    forward, up = camera.forward, camera.up
    right = np.cross(forward, up)
    half_width = (camera.frustum_top - camera.frustum_bottom) * aspect / 2
    left = (camera.frustum_center - half_width) / camera.frustum_near
    right_edge = (camera.frustum_center + half_width) / camera.frustum_near
    lower = camera.frustum_bottom / camera.frustum_near
    upper = camera.frustum_top / camera.frustum_near
    normals = np.stack((right - left * forward, right_edge * forward - right,
                        up - lower * forward, upper * forward - up))
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def excluded(world, renderer, bounds):
    import mujoco

    scene = renderer.scene
    geoms = scene.geoms[:scene.ngeom]
    shadows = scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] and any(light.castshadow for light in scene.lights[:scene.nlight])
    reflections = scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] and any(geom.reflectance > 0 for geom in geoms)
    if shadows or reflections:
        raise ValueError("Offscreen geometry can affect a shadow or reflection in this scene")
    ids = bounds["geom"]
    rendered = [geom.objid for geom in geoms
                if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.type == mujoco.mjtGeom.mjGEOM_MESH]
    rotations = world.data.geom_xmat[ids].reshape(-1, 3, 3)
    center = world.data.geom_xpos[ids] + np.einsum("gij,gj->gi", rotations, bounds["center"])
    invisible = np.isin(ids, rendered)
    for camera in scene.camera:
        normals = camera_normals(camera, renderer.width / renderer.height)
        mean = (center - camera.pos) @ normals.T
        axis = np.einsum("gij,pi->gpj", rotations, normals)
        radius = np.einsum("gpj,gj->gp", np.abs(axis), bounds["half_extent"])
        invisible &= np.any(mean + radius < -.005, axis=1)
    return ids[invisible], int(bounds["face_count"][invisible].sum())


def render_candidate(world, name, bounds, *, renderer=None):
    import mujoco

    renderer = world.renderer if renderer is None else renderer
    renderer.update_scene(world.data, camera=name)
    ids, faces = excluded(world, renderer, bounds)
    options = mujoco.MjvOption()
    hidden, = [index for index in range(len(options.geomgroup) - 1, -1, -1) if not options.geomgroup[index]][:1]
    group = renderer.model.geom_group
    original = group[ids].copy()
    try:
        group[ids] = hidden
        renderer.update_scene(world.data, camera=name, scene_option=options)
    finally:
        group[ids] = original
    return renderer.render(), faces


class MotorWorld(World):
    """Share the physical world and use the explicitly bound two-camera drawing path."""

    def __init__(self, *args, visual_profile=STOCK, **kwargs):
        self.motor_visual = visual_binding(visual_profile)
        super().__init__(*args, **kwargs)

    def image(self, camera):
        if self.motor_visual["visual_profile"] != COMPACT:
            return super().image(camera)
        if self.renderer is None:
            import mujoco

            drawing, _ = visual_copy(self.model)
            self.mesh_bounds = mesh_bounds(drawing)
            width, height = self.frame_size
            self.renderer = mujoco.Renderer(drawing, height=height, width=width)
        picture, _ = render_candidate(self, camera, self.mesh_bounds)
        return picture.copy()
