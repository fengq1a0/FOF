from ._marching_cubes_lewiner import marching_cubes, mesh_surface_area
from .solver import smooth

def mc(volume, threshold):
    v,f,vn, not_z = marching_cubes(volume, threshold)
    smooth(v,f,not_z)
    return v, f