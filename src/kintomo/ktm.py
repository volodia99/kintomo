from itertools import product
from dataclasses import dataclass
from enum import StrEnum, auto

import numpy as np
from skimage import measure
import astropy.units as u
import astropy.constants as uc
import gpgi

from kintomo._typing import FArray1D, FArray2D

class Geometry(StrEnum):
    CARTESIAN = auto()
    SPHERICAL = auto()
    CYLINDRICAL = auto()

@dataclass(kw_only=True, frozen=True, slots=True)
class Velocity:
    geometry: Geometry
    v1: FArray1D
    v2: FArray1D
    v3: FArray1D

    def __add__(self, other_velocity:"Velocity") -> "Velocity":
        if self.geometry!=other_velocity.geometry:
            raise ValueError(f"{self.geometry}!={other_velocity.geometry}. Should be the same to add them")
        return Velocity(
            geometry=self.geometry,
            v1=np.append(self.v1, other_velocity.v1),
            v2=np.append(self.v2, other_velocity.v2),
            v3=np.append(self.v3, other_velocity.v3),
        )

    def to(self, *, unit:u.Unit=u.dimensionless_unscaled) -> "Velocity":
        if self.geometry is not Geometry.CARTESIAN:
            raise ValueError(f"{self.geometry=} should be cartesian before using the 'to' conversion method")
        if u.get_physical_type(unit) not in ("velocity", "dimensionless"):
            raise ValueError(f"unit: {u.get_physical_type(unit)} should be a 'velocity' or a 'dimensionless' quantity.")
        return Velocity(
            geometry=Geometry.CARTESIAN, 
            v1=self.v1*unit, 
            v2=self.v2*unit, 
            v3=self.v3*unit,
        )

    def _remove_unit(self) -> "Velocity":
        if self.geometry is not Geometry.CARTESIAN:
            raise ValueError(f"{self.geometry=} should be cartesian before using the '_remove_unit' method")
        if not(u.get_physical_type(self.v1)==u.get_physical_type(self.v2)==u.get_physical_type(self.v3)):
            raise ValueError("The velocity components do not have the same unit.")
        if (u.get_physical_type(self.v1)!="velocity"):
            raise ValueError(f"unit:{u.get_physical_type(self.v1)}. Cannot use the '_remove_unit' method on a quantity!='velocity'.")
        return Velocity(
            geometry=Geometry.CARTESIAN, 
            v1=self.v1.value, 
            v2=self.v2.value, 
            v3=self.v3.value,
        )

    @property
    def get_unit(self) -> str:
        if self.geometry is not Geometry.CARTESIAN:
            raise ValueError(f"{self.geometry=} should be cartesian before using the '_remove_unit' method")
        if not(u.get_physical_type(self.v1)==u.get_physical_type(self.v2)==u.get_physical_type(self.v3)):
            raise ValueError("The velocity components do not have the same unit.")
        if u.get_physical_type(self.v1) not in ("velocity", "dimensionless"):
            raise ValueError(f"unit: {u.get_physical_type(self.v1)} should be a 'velocity' or a 'dimensionless' quantity.")
        if u.get_physical_type(self.v1)=="velocity":
            unit = self.v1.unit
        elif u.get_physical_type(self.v1)=="dimensionless":
            unit = "c.u."
        return unit

    # TODO: as a __post_init__ to automatically convert to cartesian ?
    def to_cartesian(self, *, phi:FArray1D|None=None, theta:FArray1D|None=None) -> "Velocity":
        if self.geometry is Geometry.CARTESIAN:
            vx = self.v1
            vy = self.v2
            vz = self.v3
        elif self.geometry is Geometry.CYLINDRICAL:
            if phi is None:
                raise ValueError(f"{phi=} should be defined")
            vx = self.v1*np.cos(phi) - self.v2*np.sin(phi)
            vy = self.v1*np.sin(phi) + self.v2*np.cos(phi)
            vz = self.v3
        elif self.geometry is Geometry.SPHERICAL:
            if phi is None:
                raise ValueError(f"{phi=} should be defined")
            if theta is None:
                raise ValueError(f"{theta=} should be defined")
            vx = self.v1*np.sin(theta)*np.cos(phi) + self.v2*np.cos(theta)*np.cos(phi) - self.v3*np.sin(phi)
            vy = self.v1*np.sin(theta)*np.sin(phi) + self.v2*np.cos(theta)*np.sin(phi) + self.v3*np.cos(phi)
            vz = self.v1*np.cos(theta) - self.v2*np.sin(theta)
        else:
            raise ValueError(f"{self.geometry=} should be cartesian, cylindrical or spherical before using the 'to_cartesian' method")

        return Velocity(
            geometry=Geometry.CARTESIAN, 
            v1=vx, 
            v2=vy, 
            v3=vz,
        )

    @property
    def components(self):
        return self.v1, self.v2, self.v3

    @property
    def norm(self) -> FArray1D:
        return np.sqrt(self.v1**2+self.v2**2+self.v3**2)

    def shift(self, *, get:str|None=None) -> float:
        if get is None:
            raise ValueError(f"'get' argument is None, choose get='mean' or get='max'")
        newvel = self
        if (u.get_physical_type(self.v1)==u.get_physical_type(self.v2)==u.get_physical_type(self.v3)) and (u.get_physical_type(self.v1)=="velocity"):
            newvel = self._remove_unit()
        if get=="mean":
            return newvel.norm.mean()
        elif get=="max":
            return newvel.norm.max()
        else:
            raise ValueError(f"Unknown {get=} argument, should be 'mean' or 'max'.")

    @property
    def stacked_3d(self) -> FArray2D:
        return(np.vstack((self.v1, self.v2, self.v3)).T)

    def projected_on_sky(self, *, angle:float) -> "Velocity":
        # Project onto the plane perpendicular to the line of sight
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [0, np.sin(np.radians(angle)), np.cos(np.radians(angle))]
        ])
        projected_velocities = np.dot(self.stacked_3d, rotation_matrix.T)
        vxproj, vyproj, vzproj = projected_velocities.T
        return Velocity(
            geometry=Geometry.CARTESIAN, 
            v1=vxproj, 
            v2=vyproj, 
            v3=vzproj,
        )

    @classmethod
    def keplerian(cls, *, r:FArray1D) -> "Velocity":
        if (u.get_physical_type(r)=="length"):
            r = r.value
        geometry = Geometry.CYLINDRICAL
        v1 = np.zeros_like(r)
        v2 = np.sqrt(1/r)
        v3 = np.zeros_like(r)
        return Velocity(
            geometry=geometry, 
            v1=v1, 
            v2=v2, 
            v3=v3,
        )

    def visible(self, *, ds, method:str="ngp") -> FArray2D:
        particle_v = ds.deposit(
            "velocity", 
            weight_field="mass",
            method=method,
        )
        particle_n = ds.deposit(
            "mass",
            method=method,
        )
        nx, ny = particle_v.shape[:2]
        index_thick = np.zeros((nx,ny))
        v_visible_thick = np.zeros((nx,ny))
        for ix, iy in product(np.arange(nx), np.arange(ny)):
            index_thick[ix,iy] = next((i for i, x in enumerate(particle_n[ix,iy,:]) if x), 0.0)
            v_visible_thick[ix,iy] = particle_v[ix,iy,int(index_thick[ix,iy])]
        return v_visible_thick

@dataclass(kw_only=True, frozen=True, slots=True)
class Sculpture:
    x: FArray1D
    y: FArray1D
    z: FArray1D

    def __add__(self, other_sculpture):
        return Sculpture(
            x=np.append(self.x, other_sculpture.x),
            y=np.append(self.y, other_sculpture.y),
            z=np.append(self.z, other_sculpture.z),
        )

    def to(self, unit:u.Unit=u.dimensionless_unscaled) -> "Sculpture":
        if u.get_physical_type(unit) not in ("length", "dimensionless"):
            raise ValueError(f"unit: {u.get_physical_type(unit)} should be a 'length' or a 'dimensionless' quantity.")
        return Sculpture(
            x=self.x*unit, 
            y=self.y*unit, 
            z=self.z*unit,
        )

    def _remove_unit(self) -> "Sculpture":
        if not(u.get_physical_type(self.x)==u.get_physical_type(self.y)==u.get_physical_type(self.z)):
            raise ValueError("The velocity is not cartesian")
        if (u.get_physical_type(self.x)!="length"):
            raise ValueError(f"unit:{u.get_physical_type(self.x)}. Cannot use the '_remove_unit' method on a quantity!='length'.")
        return Sculpture(
            x=self.x.value, 
            y=self.y.value, 
            z=self.z.value,
        )

    @property
    def get_unit(self) -> str:
        if not(u.get_physical_type(self.x)==u.get_physical_type(self.y)==u.get_physical_type(self.z)):
            raise ValueError("The velocity is not cartesian and should be.")
        if u.get_physical_type(self.x) not in ("length", "dimensionless"):
            raise ValueError(f"unit: {u.get_physical_type(self.x)} should be a 'length' or a 'dimensionless' quantity.")
        if u.get_physical_type(self.x)=="length":
            unit = self.x.unit
        elif u.get_physical_type(self.x)=="dimensionless":
            unit = "c.u."
        return unit

    @classmethod
    def cube(cls, num_points:int) -> "Sculpture":
        x0, y0, z0 = (2 * np.random.rand(num_points) - 1 for _ in range(3))
        return Sculpture(
            x=x0, 
            y=y0, 
            z=z0,
        )

    @property
    def coordinates(self) -> dict[str, FArray1D]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }

    def cartesian_coordinates(self, *, x_offset:float|None=None, y_offset:float|None=None, z_offset:float|None=None) -> tuple[FArray1D, FArray1D, FArray1D]:
        xcoord = self.x
        ycoord = self.y
        zcoord = self.z
        if x_offset is not None:
            xcoord = self.x - x_offset
        if y_offset is not None:
            ycoord = self.y - y_offset
        if z_offset is not None:
            zcoord = self.z - z_offset
        return xcoord, ycoord, zcoord

    def cylindrical_coordinates(self, *, x_offset:float|None=None, y_offset:float|None=None, z_offset:float|None=None) -> tuple[FArray1D, FArray1D, FArray1D]:
        xcoord = self.x
        ycoord = self.y
        zcoord = self.z
        if x_offset is not None:
            xcoord = self.x - x_offset
        if y_offset is not None:
            ycoord = self.y - y_offset
        if z_offset is not None:
            zcoord = self.z - z_offset
        R = np.sqrt(xcoord**2 + ycoord**2)
        phi = np.arctan2(ycoord, xcoord)
        return R, phi, zcoord

    def spherical_coordinates(self, *, x_offset:float|None=None, y_offset:float|None=None, z_offset:float|None=None) -> tuple[FArray1D, FArray1D, FArray1D]:
        xcoord = self.x
        ycoord = self.y
        zcoord = self.z
        if x_offset is not None:
            xcoord = self.x - x_offset
        if y_offset is not None:
            ycoord = self.y - y_offset
        if z_offset is not None:
            zcoord = self.z - z_offset
        r = np.sqrt(xcoord**2 + ycoord**2 + zcoord**2)
        theta = np.arccos(zcoord/r)
        phi = np.arctan2(ycoord, xcoord)
        return r, theta, phi

    @property
    def max_size_yz(self) -> FArray1D:
        return np.sqrt(self.y**2 + self.z**2).max()

    @property
    def stacked_3d(self) -> FArray2D:
        return(np.vstack((self.x, self.y, self.z)).T)

    def projected_on_sky(self, *, angle:float) -> "Sculpture":
        # Project onto the plane perpendicular to the line of sight
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [0, np.sin(np.radians(angle)), np.cos(np.radians(angle))]
        ])
        projected_coordinates = np.dot(self.stacked_3d, rotation_matrix.T)
        xproj, yproj, zproj = projected_coordinates.T
        return Sculpture(
            x=xproj, 
            y=yproj, 
            z=zproj,
        )

    def carve(self, *, shape:"Shape") -> "Sculpture":
        return Sculpture(
            x=self.x[shape.extracted_array], 
            y=self.y[shape.extracted_array], 
            z=self.z[shape.extracted_array],
        )

@dataclass(frozen=True, slots=True)
class Shape:
    extracted_array: FArray1D

    @classmethod
    def cylinder(cls, *, x:FArray1D, y:FArray1D, z:FArray1D, rmin:float|None=None, rmax:float, height:float) -> "Shape":
        if rmin is None:
            rmin = 0.0
        extracted_array = ((x**2 + y**2 < rmax**2) & (x**2 + y**2 > rmin**2) & (abs(z) < height))
        return Shape(extracted_array)

@dataclass(kw_only=True, frozen=True, slots=True)
class Grid:
    xedge: FArray1D
    yedge: FArray1D
    zedge: FArray1D

    @property
    def xmed(self) -> FArray1D:
        return 0.5*(self.xedge[1:]+self.xedge[:-1])

    @property
    def ymed(self) -> FArray1D:
        return 0.5*(self.yedge[1:]+self.yedge[:-1])

    @property
    def zmed(self) -> FArray1D:
        return 0.5*(self.zedge[1:]+self.zedge[:-1])

    @property
    def cell_edges(self) -> dict[str, FArray1D]:
        return {
            "x": self.xedge,
            "y": self.yedge,
            "z": self.zedge,
        }

    @property
    def cell_centers(self) -> dict[str, FArray1D]:
        return {
            "x": self.xmed,
            "y": self.ymed,
            "z": self.zmed,
        }

    @classmethod
    def encompass(cls, *, sculpture:"Sculpture", dimension:tuple[int, int, int]) -> "Grid":
        if (u.get_physical_type(sculpture.x)==u.get_physical_type(sculpture.y)==u.get_physical_type(sculpture.z)) and (u.get_physical_type(sculpture.x)=="length"):
            sculpture = sculpture._remove_unit()
        return Grid(
            xedge = np.linspace(2*sculpture.x.min(), 2*sculpture.x.max(), dimension[0]+1),
            yedge = np.linspace(-2*sculpture.max_size_yz, 2*sculpture.max_size_yz, dimension[1]+1),
            zedge = np.linspace(-2*sculpture.max_size_yz, 2*sculpture.max_size_yz, dimension[2]+1),
        )

    def deposition(self, *, sculpture:"Sculpture", velocity:"Velocity"):
        if (u.get_physical_type(sculpture.x)==u.get_physical_type(sculpture.y)==u.get_physical_type(sculpture.z)) and (u.get_physical_type(sculpture.x)=="length"):
            sculpture = sculpture._remove_unit()
        return (
            gpgi.load(
                geometry="cartesian",
                grid={
                    "cell_edges": self.cell_edges,
                    "cell_centers": self.cell_centers,
                },
                particles={
                    "coordinates": sculpture.coordinates,
                    "fields": {
                        "mass": np.ones(len(sculpture.z)),
                        "velocity": velocity.v3,
                    },
                },
            )
        )

#TODO: let it as a rogue function? or as a method of grid/sculpture/velocity?
def projection_deposition_visible(*, grid:"Grid", sculpture:"Sculpture", velocity:"Velocity", angle:float, method:str="ngp") -> FArray2D:
    projected_sculpture = sculpture.projected_on_sky(angle=angle)
    projected_velocity = velocity.projected_on_sky(angle=angle)
    ds = grid.deposition(
        sculpture=projected_sculpture, 
        velocity=projected_velocity
    )
    v_visible = projected_velocity.visible(
        ds=ds,
        method=method,
    )
    return v_visible
