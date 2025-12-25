# kintomo

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

`kintomo` is a Python library to create optically thick kinematic toy models and explore their kinematics, given a user-defined structure and velocity, projected on a grid.

## Development status

`kintomo` is still in development, but we are trying to keep breaking changes to a minimum. Possible changes include:

- ENH: improve notebooks
- DOC: to be added.
- REL: publish to pipy
- RFC: keep `projection_deposition_visible` as a rogue function?
- ENH: add additional shapes as classmethods. 
- ENH: add optically-thin sculptures.
- RFC: use `__post_init__` method to automatically convert sculpture and velocity profile to cartesian.

## Installation

We recommend to install this repo using the package and project manager `uv`. See the [documentation](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) to install `uv` on your system. To install `kintomo`, run (***ongoing work***)

## Notebooks

Examples are provided in the `kintomo/notebooks` directory. To run them:

```shell
uv run jupyter lab $PATH_TO_NOTEBOOK/notebook.ipynb
```

with the corresponding path to the notebook.ipynb of your choice.

## Usage

The usual way of using `kintomo` starts with three steps:
1. Define a point cloud as a cartesian `Sculpture`.
2. Define a `Velocity` profile for every point in the `Sculpture`.
3. Define a `Grid` to project the point cloud on.

#### `Sculpture`

There are several ways of defining a `Sculpture` object. One possibility is to define first a cube of points `cartesian_cube` as a `Sculpture`, with arguments `x`, `y`, `z` (1D arrays):

```python
num_points = 5
x0, y0, z0 = (2 * np.random.rand(num_points) - 1 for _ in range(3))
cartesian_cube = Sculpture(x=x0, y=y0, z=z0)
```

Here, the point cloud is composed of 5 points and is contained inside a cube of length `2`, in the interval `[-1,1]` for every direction.

:information_source: ***Remark***: You can also use a shorter version where you use directly the `cube` classmethod:
```python
cartesian_cube = Sculpture.cube(500000)
x0, y0, z0 = (cartesian_cube.coordinates[k] for k in ("x","y","z"))
# equivalent to x0, y0, z0 = (cartesian_cube.x, cartesian_cube.y, cartesian_cube.z) 
```

Then, it is possible to carve a user-defined `Shape`, like a cylinder, from `cartesian_cube`:

```python
cylinder = (x0**2 + y0**2 < 1.0**2) & (abs(z0) < 0.02)
sculpture = cartesian_cube.carve(
    shape=Shape(cylinder)
)
```

:information_source: ***Remark***: There are several shapes that are already defined in `kintomo` if needed, like the cylinder:
```python
sculpture = Sculpture(x=x0, y=y0, z=z0).carve(
    shape=Shape.cylinder(
        x=x0, 
        y=y0, 
        z=z0, 
        rmin=0.2, 
        rmax=1.0, 
        height=0.02,
    )
)
```

#### `Velocity`

In order to define a `Velocity` depending on the points position in space, it is possible to convert and access the cartesian, cylindrical and spherical coordinates associated to the `Sculpture`:

```python
x, y, z = sculpture.cartesian_coordinates()
r, phi, z = sculpture.cylindrical_coordinates()
r, theta, phi = sculpture.spherical_coordinates()
```

If the `Shape` is defined with an offset compared to the origin (0,0,0), you can add an offset when accessing the coordinates using the arguments `x_offset`, `y_offset`, `z_offset` (see the `notebooks/double_spheres.ipynb` notebook for a concrete example).

It is then possible to define the velocity profile, with 4 arguments : the `geometry` (`"cartesian"`,`"cylindrical"`,`"spherical"`) and the 3 components of the velocity (`v1`,`v2`,`v3`) with the correct order depending on the geometry.

Example (spherical) : (v1, v2, v3) $\rightarrow$ (v$_\rm r$, v$_\theta$, v$_\phi$).
Example of a userfef velocity profile corresponding to a keplerian profile:

```python
velocity = Velocity(
    geometry=Geometry("cylindrical"),
    v1=np.zeros_like(r),
    v2=np.sqrt(1/r),
    v3=np.zeros_like(r),
).to_cartesian(phi=phi)
```

For now, note that the `Velocity` **must** be converted to cartesian if not already, using the `to_cartesian` method, specifying: 
- the `phi` coordinate if the native geometry is `"cylindrical"`
- the `phi` and `theta` coordinates if the native geometry is `"spherical"`

:information_source: ***Remark***: There are several velocity profiles that are already defined in `kintomo` if needed, e.g.,:

```python
velocity = Velocity.keplerian(r=r).to_cartesian(phi=phi)
```

#### `Grid`

To define the `Grid` on which will be deposited the point cloud, you can use the `encompass` override method: 

```python
nx, ny, nz = (65, 65, 33)
grid = Grid.encompass(
    sculpture=sculpture, 
    dimension=(nx, ny, nz),
)
```

It is also possible to have a user-defined grid. The following example corresponds to what is performed in the `encompass` override method:
```python
grid = Grid(
    xedge = np.linspace(2*x.min(), 2*x.max(), nx+1),
    yedge = np.linspace(-2*sculpture.max_size_yz, 2*sculpture.max_size_yz, ny+1),
    zedge = np.linspace(-2*sculpture.max_size_yz, 2*sculpture.max_size_yz, nz+1),
)
```

For more info, see the `notebooks/keplerian_disk.ipynb` notebook.

#### Additional remarks

*`kintomo` & simulations*. It is possible to use outputs from simulations to take a look at the kinematics, as long as the arrays are flattened into 1D arrays. For more info, see the `notebooks/from_idefix_outputs.ipynb` notebook.

*`kintomo` & multiple sculptures*. You can combine multiple sculpture objects and their corresponding velocities, using the `+` operator:

```python
(sculpture, velocity) = (sculpture_1+sculpture_2, velocity_1+velocity_2)
```

For more info, see the `notebooks/double_spheres.ipynb` notebook.

*`kintomo` & units*. You can dimensionalize your problem with astropy.units using the `to` method:

```python
unit_l = 100.0*u.au
cube = Sculpture.cube(5).to(unit=unit_l)
```
