# Introduction

This package is used to simulate the quasi-dynamic earthquake cycles under the framework of rate-and-state friction on a transfinite-mesh transform fault overlaying a viscoelastic hexahedron-mesh mantle using boundary-element-method (BEM). This package is an updated subset version of [Quaycle.jl](https://github.com/shipengcheng1230/Quaycle.jl) which includes dipping fault, triangular-mesh (fault) and tetrahedron-mesh (mantle).

## Supported physics

In this package, for frictional law we support `DieterichStateLaw`. For rate-and-state friction, we support the regularized form and dilatancy machanism. For viscoelasticity, we support power-law rheology. Users can extend new physics by implementing new equations, providing new parameters if necessary, and aseembling them into the ODE function. Please see the existing ODE functions to get started.

## Thrid-party libraries

This package uses HDF5 to save the numerical output for further analysis. It also supports writing output into VTK for visualization and animation.

A collection of commonly used Green's functions can be accessed at [GeoGreensFunctions.jl](https://github.com/shipengcheng1230/GeoGreensFunctions.jl). The package uses [Gmsh](https://gmsh.info/) for domain discreitzation. See [GmshTools.jl](https://github.com/shipengcheng1230/GmshTools.jl) also for a more convenient way to use Gmsh in Julia.

## Installation

You can install this package of a specific version:

```julia
(@v1.11) pkg> add https://github.com/shipengcheng1230/Oetqf.jl#<version_number>
```

## Contributing

Contributions are highly welcome and encouraged! Whether you’re interested in extending the physics (e.g., fault or mantle dynamics), improving geometric or meshing capabilities, or optimizing performance and parallelism, your input is hugely valuable!

- For ideas or feature requests, please open an issue or discussion thread.
- For code contributions, please work from a fork of this repository and open a pull request when your changes are ready. Small improvements and major additions are equally appreciated.
- If you’d like to collaborate on research-level extensions, don’t hesitate to reach out by email.

## Known Issues

- The competition between Julia threads and BLAS threads when hyperthreading is disabled, see [this example](https://discourse.julialang.org/t/possible-performance-drop-when-using-more-than-one-socket-threads/62022).
