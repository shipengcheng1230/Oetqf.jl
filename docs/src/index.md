# Introduction

This package is used to simulate the quasi-dynamic earthquake cycles under the framework of rate-and-state friction on a transfinite-mesh transform fault overlaying a viscoelastic hexahedron-mesh mantle using boundary-element-method (BEM). This package is an updated subset version of [Quaycle.jl](https://github.com/shipengcheng1230/Quaycle.jl) which includes dipping fault, triangular-mesh (fault) and tetrahedron-mesh (mantle).

A collection of commonly used Green's functions can be accessed at [GeoGreensFunctions.jl](https://github.com/shipengcheng1230/GeoGreensFunctions.jl). The package uses [Gmsh](https://gmsh.info/) for domain discreitzation. See [GmshTools.jl](https://github.com/shipengcheng1230/GmshTools.jl) also for a more convenient way to use Gmsh in Julia.

## Known Issues

- The competition between Julia threads and BLAS threads when hyperthreading is disabled, see [this example](https://discourse.julialang.org/t/possible-performance-drop-when-using-more-than-one-socket-threads/62022).