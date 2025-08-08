# Testing

## Unit Testing

We use [GitHub Actions](https://github.com/shipengcheng1230/Oetqf.jl/blob/master/.github/workflows/ci.yml) for continuous integration. The tests cover:

- ODE solution storage with HDF5
- PVD generation from solution outputs
- BLAS backend switching
- Mesh generation (built-in transfinite fault and GMSH)
- ODE assembly for various problem types
- Domain properties loading and saving
- Viscoelastic laws
- FFT convolution for translational symmetry Green's function

To run the tests locally:

```julia
(@v1.11) pkg> activate .
    Activating project at `~/Projects/Oetqf.jl`

(Oetqf) pkg> test
```

## Integration Testing

A notebook example provides integration testing, demonstrating end-to-end simulation of earthquake cycles on a plane fault coupled with a viscoelastic mantle. This notebook is included in the [continuous deployment](https://github.com/shipengcheng1230/Oetqf.jl/blob/master/.github/workflows/documentation.yml).

## Supported Environment

| OS            | Architecture | Julia Version |
|---------------|--------------|---------------|
| ubuntu-latest | x64          | v1.11         |

!!! note
    Due to resource limitation on GitHub, we only include Linux building in the CI/CD. This package has been manually tested on macOS.
