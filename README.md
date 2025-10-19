# Oetqf

[![CI](https://github.com/shipengcheng1230/Oetqf.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/shipengcheng1230/Oetqf.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/shipengcheng1230/Oetqf.jl/branch/master/graph/badge.svg?token=e85AwCR80f)](https://codecov.io/gh/shipengcheng1230/Oetqf.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://shipengcheng1230.github.io/Oetqf.jl/dev/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://shipengcheng1230.github.io/Oetqf.jl/stable)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.08597/status.svg)](https://doi.org/10.21105/joss.08597)

## Introduction

This package is used to simulate the quasi-dynamic earthquake cycles under the framework of rate-and-state friction on a transfinite-mesh transform fault overlaying a viscoelastic hexahedron-mesh mantle using boundary-element-method (BEM). In the doc, we provide a detailed example used by Shi et al., 2022.

Currently, this package is under maintenance mode. Contributions and questions are welcome, feel free to raise them in the GitHub issues or dicussions pages. If you find this package useful in your research, please cite the reference listed below.

## Installation

You can install this package of the latest stable version via:

```julia
(@v1.11) pkg> add https://github.com/shipengcheng1230/Oetqf.jl#v0.3.6
```

## Contributing

Contributions are highly welcome and encouraged! Whether you’re interested in extending the physics (e.g., fault or mantle dynamics), improving geometric or meshing capabilities, or optimizing performance and parallelism, your input is hugely valuable!

- For ideas or feature requests, please open an issue or discussion thread.
- For code contributions, please work from a fork of this repository and open a pull request when your changes are ready. Small improvements and major additions are equally appreciated.
- If you’d like to collaborate on research-level extensions, don’t hesitate to reach out by email.

## Reference

- Wei, M., & Shi, P. (2021). Synchronization of Earthquake Cycles of Adjacent Segments on Oceanic Transform Faults Revealed by Numerical Simulation in the Framework of Rate-and-State Friction. Journal of Geophysical Research: Solid Earth, 126(1), e2020JB020231. https://doi.org/10.1029/2020JB020231

- Shi, P., Wei, M., & Barbot, S. (2022). Contribution of Viscoelastic Stress to the Synchronization of Earthquake Cycles on Oceanic Transform Faults. Journal of Geophysical Research: Solid Earth, 127, e2022JB024069. https://doi.org/10.1029/2022JB024069

- Shi et al., (2025). Oetqf: A Julia package for quasi-dynamic earthquake cycle simulation. Journal of Open Source Software, 10(114), 8597, https://doi.org/10.21105/joss.08597