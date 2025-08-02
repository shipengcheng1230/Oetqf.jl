# Oetqf

[![CI](https://github.com/shipengcheng1230/Oetqf.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/shipengcheng1230/Oetqf.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/shipengcheng1230/Oetqf.jl/branch/master/graph/badge.svg?token=e85AwCR80f)](https://codecov.io/gh/shipengcheng1230/Oetqf.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://shipengcheng1230.github.io/Oetqf.jl/dev/)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://shipengcheng1230.github.io/Oetqf.jl/stable)

## Introduction

This package is used to simulate the quasi-dynamic earthquake cycles under the framework of rate-and-state friction on a transfinite-mesh transform fault overlaying a viscoelastic hexahedron-mesh mantle using boundary-element-method (BEM). In the doc, we provide a detailed example used by Shi et al., 2022.

Currently, this package is under maintenance mode. Contributions and questions are welcome, feel free to raise them in the GitHub issues or dicussions pages. If you find this package useful in your research, please cite the reference listed below.

## Installation

You can install this package of a specific version:

```julia
(@v1.11) pkg> add https://github.com/shipengcheng1230/Oetqf.jl#v0.3.1
```

## Reference

- Wei, M., & Shi, P. (2021). Synchronization of Earthquake Cycles of Adjacent Segments on Oceanic Transform Faults Revealed by Numerical Simulation in the Framework of Rate-and-State Friction. Journal of Geophysical Research: Solid Earth, 126(1), e2020JB020231. https://doi.org/10.1029/2020JB020231

- Shi, P., Wei, M., & Barbot, S. (2022). Contribution of Viscoelastic Stress to the Synchronization of Earthquake Cycles on Oceanic Transform Faults. Journal of Geophysical Research: Solid Earth, 127, e2022JB024069. https://doi.org/10.1029/2022JB024069
