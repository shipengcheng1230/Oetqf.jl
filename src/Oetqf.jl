module Oetqf

using Reexport

@reexport using OrdinaryDiffEq
@reexport using DiffEqCallbacks
@reexport using RecursiveArrayTools

using LinearAlgebra
using HDF5
using Parameters
using GmshTools
using GeoGreensFunctions
using FFTW

using Base.Threads

const BEM_DIR = joinpath(@__DIR__, "BEM")
const BEM_SRC = ["mesh.jl", "GF.jl", "property.jl", "equation.jl"]
foreach(x -> include(joinpath(BEM_DIR, x)), BEM_SRC)

export
    gen_mesh, gen_gmsh_mesh,
    stress_greens_function, StrikeSlip,
    RateStateQuasiDynamicProperty,
    assemble

end # module
