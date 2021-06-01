module Oetqf

using Reexport

@reexport using OrdinaryDiffEq
@reexport using DiffEqCallbacks
@reexport using RecursiveArrayTools
@reexport using HDF5
@reexport using GmshTools
@reexport using LinearAlgebra
@reexport using FFTW
@reexport using BSON

using Parameters
using GeoGreensFunctions
using Strided
using Printf
using MLStyle
using Polyester
using WriteVTK
using Formatting

using Base.Threads

include("io.jl")

const BEM_DIR = joinpath(@__DIR__, "BEM")
const BEM_SRC = ["mesh.jl", "GF.jl", "property.jl", "equation.jl"]
foreach(x -> include(joinpath(BEM_DIR, x)), BEM_SRC)

include("vtk.jl")

export
    gen_mesh, gen_gmsh_mesh,
    stress_greens_function, StrikeSlip,
    RateStateQuasiDynamicProperty, PowerLawViscosityProperty,
    assemble,
    wsolve,
    save_property, load_property,
    gen_pvd, gen_vtk_grid

end # module
