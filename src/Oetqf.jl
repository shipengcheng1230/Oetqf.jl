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
using Preferences
using LoopVectorization

using Base.Threads

include("utils.jl")
include("pref.jl")

include("io.jl")
export wsolve

include("BEM/mesh.jl")
export gen_mesh, gen_gmsh_mesh

include("BEM/GF.jl")
export stress_greens_function, StrikeSlip

include("BEM/property.jl")
export RateStateQuasiDynamicProperty, PowerLawViscosityProperty,
       save_property, load_property

include("BEM/equation.jl")
export assemble

include("vtk.jl")
export gen_pvd, gen_vtk_grid

end # module
