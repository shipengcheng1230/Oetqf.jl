module Oetqf

using LinearAlgebra
using Distributed
using HDF5
using Parameters
using GmshTools

const BEM_DIR = joinpath(@__DIR__, "BEM")
foreach(x -> include(joinpath(BEM_DIR, x)), readdir(BEM_DIR))

export gen_mesh, gen_gmsh_mesh

end # module
