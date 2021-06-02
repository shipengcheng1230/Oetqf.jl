using Oetqf
using Test
using LinearAlgebra

println("Number of threads: $(Threads.nthreads()).")
include("tests.jl")
include("BEM/tests.jl")