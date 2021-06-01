using Oetqf
using Test

println("Number of threads: $(Threads.nthreads()).")
include("tests.jl")
include("BEM/tests.jl")