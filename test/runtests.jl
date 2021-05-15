using Oetqf
using Test

println("Number of threads: $(Threads.nthreads()).")

BEM_DIR = joinpath(@__DIR__, "BEM")
foreach(x -> include(joinpath(BEM_DIR, x)), readdir(BEM_DIR))