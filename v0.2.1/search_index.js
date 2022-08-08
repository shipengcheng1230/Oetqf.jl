var documenterSearchIndex = {"docs":
[{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"EditURL = \"https://github.com/shipengcheng1230/Oetqf.jl/blob/master/examples/otf-with-mantle.jl\"","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"note: Note\nThis example corresponds to the simulations in Shi, P., Wei, M., & Barbot, S., (2022), submitted to JGR - Solid Earth. The mesh size is downgraded for speed of the document building","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"using Oetqf, SpecialFunctions, Optim","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"Generate the mesh for the transform fault, which is suited for using Okaka, (1992) equation:","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"mf = gen_mesh(Val(:RectOkada), 80e3, 8e3, 10e3, 2e3, 90.0);\nnothing #hide","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"Use Gmsh to generate the mantle mesh, which is suited for using Barbot et al., (2017) equation, with no refinement in x or y direction while cell sizes are 1.5 times progressively larger along z axes:","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"gen_gmsh_mesh(Val(:BEMHex8Mesh), -40e3, -2.5e3, -8e3, 80e3, 5e3, -22e3, 4, 3, 3;\n    output = joinpath(@__DIR__, \"mantle.vtk\"),\n    rfzh = cumprod(ones(3) * 1.5), rfy = 1.0, rfyType = \"Bump\"\n)\nma = gen_mesh(Val(:BEMHex8Mesh), joinpath(@__DIR__, \"mantle.vtk\"));\nnothing #hide","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"Compute the stress Green's function between the two meshes:","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"λ = μ = 3e10\ngffile = joinpath(@__DIR__, \"gf.h5\")\nisfile(gffile) && rm(gffile)\n@time gf₁₁ = stress_greens_function(mf, λ, μ; buffer_ratio = 1)\nh5write(gffile, \"gf₁₁\", gf₁₁) # fault -> fault\n@time gf₁₂ = stress_greens_function(mf, ma, λ, μ; buffer_ratio = 1, qtype = \"Gauss1\")\nh5write(gffile, \"gf₁₂\", gf₁₂) # fault -> mantle\n@time gf₂₁ = stress_greens_function(ma, mf, λ, μ)\nh5write(gffile, \"gf₂₁\", gf₂₁) # mantle -> fault\n@time gf₂₂ = stress_greens_function(ma, λ, μ; qtype = \"Gauss1\")\nh5write(gffile, \"gf₂₂\", gf₂₂) # mantle -> mantle","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"tip: Tip\nThe buffer_ratio denotes the fraction to the original fault length on the two sides of the fault in which no dislocation occurs. It serves as a buffer zone to immitate the ridge section on the edges of an oceanic transform fault (personal communication with Yajing Liu). Basically, it affects how the stiffness tensor are periodically summed.","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"tip: Tip\nNotice that, in Gmsh before v4.9, the quadrature type \"Gauss2\" does not stand for the product rule, instead it is an optimized cubature rule (see this issue). For more cubature rules, see quadpy.","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"Set up the rate-and-state friction parameters in the fault:","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"cs = 3044.14 # m/s\nvpl = 140e-3 / 365 / 86400 # 140 mm/yr\nv0 = 1e-6\nf0 = 0.6\nμ = 3e10\nη = μ / 2cs # radiation damping\nν = λ / 2(λ + μ)\navw = 0.015\nabvw = 0.0047\nDc = 8e-3\nσmax = 5e7\na = ones(mf.nx, mf.nξ) .* avw\nb = ones(mf.nx, mf.nξ) .* (avw - abvw)\nL = ones(mf.nx, mf.nξ) .* Dc\nσ = [min(σmax, 1.5e6 + 18.0e3 * z) for z in -mf.z] # Pa\nσ = repeat(σ, 1, mf.nx)' |> Matrix # Pa\nleft_patch = @. -25.e3 ≤ mf.x ≤ -5.e3\nright_patch = @. 5.e3 ≤ mf.x ≤ 25.e3\nvert_patch = @. -6.e3 ≤ mf.z ≤ -1e3\nb[xor.(left_patch, right_patch), vert_patch] .= avw + abvw # assign velocity weakening\npf = RateStateQuasiDynamicProperty(a, b, L, σ, η, vpl, f0, v0)\nsave_property(joinpath(@__DIR__, \"para-fault.bson\"), pf);\nnothing #hide","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"Set up rheology parameters in the mantle assuming power-law viscosity with lab-derived results:","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"A_wet_dis = 3e1\nQ_wet_dis = 480e3\nV_wet_dis = 11e-6\nm_wet_dis = 0\nr_wet_dis = 1.2\nn_wet_dis = 3.5\ngrain_size = 10000.0 # μm\nCOH = 1000 # ppm / HSi\n𝙍 = 8.314 # gas contant\ncrust_depth = 7e3\nκ = 8e-7\n𝚃(z) = 1673 * erf(z / sqrt(4κ * 1e6 * 365 * 86400)) # 1 Myr OTF\n𝙿(z) = 2800 * 9.8 * crust_depth + 3300 * 9.8 * (z - crust_depth)\nprefactor_dis(z) = A_wet_dis / (1e6)^n_wet_dis * COH^r_wet_dis * grain_size^m_wet_dis * exp(-(Q_wet_dis + 𝙿(z) * V_wet_dis) / 𝙍 / 𝚃(z))\nrel_dϵ = [0.0, -1e-12, 0.0, 0.0, 0.0, 0.0]\namplifier = 1e0\nγ_dis = prefactor_dis.(-ma.cz) .* amplifier\npa = PowerLawViscosityProperty(γ_dis, ones(length(ma.cz)) * (n_wet_dis - 1), rel_dϵ) # notice to save `n-1` instead of `n` where `n` refers the stress power\nsave_property(joinpath(@__DIR__, \"para-mantle\" * \".bson\"), pa);\nnothing #hide","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"warning: Warning\nMake sure your units are consistent across the whole variable space. Also, notice that we save n-1 instead of n where n refers the stress power.","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"tip: Tip\nTo load existing properties, use load_property(YOUR_FILE, :RateStateQuasiDynamicProperty) or load_property(YOUR_FILE, :PowerLawViscosityProperty) accordingly.","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"Set up initial conditions on the fault with an offset between left and right half fault:","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"vinit = pf.vpl .* ones(size(pf.a))\nθinit = pf.L ./ vinit\nθinit[1: size(θinit, 1) >> 1, :] ./= 1.1\nθinit[size(θinit, 1) >> 1 + 1: end, :] ./= 2.5\nδinit = zeros(size(pf.a));\nnothing #hide","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"Set up initial conditions in the mantle","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"ϵinit = zeros(length(pa.γ), 6)\nP = map(z -> 2800 * 9.8 * crust_depth + 3300 * 9.8 * (z - crust_depth), -ma.cz) # change the depth of crust\nσinit = repeat(P, 1, 6)\nσinit[:,3] .= 0.0 # xz\nσinit[:,5] .= 0.0 # yz\ntarget(i) = x -> (pa.γ[i] * (sqrt(2) * x) ^ (pa.n[i]) * x - abs(pa.dϵ₀[2])) ^ 2\nσxyinit = -map(i -> Optim.minimizer(optimize(target(i), 1e1, 1e14)), 1: length(pa.γ))\nreldϵ = map(i -> pa.γ[i] * (sqrt(2) * abs(σxyinit[i])) ^ (pa.n[i]) * σxyinit[i], 1: length(pa.γ))\n@assert all(isapprox.(reldϵ, pa.dϵ₀[2]; rtol=1e-3))\nσinit[:,2] .= σxyinit;\nnothing #hide","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"Assemble the problem:","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"uinit = ArrayPartition(vinit, θinit, ϵinit, σinit, δinit)\nprob = assemble(gf₁₁, gf₁₂, gf₂₁, gf₂₂, pf, pa, uinit, (0.0, 0.1 * 365 * 86400));\nnothing #hide","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"Set up the saving scheme and solve the equation:","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"handler(u::ArrayPartition, t, integrator) = (u.x[1], u.x[2], integrator(integrator.t, Val{1}).x[3], u.x[3], u.x[4], u.x[5])\noutput = joinpath(@__DIR__, \"output.h5\")\n@time sol = wsolve(prob, VCABM5(), output, 100, handler, [\"v\", \"θ\", \"dϵ\", \"ϵ\", \"σ\", \"δ\"], \"t\";\n    reltol=1e-6, abstol=1e-8, dtmax=0.2*365*86400, dt=1e-8, maxiters=1e9, stride=100, force=true\n)","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"tip: Tip\nSee this issue to know more about retrieving derivatives in the solution.","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"","category":"page"},{"location":"generated/otf-with-mantle/","page":"A 2D transform fault overlaying a 3D mantle","title":"A 2D transform fault overlaying a 3D mantle","text":"This page was generated using Literate.jl.","category":"page"},{"location":"APIs/#Public-Interface","page":"APIs","title":"Public Interface","text":"","category":"section"},{"location":"APIs/","page":"APIs","title":"APIs","text":"Modules = [Oetqf]\nPages = []\nPrivate = false\nOrder = [:type, :function, :constant, :macro]","category":"page"},{"location":"APIs/#Oetqf.gen_gmsh_mesh-Union{Tuple{I}, Tuple{T}, Tuple{Val{:BEMHex8Mesh}, T, T, T, T, T, T, I, I, I}} where {T, I}","page":"APIs","title":"Oetqf.gen_gmsh_mesh","text":"gen_gmsh_mesh(::Val{:BEMHex8Mesh},\n    llx::T, lly::T, llz::T, dx::T, dy::T, dz::T, nx::I, ny::I, nz::I;\n    rfx::T=one(T), rfy::T=one(T), rfzh::AbstractVector=ones(nz),\n    rfxType::AbstractString=\"Bump\", rfyType::AbstractString=\"Bump\",\n    output::AbstractString=\"temp.msh\"\n) where {T, I}\n\nGernate a box using 8-node hexahedron elements by vertically extruding transfinite curve on xy plane, allowing     total flexibility on the mesh size in z direction, and refinement in xy plane.\n\nArguments\n\nllx, lly, llz: coordinates of low-left corner on the top surface\ndx, dy, dz: x-, y-, z-extension\nnx, ny: number of cells along x-, y-axis\nrfx, rfy: refinement coefficients along x-, y-axis using Bump algorithm, please refer gmsh.model.geo.mesh.setTransfiniteCurve\nrfzh: accumulated height of cells along z-axis which will be normalized automatically, please refer heights in gmsh.model.geo.extrude\n\n\n\n\n\n","category":"method"},{"location":"APIs/#Oetqf.wsolve-Tuple{ODEProblem, OrdinaryDiffEqAlgorithm, Any, Any, Any, Any, Any}","page":"APIs","title":"Oetqf.wsolve","text":"wsolve(prob::ODEProblem, alg::OrdinaryDiffEqAlgorithm,\n    file, nstep, getu, ustrs, tstr; kwargs...)\n\nWrite the solution to HDF5 file while solving the ODE. The interface     is exactly the same as     solve an ODEProblem     except a few more about the saving procedure. Notice, it will set     save_everystep=false so to avoid memory blow up. The return code     will be written as an attribute in tstr data group.\n\nExtra Arguments\n\nfile::AbstractString: name of file to be saved\nnstep::Integer: number of steps after which a saving operation will be performed\ngetu::Function: function handler to extract desired solution for saving\nustr::AbstractVector: list of names to be assigned for each components, whose   length must equal the length of getu output\ntstr::AbstractString: name of time data\n\nKWARGS\n\nstride::Integer=1: downsampling rate for saving outputs\nappend::Bool=false: if true then append solution after the end of file\nforce::Bool=false: force to overwrite the existing solution file\n\n\n\n\n\n","category":"method"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package is used to simulate the quasi-dynamic earthquake cycles under the framework of rate-and-state friction on a transfinite-mesh transform fault overlaying a viscoelastic hexahedron-mesh mantle using boundary-element-method (BEM). This package is an updated subset version of Quaycle.jl which includes dipping fault, triangular-mesh (fault) and tetrahedron-mesh (mantle).","category":"page"},{"location":"","page":"Home","title":"Home","text":"A collection of commonly used Green's functions can be accessed at GeoGreensFunctions.jl. The package uses Gmsh for domain discreitzation. See GmshTools.jl also for a more convenient way to use Gmsh in Julia.","category":"page"},{"location":"#Known-Issues","page":"Home","title":"Known Issues","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The competition between Julia threads and BLAS threads when hyperthreading is disabled, see this example.","category":"page"}]
}
