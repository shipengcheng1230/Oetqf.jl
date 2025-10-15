# !!! note
#     This example corresponds to the simulations in Shi, P., Wei, M., & Barbot, S. (2022), JGR - Solid Earth - 10.1029/2022jb024069. 
#     The mesh size is reduced to improve CI/CD speed and avoid timeouts.

# # Problem statement
# We would like to investigate the stress interaction between a 2D plane transform fault and a 3D mantle 
# and how it affects the seismic pattern.

using Oetqf, SpecialFunctions, Optim

# !!! tip
#     You will need to install [SpecialFunctions](https://github.com/JuliaMath/SpecialFunctions.jl) and [Optim](https://github.com/JuliaNLSolvers/Optim.jl) to run this example. We use them to help set up the initial conditions for the equations.

# # Generate the meshes

# Generate the mesh for the transform fault, which is suitable for using the Okada (1992) equation.
# The fault is 80 km long, 8 km deep, with grid sizes of 10 km and 2 km respectively, and a dip angle of 90 degrees (vertical).
mf = gen_mesh(Val(:RectOkada), 80e3, 8e3, 10e3, 2e3, 90.0);

# The mesh `mf` is a `RectOkadaMesh`, which contains the fault geometry, centroid coordinates, and other properties.

# Use Gmsh to generate the mantle mesh, which is suitable for using the Barbot et al. (2017) equation.
# The volume is 80 km long, 5 km wide, and 14 km deep, with a grid size of 4 cells along the **x** direction, 3 cells along the **y** direction, and 3 cells along the **z** direction.
# There is no refinement in the **x** or **y** direction, while cell sizes are 1.5 times progressively larger along the **z** axis.
gen_gmsh_mesh(Val(:BEMHex8Mesh), -40e3, -2.5e3, -8e3, 80e3, 5e3, -22e3, 4, 3, 3;
    output = joinpath(@__DIR__, "mantle.vtk"),
    rfzh = cumprod(ones(3) * 1.5), rfy = 1.0, rfyType = "Bump"
)
ma = gen_mesh(Val(:BEMHex8Mesh), joinpath(@__DIR__, "mantle.vtk"));

# The mesh `ma` is a `BEMHex8Mesh`, which contains the mantle geometry, centroid coordinates, and other properties.

# # Compute the stress Green's functions

# Assume the shear modulus and the Lamé parameter are both 3e10 Pa.
λ = μ = 3e10;

# Initialize the path to save the stress Green's functions.
gffile = joinpath(@__DIR__, "gf.h5")
isfile(gffile) && rm(gffile);

# Compute the stress Green's functions within the fault. We add a buffer zone of 1 times the fault length on both sides of the fault to avoid edge effects.
@time gf₁₁ = stress_greens_function(mf, λ, μ; buffer_ratio = 1)
h5write(gffile, "gf₁₁", gf₁₁); # fault -> fault

# Compute the stress Green's functions from the fault to the mantle.
@time gf₁₂ = stress_greens_function(mf, ma, λ, μ; buffer_ratio = 1, qtype = "Gauss1")
h5write(gffile, "gf₁₂", gf₁₂); # fault -> mantle

# Compute the stress Green's functions from the mantle to the fault and within the mantle.
@time gf₂₁ = stress_greens_function(ma, mf, λ, μ)
h5write(gffile, "gf₂₁", gf₂₁); # mantle -> fault

# Compute the stress Green's functions within the mantle, using Gauss1 quadrature.
@time gf₂₂ = stress_greens_function(ma, λ, μ; qtype = "Gauss1")
h5write(gffile, "gf₂₂", gf₂₂); # mantle -> mantle

# !!! tip
#     The `buffer_ratio` denotes the fraction of the original fault length
#     on the two sides of the fault in which no dislocation occurs.
#     It serves as a buffer zone to imitate the ridge section on the edges of an oceanic transform fault (personal communication with Yajing Liu).
#     Basically, it affects how the stiffness tensor is periodically summed.

# !!! tip
#     Notice that, in Gmsh before v4.9, the quadrature type "Gauss2" does not stand for the product rule; instead, it is an optimized cubature
#     rule (see [this issue](https://gitlab.onelab.info/gmsh/gmsh/-/issues/1351)). For more cubature rules, see [quadpy](https://github.com/nschloe/quadpy).

# !!! tip
#     It is recommended to use at least `Gauss2` quadrature for the mantle mesh to ensure the maximum real part of the eigenvalues of the stiffness tensor is small.
#     The volumetric mean has drastically reduced it to avoid numerical instability. See the appendix of the [Shi et al., 2022](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022JB024069) for more details.

# # Set up parameters
# Set up the rate-and-state friction parameters on the fault. We include two velocity-weakening patches on the fault:
# one on the left side between -25 km and -5 km, and the other on the right side between 5 km and 25 km, and the depth between -6 km and -1 km.
cs = 3044.14 # m/s
vpl = 140e-3 / 365 / 86400 # 140 mm/yr
v0 = 1e-6
f0 = 0.6
μ = 3e10
η = μ / 2cs # radiation damping
ν = λ / 2(λ + μ)
avw = 0.015
abvw = 0.0047
Dc = 8e-3
σmax = 5e7
a = ones(mf.nx, mf.nξ) .* avw
b = ones(mf.nx, mf.nξ) .* (avw - abvw)
L = ones(mf.nx, mf.nξ) .* Dc
σ = [min(σmax, 1.5e6 + 18.0e3 * z) for z in -mf.z] # Pa
σ = repeat(σ, 1, mf.nx)' |> Matrix # Pa
left_patch = @. -25.e3 ≤ mf.x ≤ -5.e3
right_patch = @. 5.e3 ≤ mf.x ≤ 25.e3
vert_patch = @. -6.e3 ≤ mf.z ≤ -1e3
b[xor.(left_patch, right_patch), vert_patch] .= avw + abvw # assign velocity weakening
pf = RateStateQuasiDynamicProperty(a, b, L, σ, η, vpl, f0, v0)
save_property(joinpath(@__DIR__, "para-fault.bson"), pf);

# Set up rheology parameters in the mantle assuming power-law viscosity with lab-derived results.
#src # wet dislocation
A_wet_dis = 3e1
Q_wet_dis = 480e3
V_wet_dis = 11e-6
m_wet_dis = 0
r_wet_dis = 1.2
n_wet_dis = 3.5
#src # others
grain_size = 10000.0 # μm
COH = 1000 # ppm / HSi
𝙍 = 8.314 # gas constant
#src # Pressure, Temperature
crust_depth = 7e3
κ = 8e-7
𝚃(z) = 1673 * erf(z / sqrt(4κ * 1e6 * 365 * 86400)) # 1 Myr OTF
𝙿(z) = 2800 * 9.8 * crust_depth + 3300 * 9.8 * (z - crust_depth)
#src # plastic law
prefactor_dis(z) = A_wet_dis / (1e6)^n_wet_dis * COH^r_wet_dis * grain_size^m_wet_dis * exp(-(Q_wet_dis + 𝙿(z) * V_wet_dis) / 𝙍 / 𝚃(z))
rel_dϵ = [0.0, -1e-12, 0.0, 0.0, 0.0, 0.0]
amplifier = 1e0
γ_dis = prefactor_dis.(-ma.cz) .* amplifier
pa = PowerLawViscosityProperty(γ_dis, ones(length(ma.cz)) * (n_wet_dis - 1), rel_dϵ) # note to save `n-1` instead of `n`, where `n` refers to the stress power
save_property(joinpath(@__DIR__, "para-mantle" * ".bson"), pa);

# !!! warning
#     Make sure your units are consistent across the whole variable space.
#     Also, note that we save `n-1` instead of `n`, where `n` refers to the stress power.

# !!! tip
#     To load existing properties, use `load_property(YOUR_FILE, :RateStateQuasiDynamicProperty)` or `load_property(YOUR_FILE, :PowerLawViscosityProperty)` accordingly.

# # Set up initial conditions
# We set up a uniform initial velocity field equal to the plate rate on the fault, with slight perturbations in the state variable of the left and right halves of the fault.
vinit = pf.vpl .* ones(size(pf.a))
θinit = pf.L ./ vinit
θinit[1: size(θinit, 1) >> 1, :] ./= 1.1
θinit[size(θinit, 1) >> 1 + 1: end, :] ./= 2.5
δinit = zeros(size(pf.a));

# We set up initial conditions in the mantle where the initial stress matches the background strain rate through a simple 1D optimization.
ϵinit = zeros(length(pa.γ), 6)
P = map(z -> 2800 * 9.8 * crust_depth + 3300 * 9.8 * (z - crust_depth), -ma.cz) # change the depth of crust
σinit = repeat(P, 1, 6)
σinit[:,3] .= 0.0 # xz
σinit[:,5] .= 0.0 # yz
#src # balance the given background strain rate
target(i) = x -> (pa.γ[i] * (sqrt(2) * x) ^ (pa.n[i]) * x - abs(pa.dϵ₀[2])) ^ 2
σxyinit = -map(i -> Optim.minimizer(optimize(target(i), 1e1, 1e14)), 1: length(pa.γ))
reldϵ = map(i -> pa.γ[i] * (sqrt(2) * abs(σxyinit[i])) ^ (pa.n[i]) * σxyinit[i], 1: length(pa.γ))
@assert all(isapprox.(reldϵ, pa.dϵ₀[2]; rtol=1e-3))
σinit[:,2] .= σxyinit;

# # Assemble the problem
# Using the Green's functions, the properties, and the initial conditions, we assemble the problem. Notice the order of the variables here must be velocity, state variable, strain, stress, and slip.
uinit = ArrayPartition(vinit, θinit, ϵinit, σinit, δinit)
prob = assemble(gf₁₁, gf₁₂, gf₂₁, gf₂₂, pf, pa, uinit, (0.0, 0.1 * 365 * 86400));

# We set up the saving scheme and solve the equation. Here, we will save (in the order of) velocity, state variable, strain rate, strain, stress, and slip, every 100 steps.
# All the variables are named exactly the same as in the equations.
handler(u::ArrayPartition, t, integrator) = (u.x[1], u.x[2], integrator(integrator.t, Val{1}).x[3], u.x[3], u.x[4], u.x[5])
output = joinpath(@__DIR__, "output.h5")
@time sol = wsolve(prob, VCABM5(), output, 100, handler, ["v", "θ", "dϵ", "ϵ", "σ", "δ"], "t";
    reltol=1e-6, abstol=1e-8, dtmax=0.2*365*86400, dt=1e-8, maxiters=1e9, stride=100, force=true
)

# !!! tip
#     See [this issue](https://github.com/SciML/OrdinaryDiffEq.jl/issues/785) to learn more about retrieving derivatives in the solution.

# !!! tip
#     We often find that multi-step solvers like `VCABM5` are more efficient than single-step solvers like `Tsit5` for this kind of problem.

# # Analyze the results

# The solution is saved in the `output.h5` file, which contains the time series of velocity, state variable, strain rate, strain, stress, and slip.
# We can load the solution and analyze the results, for example, extracting the earthquake catalog from the velocity time series, visualizing the fault rupture, mantle strain flow, etc.
# Readers are encouraged to explore the figures in the [Shi et al., 2022](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022JB024069) for more insights.

# !!! tip
#     To generate the PVD files for visualization in Paraview, we can use the following functions. 
#     The output PVD file can be opened in Paraview to visualize the animation of the fault and mantle evolution.
#     It includes the velocity and state variable on the fault and strain rate in the mantle, for all the time steps.
#     ```julia
#     gen_pvd(mf, joinpath(@__DIR__, "mantle.vtk"), output, "t", ["v", "θ"], ["dϵ"], 1: length(sol.t), joinpath(@__DIR__, "sol.pvd"))
#     ```
