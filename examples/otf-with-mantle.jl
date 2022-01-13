# !!! note
#     This example corresponds to the simulations in Shi, P., Wei, M., & Barbot, S., (2022), submitted to JGR - Solid Earth. The mesh size is
#     downgraded for speed of the document building

using Oetqf, SpecialFunctions, Optim


# Generate the mesh for the transform fault, which is suited for using Okaka, (1992) equation:
mf = gen_mesh(Val(:RectOkada), 80e3, 8e3, 10e3, 2e3, 90.0);


# Use Gmsh to generate the mantle mesh, which is suited for using Barbot et al., (2017) equation,
# with no refinement in **x** or **y** direction while cell sizes are 1.5 times progressively larger along **z** axes:
gen_gmsh_mesh(Val(:BEMHex8Mesh), -40e3, -2.5e3, -8e3, 80e3, 5e3, -22e3, 4, 3, 3;
    output = joinpath(@__DIR__, "mantle.vtk"),
    rfzh = cumprod(ones(3) * 1.5), rfy = 1.0, rfyType = "Bump"
)
ma = gen_mesh(Val(:BEMHex8Mesh), joinpath(@__DIR__, "mantle.vtk"));


# Compute the stress Green's function between the two meshes:
λ = μ = 3e10
gffile = joinpath(@__DIR__, "gf.h5")
isfile(gffile) && rm(gffile)
@time gf₁₁ = stress_greens_function(mf, λ, μ; buffer_ratio = 1)
h5write(gffile, "gf₁₁", gf₁₁) # fault -> fault
@time gf₁₂ = stress_greens_function(mf, ma, λ, μ; buffer_ratio = 1, qtype = "Gauss1")
h5write(gffile, "gf₁₂", gf₁₂) # fault -> mantle
@time gf₂₁ = stress_greens_function(ma, mf, λ, μ)
h5write(gffile, "gf₂₁", gf₂₁) # mantle -> fault
@time gf₂₂ = stress_greens_function(ma, λ, μ; qtype = "Gauss1")
h5write(gffile, "gf₂₂", gf₂₂) # mantle -> mantle


# !!! tip
#     The `buffer_ratio` denotes the fraction to the original fault length
#     on the two sides of the fault in which no dislocation occurs.
#     It serves as a buffer zone to immitate the ridge section on the edges of an oceanic transform fault (personal communication with Yajing Liu).
#     Basically, it affects how the stiffness tensor are periodically summed.


# !!! tip
#     Notice that, in Gmsh before v4.9, the quadrature type "Gauss2" does not stand for the product rule, instead it is an optimized cubature
#     rule (see [this issue](https://gitlab.onelab.info/gmsh/gmsh/-/issues/1351)). For more cubature rules, see [quadpy](https://github.com/nschloe/quadpy).


# Set up the rate-and-state friction parameters in the fault:
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


# Set up rheology parameters in the mantle assuming power-law viscosity with lab-derived results:
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
𝙍 = 8.314 # gas contant
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
pa = PowerLawViscosityProperty(γ_dis, ones(length(ma.cz)) * (n_wet_dis - 1), rel_dϵ) # notice to save `n-1` instead of `n` where `n` refers the stress power
save_property(joinpath(@__DIR__, "para-mantle" * ".bson"), pa);


# !!! warning
#     Make sure your units are consistent across the whole variable space.
#     Also, notice that we save `n-1` instead of `n` where `n` refers the stress power.

# !!! tip
#     To load existing properties, use `load_property(YOUR_FILE, :RateStateQuasiDynamicProperty)` or `load_property(YOUR_FILE, :PowerLawViscosityProperty)` accordingly.


# Set up initial conditions on the fault with an offset between left and right half fault:
vinit = pf.vpl .* ones(size(pf.a))
θinit = pf.L ./ vinit
θinit[1: size(θinit, 1) >> 1, :] ./= 1.1
θinit[size(θinit, 1) >> 1 + 1: end, :] ./= 2.5
δinit = zeros(size(pf.a));


# Set up initial conditions in the mantle
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


# Assemble the problem:
uinit = ArrayPartition(vinit, θinit, ϵinit, σinit, δinit)
prob = assemble(gf₁₁, gf₁₂, gf₂₁, gf₂₂, pf, pa, uinit, (0.0, 0.1 * 365 * 86400));


# Set up the saving scheme and solve the equation:
handler(u::ArrayPartition, t, integrator) = (u.x[1], u.x[2], integrator(integrator.t, Val{1}).x[3], u.x[3], u.x[4], u.x[5])
output = joinpath(@__DIR__, "output.h5")
@time sol = wsolve(prob, VCABM5(), output, 100, handler, ["v", "θ", "dϵ", "ϵ", "σ", "δ"], "t";
    reltol=1e-6, abstol=1e-8, dtmax=0.2*365*86400, dt=1e-8, maxiters=1e9, stride=100, force=true
)

# !!! tip
#     See [this issue](https://github.com/SciML/OrdinaryDiffEq.jl/issues/785) to know more about retrieving derivatives in the solution.