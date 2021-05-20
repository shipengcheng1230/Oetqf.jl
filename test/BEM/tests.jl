using Test
using TensorOperations

@testset "Basic Mesh Generator" begin
    @testset "Rect for dc3d" begin
        msh = gen_mesh(Val(:RectOkada), 100.0, 50.0, 2.0, 2.0, 33.0)
        @test msh.nx == 50
        @test msh.nξ == 25
        @test msh.aξ[end][1] == msh.ξ[end] - msh.Δξ/2
        @test msh.y[end] == msh.ξ[end] * cosd(msh.dip)
        @test msh.z[end] == msh.ξ[end] * sind(msh.dip)
        @test msh.x[end] - msh.x[1] == msh.Δx * (msh.nx - 1)
    end

    @testset "Hex8 Mesh" begin
        file = tempname() * ".msh"
        llx, lly, llz = rand(3)
        dx, dy, dz = rand(3) * 5
        nx, ny, nz = [rand(2: 10) for _ in 1: 3]

        gen_gmsh_mesh(Val(:BEMHex8Mesh), llx, lly, llz, dx, dy, -dz, nx, ny, nz; output=file)
        me = gen_mesh(Val(:BEMHex8Mesh), file)

        for (x, tx) in zip([me.cx, me.cy, me.cz], [nx, ny, nz])
            @test length(unique(x -> round(x; digits=2), x)) == tx
        end
        @test me.cx ≈ me.qx
        @test (me.cy .- me.Δy / 2) ≈ me.qy
        @test me.cz .+ me.Δz / 2 ≈ me.qz
        @test maximum(me.cx + me.Δx / 2) ≈ llx + dx
        @test minimum(me.cx - me.Δx / 2) ≈ llx
        @test maximum(me.cy + me.Δy / 2) ≈ lly + dy
        @test minimum(me.cy - me.Δy / 2) ≈ lly
        @test maximum(me.cz + me.Δz / 2) ≈ llz
        @test minimum(me.cz - me.Δz / 2) ≈ llz - dz
    end
end

@testset "2D FFT conv" begin
    import Oetqf: gen_alloc, relative_velocity!, dτ_dt!

    mf = gen_mesh(Val(:RectOkada), 100.0, 100.0, 10.0, 10.0, 90.0)
    gf1 = stress_greens_function(mf, 3e10, 3e10; fourier=true)
    gf2 = stress_greens_function(mf, 3e10, 3e10; fourier=false)
    gf3 = Array{Float64}(undef, mf.nx, mf.nξ, mf.nx, mf.nξ)
    for l = 1: mf.nξ, k = 1: mf.nx, j = 1: mf.nξ, i = 1: mf.nx
        gf3[i,j,k,l] = gf2[abs(i-k)+1,j,l]
    end
    alloc = gen_alloc(Val(:BEMFault), mf.nx, mf.nξ)
    v = rand(mf.nx, mf.nξ)
    vpl = 0.1
    relv = v .- vpl
    relative_velocity!(alloc, vpl, v)
    dτ_dt!(gf1, alloc)
    @tensor begin
        E[i,j] := gf3[i,j,k,l] * relv[k,l]
    end
    @test E ≈ alloc.dτ_dt
end

@testset "Okada assemble" begin
    mesh = gen_mesh(Val(:RectOkada), 10., 10., 2., 2., 90.)
    gf = stress_greens_function(mesh, 1.0, 1.0; buffer_ratio=1.0)
    p = RateStateQuasiDynamicProperty([rand(mesh.nx, mesh.nξ) for _ in 1: 4]..., rand(4)...)
    u0 = ArrayPartition([rand(mesh.nx, mesh.nξ) for _ in 1: 3]...)
    prob = assemble(gf, p, u0, (0., 1.0))
    du = similar(u0)
    @test_nowarn @inferred prob.f(du, u0, prob.p, 1.0)
end

@testset "HDF5 storage" begin
    function foo(du, u, p, t)
        du.x[1] .= -u.x[1] / 10
        du.x[2] .= -u.x[2] / 100
        du.x[3] .= -u.x[3] / 1000
    end

    u0 = ArrayPartition(rand(2, 3), rand(5), rand(3, 2))
    tspan = (0.0, 5000.0)
    prob = ODEProblem(foo, u0, tspan)
    sol = solve(prob, Tsit5())
    ustrs = ["u1", "u2", "u3"]
    tmp = tempname() * ".h5"
    getu = (u, t, integrator) -> (u.x[1], u.x[2], u.x[3])
    wsolve(prob, Tsit5(), tmp, 50, getu, ["u1", "u2", "u3"], "t")
    @test h5read(tmp, "t") == sol.t
    for m in eachindex(u0.x)
        x = Array(VectorOfArray([sol.u[i].x[m] for i in eachindex(sol.t)]))
        @test h5read(tmp, ustrs[m]) == x
    end

    # appended storage
    prob2 = ODEProblem(foo, u0, (6000.0, 7000.0))
    sol2 = solve(prob2, Tsit5())
    wsolve(prob2, Tsit5(), tmp, 50, getu, ["u1", "u2", "u3"], "t"; append=true)
    tcat = cat(sol.t, sol2.t; dims=1)
    ucat = cat(sol.u, sol2.u; dims=1)
    @test tcat == h5read(tmp, "t")
    for m in eachindex(u0.x)
        x = Array(VectorOfArray([ucat[i].x[m] for i in eachindex(tcat)]))
        @test h5read(tmp, ustrs[m]) ≈ x
    end

    # strided storage
    stride = 11
    wsolve(prob, Tsit5(), tmp, 50, getu, ["u1", "u2", "u3"], "t"; stride=stride, force=true)
    @test length(h5read(tmp, "t")) == length(sol.t) ÷ stride + 1
    for m in eachindex(u0.x)
        x = Array(VectorOfArray([sol.u[i].x[m] for i in 1: stride: length(sol.t)]))
        @test h5read(tmp, ustrs[m]) == x
    end
end

@testset "Viscoelastic assemble" begin
    mf = gen_mesh(Val(:RectOkada), 100.0, 100.0, 10.0, 20.0, 90.0)
    temp = tempname() * ".msh"
    gen_gmsh_mesh(Val(:BEMHex8Mesh), -100.0, -50.0, -20.0, 200.0, 100.0, -30.0, 2, 3, 4; output=temp)
    ma = gen_mesh(Val(:BEMHex8Mesh), temp)

    nx = mf.nx
    nξ = mf.nξ
    ne = length(ma.cx)
    λ = μ = 1.0
    gf11 = stress_greens_function(mf, λ, μ)
    gf12 = stress_greens_function(mf, ma, λ, μ)
    gf21 = stress_greens_function(ma, mf, λ, μ)
    gf22 = stress_greens_function(ma, λ, μ)

    v0 = rand(nx, nξ)
    θ0 = rand(nx, nξ)
    ϵ0 = rand(ne, 6)
    σ0 = rand(ne, 6)
    δ0 = rand(nx, nξ)
    u0 = ArrayPartition(v0, θ0, ϵ0, σ0, δ0)
    pf = RateStateQuasiDynamicProperty([rand(nx, nξ) for _ in 1: 4]..., rand(4)...)
    pa = PowerLawViscosityProperty(rand(ne), 3 * ones(Int, ne), rand(6))
    prob = assemble(gf11, gf12, gf21, gf22, pf, pa, u0, (0.0, 1.0))
    du = similar(u0)
    @test_nowarn @inferred prob.f(du, u0, prob.p, 1.0)
end
