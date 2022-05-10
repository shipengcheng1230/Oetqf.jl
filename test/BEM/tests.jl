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
            @test length(unique(x -> round(x; digits=6), x)) == tx
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

    for ftype ∈ (StrikeSlip(), DipSlip())
        mf = gen_mesh(Val(:RectOkada), 100.0, 100.0, 10.0, 10.0, 41.0)
        gf1 = stress_greens_function(mf, 3e10, 3e10; ftype=ftype, fourier=true)
        gf2 = stress_greens_function(mf, 3e10, 3e10; ftype=ftype, fourier=false)
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

@testset "Property save & load" begin
    nx = nξ = ne = 4
    p1 = RateStateQuasiDynamicProperty([rand(nx, nξ) for _ in 1: 4]..., rand(4)...)
    p2 = PowerLawViscosityProperty(rand(ne), 3 * ones(Int, ne), rand(6))
    p3 = CompositePowerLawViscosityProperty([p2, p2], rand(6))
    p4 = DilatancyProperty([rand(nx, nξ) for _ in 1: 4] ...)

    ftmp = tempname()
    save_property(ftmp, p1)
    p1′ = load_property(ftmp, :RateStateQuasiDynamicProperty)
    save_property(ftmp, p2)
    p2′ = load_property(ftmp, :PowerLawViscosityProperty)
    save_property(ftmp, p3)
    p3′ = load_property(ftmp, :CompositePowerLawViscosityProperty)
    save_property(ftmp, p4)
    p4′ = load_property(ftmp, :DilatancyProperty)

    save_property(ftmp, p1, p2)
    p1′′ = load_property(ftmp, :RateStateQuasiDynamicProperty)
    p2′′ = load_property(ftmp, :PowerLawViscosityProperty)
    save_property(ftmp, p1, p3)
    p3′′ = load_property(ftmp, :CompositePowerLawViscosityProperty)
    save_property(ftmp, p1, p4)
    p4′′ = load_property(ftmp, :DilatancyProperty)
    @test p1 == p1′ == p1′′
    @test p2 == p2′ == p2′′
    @test p3 == p3′ == p3′′
    @test p4 == p4′ == p4′′
end

@testset "Viscosity Law" begin
    import Oetqf: dϵ_dt

    A, n, σ, τ = [rand() for _ ∈ 1: 4]
    p = PowerLawViscosityProperty([A], [n], rand(6))
    expected = A * σ * τ ^ n
    given = dϵ_dt(p, 1, σ, τ)
    @test given == expected
    A2, n2 = rand(), rand()
    p2 = PowerLawViscosityProperty([A2], [n2], rand(6))
    pc = CompositePowerLawViscosityProperty([p, p2], rand(6))
    @test dϵ_dt(pc, 1, σ, τ) == expected + A2 * σ * τ ^ n2
end