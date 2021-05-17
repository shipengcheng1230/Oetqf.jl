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
    alloc = gen_alloc(mf.nx, mf.nξ)
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
