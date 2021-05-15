using Test

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
