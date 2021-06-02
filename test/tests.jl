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

@testset "VTK output" begin
    mf = gen_mesh(Val(:RectOkada), 100.0, 100.0, 10.0, 10.0, 90.0)
    mafile = tempname() * ".msh"
    gen_gmsh_mesh(Val(:BEMHex8Mesh), -50e3, -20e3, -10e3, 100e3, 40e3, -30e3, 3, 4, 5; output=mafile)
    ma = gen_mesh(Val(:BEMHex8Mesh), mafile)
    solh5 = tempname() * ".h5"
    nt = rand(3: 7)
    h5write(solh5, "t", collect(1: nt))
    h5write(solh5, "x", rand(mf.nx, mf.nξ, nt))
    h5write(solh5, "y", rand(length(ma.cx), 6, nt))
    pvds = gen_pvd(mf, mafile, solh5, "t", ["x"], ["y"], 1: nt, tempname())
    @test length(pvds) == 3nt + 1
end

@testset "LoopVectorization GEMV" begin
    a1 = rand(5, 5)
    b1 = rand(5)
    c1 = similar(b1)
    a2 = copy(a1)
    b2 = copy(b1)
    c2 = similar(c1)
    α, β = rand(), rand()
    mul!(c1, a1, b1, α, β)
    Oetqf.AmulB!(c2, a2, b2, α, β)
    @test c1 ≈ c2
end