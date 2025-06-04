abstract type AbstractMesh{dim} end
abstract type StructuredMesh{dim} <: AbstractMesh{dim} end
abstract type UnstructuredMesh{dim} <: AbstractMesh{dim} end

@with_kw struct RectOkadaMesh{T, U, I, S} <: StructuredMesh{2}
    x::T # centroid along strike
    Δx::U
    nx::I
    ax::S
    ξ::T # centroid along downdip
    Δξ::U
    nξ::I
    aξ::S
    y::T
    z::T
    dep::U # fault origin depth
    dip::U # fault dipping angle

    @assert length(x) == length(ax) == nx
    @assert length(ξ) == length(aξ) == nξ == length(y) == length(z)
end

"""
    gen_mesh(::Val{:RectOkada},
        x::T, ξ::T, Δx::T, Δξ::T, dip::T) where T

Generate a rectangular mesh for Okada's fault model in 2D.

## Arguments
- `x`: length of the fault along strike
- `ξ`: length of the fault along downdip
- `Δx`: cell size along strike
- `Δξ`: cell size along downdip
- `dip`: dipping angle of the fault in degrees
"""
function gen_mesh(::Val{:RectOkada}, x::T, ξ::T, Δx::T, Δξ::T, dip::T) where T
    ξ, nξ, aξ, y, z = _equidist_mesh_downdip(ξ, Δξ, dip)
    x, nx, ax = _equidist_mesh_strike(x, Δx)
    return RectOkadaMesh(x, Δx, nx, ax, ξ, Δξ, nξ, aξ, y, z, zero(T), dip)
end

function _equidist_mesh_downdip(ξ::T, Δξ::T, dip::T) where T
    ξi = range(zero(T), stop=-ξ+Δξ, step=-Δξ) .- Δξ/2 |> collect
    aξ = [(w - Δξ/2, w + Δξ/2) for w in ξi]
    y, z = ξi .* cosd(dip), ξi .* sind(dip)
    return ξi, length(ξi), aξ, y, z
end

function _equidist_mesh_strike(x::T, Δx::T) where T
    xi = range(-x/2 + Δx/2, stop=x/2 - Δx/2, step=Δx) |> collect
    ax = [(w - Δx/2, w + Δx/2) for w in xi]
    return xi, length(xi), ax
end

@with_kw struct BEMHex8Mesh{T, U<:Real} <: UnstructuredMesh{3}
    cx::T
    cy::T
    cz::T
    qx::T
    qy::T
    qz::T
    Δx::T
    Δy::T
    Δz::T
    θ::U

    @assert size(cx) == size(cy) == size(cz) == size(qx) == size(qy) == size(qz) ==
        size(Δx) == size(Δy) == size(Δz)
end

"""
    gen_gmsh_mesh(::Val{:BEMHex8Mesh},
        llx::T, lly::T, llz::T, dx::T, dy::T, dz::T, nx::I, ny::I, nz::I;
        rfx::T=one(T), rfy::T=one(T), rfzh::AbstractVector=ones(nz),
        rfxType::AbstractString="Bump", rfyType::AbstractString="Bump",
        output::AbstractString="temp.msh"
    ) where {T, I}

Gernate a box using 8-node hexahedron elements by vertically extruding transfinite curve on xy plane, allowing
    total flexibility on the mesh size in z direction, and refinement in xy plane.

## Arguments
- `llx`, `lly`, `llz`: coordinates of low-left corner on the top surface
- `dx`, `dy`, `dz`: x-, y-, z-extension
- `nx`, `ny`: number of cells along x-, y-axis
- `rfx`, `rfy`: refinement coefficients along x-, y-axis using **Bump** algorithm, please refer `gmsh.model.geo.mesh.setTransfiniteCurve`
- `rfzh`: accumulated height of cells along z-axis which will be normalized automatically, please refer `heights` in `gmsh.model.geo.extrude`
"""
function gen_gmsh_mesh(::Val{:BEMHex8Mesh},
    llx::T, lly::T, llz::T, dx::T, dy::T, dz::T, nx::I, ny::I, nz::I;
    rfx::T=one(T), rfy::T=one(T), rfzh::AbstractVector=ones(nz),
    rfxType::AbstractString="Bump", rfyType::AbstractString="Bump",
    output::AbstractString="temp.msh") where {T, I}

    @gmsh_do begin
        @addPoint begin
            llx,      lly,      llz, 0.0, 1
            llx + dx, lly,      llz, 0.0, 2
            llx + dx, lly + dy, llz, 0.0, 3
            llx,      lly + dy, llz, 0.0, 4
        end
        @addLine begin
            1, 2, 1
            2, 3, 2
            4, 3, 3
            1, 4, 4
        end
        gmsh.model.geo.add_curve_loop([1, 2, -3, -4], 1)
        gmsh.model.geo.add_plane_surface([1], 1)
        gmsh.model.geo.mesh.set_transfinite_surface(1)
        @setTransfiniteCurve begin
            1, nx+1, rfxType, rfx
            3, nx+1, rfxType, rfx
            2, ny+1, rfyType, rfy
            4, ny+1, rfyType, rfy
        end
        gmsh.model.geo.mesh.setRecombine(2, 1)
        gmsh.model.geo.extrude(
            [(2, 1)],
            0.0, 0.0, dz,
            ones(nz), normalize(cumsum(rfzh), Inf),
            true)
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(output)
    end
end

function gen_mesh(::Val{:BEMHex8Mesh}, file::AbstractString; rotation::Real=0.0, transpose::Bool=false)
    isfile(file) || ErrorException("Mesh file $(file) does not exist.")

    @gmsh_open file begin
        nodes = gmsh.model.mesh.get_nodes()
        elprop = gmsh.model.mesh.get_element_properties(5)
        elnumnode = elprop[4]
        els = gmsh.model.mesh.get_elements(3, -1)
        etags = els[2][1]
        econn = els[3][1]
        numels = length(etags)
        centers = gmsh.model.mesh.get_barycenters(5, -1, 0, 1)
        x = centers[1: 3: end]
        y = centers[2: 3: end]
        z = centers[3: 3: end]
        qx, qy, qz, Δx, Δy, Δz = [Vector{Float64}(undef, numels) for _ in 1: 6]

        for i in 1: numels
            ntag1 = econn[elnumnode*i-elnumnode+1]
            ntag2 = econn[elnumnode*i-elnumnode+2]
            ntag4 = econn[elnumnode*i-elnumnode+4]
            ntag5 = econn[elnumnode*i-elnumnode+5]
            if transpose
                ntag2, ntag4 = ntag4, ntag2
            end
            p1x, p1y, p1z = nodes[2][3*ntag1-2], nodes[2][3*ntag1-1], nodes[2][3*ntag1]
            p2x, p2y = nodes[2][3*ntag2-2], nodes[2][3*ntag2-1]
            p4x, p4y = nodes[2][3*ntag4-2], nodes[2][3*ntag4-1]
            p5z = nodes[2][3*ntag5]
            Δx[i] = hypot(p1x - p4x, p1y - p4y)
            Δy[i] = hypot(p1x - p2x, p1y - p2y)
            Δz[i] = abs(p1z - p5z)
            qx[i] = x[i]
            qy[i] = y[i] - Δy[i] / 2
            qz[i] = z[i] + Δz[i] / 2
        end
        return BEMHex8Mesh(x, y, z, qx, qy, qz, Δx, Δy, Δz, rotation)
    end
end