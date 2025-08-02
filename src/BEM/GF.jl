# precompute Green's functions

abstract type FaultType end
struct StrikeSlip <: FaultType end
struct DipSlip <: FaultType end

# multi-threading
"""
    stress_greens_function(mesh::RectOkadaMesh, λ::T, μ::T;
        ftype::FaultType=StrikeSlip(), fourier::Bool=true,
        nrept::Integer=2, buffer_ratio::Real=0, fftw_flags::UInt32=FFTW.PATIENT
    ) where {T <: Real}
     
This function computes the stress Green's function for a rectangular Okada mesh fault.

## Arguments
- `mesh::RectOkadaMesh`: the rectangular Okada mesh object
- `λ::T`: Lamé's first parameter
- `μ::T`: shear modulus
- `ftype::FaultType=StrikeSlip()`: type of fault, either `StrikeSlip` or `DipSlip`
- `fourier::Bool=true`: if true, return the Fourier transform of the Green's function
- `nrept::Integer=2`: number of repetitions for the dislocation
- `buffer_ratio::Real=0`: ratio of buffer zone around the mesh
- `fftw_flags::UInt32=FFTW.PATIENT`: flags for FFTW plan

## Returns
- If `fourier` is true, returns the Fourier transform of the stress Green's function, 
    otherwise, returns the stress Green's function as a 3D array. The array considers the 
    translational symmetry of the transfinite mesh.
"""
function stress_greens_function(mesh::RectOkadaMesh, λ::T, μ::T;
    ftype::FaultType=StrikeSlip(), fourier::Bool=true,
    nrept::Integer=2, buffer_ratio::Real=0, fftw_flags::UInt32=FFTW.PATIENT
    ) where {T <: Real}

    @assert buffer_ratio ≥ 0 "Argument `buffer_ratio` must be ≥ 0."
    ud = unit_dislocation(ftype)
    lrept = (buffer_ratio + one(T)) * (mesh.Δx * mesh.nx)
    α = (λ + μ) / (λ + 2μ)

    st = Array{T,3}(undef, mesh.nx, mesh.nξ, mesh.nξ)
    @inbounds @threads for l ∈ eachindex(mesh.ξ)
        cache = GeoGreensFunctions.dc3d_cache(T)
        u = Vector{T}(undef, 12)
        for j ∈ eachindex(mesh.ξ), i ∈ eachindex(mesh.x)
            fill!(u, zero(T))
            for p ∈ -nrept: nrept
                jump = p * lrept
                dc3d(mesh.x[i], mesh.y[j], mesh.z[j],
                    α, mesh.dep, mesh.dip,
                    mesh.ax[1][1] + jump, mesh.ax[1][2] + jump,
                    mesh.aξ[l][1], mesh.aξ[l][2],
                    ud[1], ud[2], ud[3], cache)
                u .+= cache[1]
            end
            st[i,j,l] = shear_traction_dc3d(ftype, u, λ, μ, mesh.dip)
        end
    end

    if fourier
        x1 = Vector{T}(undef, 2mesh.nx - 1)
        p1 = plan_rfft(x1, flags=fftw_flags)
        st_dft = Array{Complex{T},3}(undef, mesh.nx, mesh.nξ, mesh.nξ)
        for l ∈ eachindex(mesh.ξ), j ∈ eachindex(mesh.ξ)
            # toeplitz matrices
            st_dft[:,j,l] .= p1 * [st[:,j,l]; reverse(st[2:end,j,l])]
        end
        return st_dft
    end
    return st
end

@inline unit_dislocation(::StrikeSlip, T=Float64) = (one(T), zero(T), zero(T))
@inline unit_dislocation(::DipSlip, T=Float64) = (zero(T), one(T), zero(T))

@inline function shear_traction_dc3d(::StrikeSlip, u::AbstractVector, λ::T, μ::T, dip::T) where T
    σxy = μ * (u[5] + u[7])
    σxz = μ * (u[6] + u[10])
    -σxy * sind(dip) + σxz * cosd(dip)
end

@inline function shear_traction_dc3d(::DipSlip, u::AbstractVector, λ::T, μ::T, dip::T) where T
    σzz = (λ + 2μ) * u[12] + λ * u[4] + λ * u[8]
    σyy = (λ + 2μ) * u[8] + λ * u[4] + λ * u[12]
    σyz = μ * (u[11] + u[9])
    (σzz - σyy) / 2 * sind(2dip) + σyz * cosd(2dip)
end

@inline function shear_traction_vol(::StrikeSlip, σ::AbstractVector, dip::T) where T
    # think carefully about the direction
    -σ[2] * sind(dip) + σ[3] * cosd(dip)
end

@inline function shear_traction_vol(::DipSlip, σ::AbstractVector, dip::T) where T
    (σ[6] - σ[4]) / 2 * sind(2dip) + σ[5] * cosd(2dip)
end

"""
    stress_greens_function(
        mf::RectOkadaMesh, ma::BEMHex8Mesh,
        λ::T, μ::T;
        ftype::FaultType=StrikeSlip(),
        qtype::Union{String, Tuple{AbstractVecOrMat, AbstractVector}}="Gauss1",
        nrept::Integer=2, buffer_ratio::Real=0,
    ) where {T}

This function computes the stress Green's function for a source in Okada mesh to a receiver on Hex8 mesh.

## Arguments
- `mf::RectOkadaMesh`: the rectangular Okada mesh object
- `ma::BEMHex8Mesh`: the BEM Hex8 mesh object
- `λ::T`: Lamé's first parameter            
- `μ::T`: shear modulus
- `ftype::FaultType=StrikeSlip()`: type of fault, either `StrikeSlip` or `DipSlip`
- `qtype::Union{String, Tuple{AbstractVecOrMat, AbstractVector}}`: quadrature type for integration, can be a string or a tuple of local coordinates and weights
- `nrept::Integer=2`: number of repetitions for the dislocation
- `buffer_ratio::Real=0`: ratio of buffer zone around the mesh  

## Returns
- A 2D array of stress Green's functions, where each column corresponds to a source fault patch 
and each row corresponds to a receiver mantle patch.
"""
function stress_greens_function(
    mf::RectOkadaMesh, ma::BEMHex8Mesh,
    λ::T, μ::T;
    ftype::FaultType=StrikeSlip(),
    qtype::Union{String, Tuple{AbstractVecOrMat, AbstractVector}}="Gauss1",
    nrept::Integer=2, buffer_ratio::Real=0,
    ) where {T}

    @assert buffer_ratio ≥ 0 "Argument `buffer_ratio` must be ≥ 0."
    nElem = length(ma.cx)
    nDisl = mf.nx * mf.nξ
    i2s = CartesianIndices((mf.nx, mf.nξ))
    lrept = (buffer_ratio + one(T)) * (mf.Δx * mf.nx)
    α = (λ + μ) / (λ + 2μ)
    ud = unit_dislocation(ftype)
    localCoords, weights = get_quadrature(qtype)

    st = zeros(T, 6nElem, nDisl)
    @inbounds @threads for j ∈ axes(st, 2) # source fault patch
        cache = GeoGreensFunctions.dc3d_cache(T)
        u = Vector{T}(undef, 12)
        q = i2s[j]
        for i ∈ eachindex(ma.cx) # receiver mantle volume
            for w ∈ eachindex(weights)
                lx = localCoords[3w - 2]
                ly = localCoords[3w - 1]
                lz = localCoords[3w]
                rx = ma.cx[i] + lx * ma.Δx[i] / 2
                ry = ma.cy[i] + ly * ma.Δy[i] / 2
                rz = ma.cz[i] + lz * ma.Δz[i] / 2
                fill!(u, zero(T))
                for p ∈ -nrept: nrept
                    jump = p * lrept
                    dc3d(rx, ry, rz,
                        α, mf.dep, mf.dip,
                        mf.ax[q[1]][1] + jump, mf.ax[q[1]][2] + jump,
                        mf.aξ[q[2]][1], mf.aξ[q[2]][2],
                        ud[1], ud[2], ud[3], cache)
                    u .+= cache[1]
                end
                λϵkk = λ * (u[4] + u[8] + u[12])
                st[i, j]          += weights[w] * (λϵkk + 2μ * u[4])   # σxx
                st[i + 1nElem, j] += weights[w] * (μ * (u[5] + u[7]))  # σxy
                st[i + 2nElem, j] += weights[w] * (μ * (u[6] + u[10])) # σxz
                st[i + 3nElem, j] += weights[w] * (λϵkk + 2μ * u[8])   # σyy
                st[i + 4nElem, j] += weights[w] * (μ * (u[9] + u[11])) # σyz
                st[i + 5nElem, j] += weights[w] * (λϵkk + 2μ * u[12])  # σzz
            end
        end
    end
    return st
end

"""
    stress_greens_function(ma::BEMHex8Mesh, mf::RectOkadaMesh, λ::T, μ::T;
        ftype::FaultType=StrikeSlip(),
    ) where {T}

This function computes the stress Green's function for a source in Hex8 mesh to a receiver on Okada mesh.

## Arguments
- `ma::BEMHex8Mesh`: the BEM Hex8 mesh object
- `mf::RectOkadaMesh`: the rectangular Okada mesh object
- `λ::T`: Lamé's first parameter
- `μ::T`: shear modulus
- `ftype::FaultType=StrikeSlip()`: type of fault, either `StrikeSlip` or `DipSlip`

## Returns
- A 2D array of stress Green's functions, where each column corresponds to a source
mantle patch and each row corresponds to a receiver fault patch.
"""
function stress_greens_function(
    ma::BEMHex8Mesh, mf::RectOkadaMesh,
    λ::T, μ::T;
    ftype::FaultType=StrikeSlip(),
    ) where {T}

    nElem = length(ma.cx)
    nDisl = mf.nx * mf.nξ
    st = zeros(T, nDisl, 6nElem)
    ν = λ / 2 / (λ + μ)
    i2s = CartesianIndices((mf.nx, mf.nξ))

    for p ∈ 1:6
        epsv = zeros(T, 6)
        epsv[p] = one(T)

        @inbounds @threads for j ∈ eachindex(ma.cx) # source mantle volume
            temp = Vector{T}(undef, 6)
            jcol = (p - 1) * nElem + j
            for i ∈ axes(st, 1) # receiver fault patch
                q = i2s[i]
                stress_vol_hex8!(temp,
                    mf.x[q[1]], mf.y[q[2]], mf.z[q[2]], # receiver
                    ma.qx[j], ma.qy[j], ma.qz[j], # source
                    ma.Δx[j], ma.Δy[j], ma.Δz[j], # T-L-W <=> x-y-z
                    zero(T), # no rotation
                    epsv[1], epsv[2], epsv[3], epsv[4], epsv[5], epsv[6],
                    μ, ν)
                st[i, jcol] = shear_traction_vol(ftype, temp, mf.dip)
            end
        end
    end
    return st
end

"""
    stress_greens_function(
        mesh::BEMHex8Mesh,
        λ::T, μ::T;
        qtype::Union{String, Tuple{AbstractVecOrMat, AbstractVector}}="Gauss1",
        checkeigvals::Bool=true,
    ) where {T}

This function computes the stress Green's function for a BEM Hex8 mesh.

## Arguments
- `mesh::BEMHex8Mesh`: the BEM Hex8 mesh object
- `λ::T`: Lamé's first parameter
- `μ::T`: shear modulus
- `qtype::Union{String, Tuple{AbstractVecOrMat, AbstractVector}}` quadrature type for integration, can be a string or a tuple of local coordinates and weights
- `checkeigvals::Bool=true`: if true, check the eigenvalues of the resulting stress matrix and print the maximum real part   

## Returns
- A 2D array of stress Green's functions, where each column corresponds to a source
mantle patch and each row corresponds to another mantle patch.
"""
function stress_greens_function(
    mesh::BEMHex8Mesh,
    λ::T, μ::T;
    qtype::Union{String, Tuple{AbstractVecOrMat, AbstractVector}}="Gauss1",
    checkeigvals::Bool=true,
    ) where {T}

    nElem = length(mesh.cx)
    st = zeros(T, 6nElem, 6nElem)
    ν = λ / 2 / (λ + μ)
    localCoords, weights = get_quadrature(qtype)

    for p ∈ 1:6 # xx, xy, xz, yy, yz, zz
        epsv = zeros(T, 6)
        epsv[p] = one(T)

        @inbounds @threads for i ∈ eachindex(mesh.cx) # source
            temp = Vector{T}(undef, 6)
            icol = (p - 1) * nElem + i
            for j ∈ eachindex(mesh.cx) # receiver
                for q ∈ eachindex(weights)
                    lx = localCoords[3q - 2]
                    ly = localCoords[3q - 1]
                    lz = localCoords[3q]
                    rx = mesh.cx[j] + lx * mesh.Δx[j] / 2
                    ry = mesh.cy[j] + ly * mesh.Δy[j] / 2
                    rz = mesh.cz[j] + lz * mesh.Δz[j] / 2
                    stress_vol_hex8!(temp,
                        rx, ry, rz, # receiver
                        mesh.qx[i], mesh.qy[i], mesh.qz[i], # source
                        mesh.Δx[i], mesh.Δy[i], mesh.Δz[i], # T-L-W <=> x-y-z
                        zero(T), # no rotation
                        epsv[1], epsv[2], epsv[3], epsv[4], epsv[5], epsv[6],
                        μ, ν)
                    for k ∈ 1:6
                        st[(k - 1) * nElem + j, icol] += temp[k] * weights[q]
                    end
                end
            end
        end
    end
    if checkeigvals
        maxrλ = maximum(real, eigvals(st))
        @printf "Maximum real part of eigval is: %.4f\n" maxrλ
    end
    return st
end

const QuadratureType = Tuple{AbstractVecOrMat, AbstractVector}

"""
    gmsh_quadrature(etype::Integer, qtype::String)::QuadratureType

This function retrieves the integration points and weights for a given element type and quadrature type from GMSH.

## Arguments
- `etype::Integer`: the element type ID in GMSH
- `qtype::String`: the quadrature type, e.g., "Gauss1"

## Returns
- A tuple containing the local coordinates and weights for the quadrature points.
"""
function gmsh_quadrature(etype::Integer, qtype::String)::QuadratureType
    @gmsh_do begin
        return gmsh.model.mesh.getIntegrationPoints(etype, qtype)
    end
end

function get_quadrature(qtype::String)::QuadratureType
    # only for Hex8
    localCoords, weights = gmsh_quadrature(5, qtype)
    weights ./= sum(weights)
    (localCoords, weights)
end

function get_quadrature(qtype::QuadratureType)
    @assert length(qtype[1]) == 3 * length(qtype[2]) "Wrong format of quadrature!"
    qtype
end