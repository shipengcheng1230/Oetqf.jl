# precompute Green's functions

abstract type FaultType end
struct StrikeSlip <: FaultType end

# multi-threading

function stress_greens_function(mesh::RectOkadaMesh, λ::T, μ::T;
    ftype::FaultType=StrikeSlip(), fourier::Bool=true,
    nrept::Integer=2, buffer_ratio::Real=0, fftw_flags::UInt32=FFTW.MEASURE
    ) where {T <: Real}

    @assert buffer_ratio ≥ 0 "Argument `buffer_ratio` must be ≥ 0."
    ud = unit_dislocation(ftype)
    lrept = (buffer_ratio + one(T)) * (mesh.Δx * mesh.nx)
    α = (λ + μ) / (λ + 2μ)

    st = Array{T,3}(undef, mesh.nx, mesh.nξ, mesh.nξ)
    @inbounds @threads for l ∈ 1:mesh.nξ
        u = Vector{T}(undef, 12)
        for j ∈ 1:mesh.nξ, i ∈ 1:mesh.nx
            fill!(u, zero(T))
            for p ∈ -nrept:nrept
                u .+= dc3d(
                    mesh.x[i], mesh.y[j], mesh.z[j],
                    α, mesh.dep, mesh.dip,
                    mesh.ax[1] .+ p * lrept,
                    mesh.aξ[l],
                    ud,
                    )
            end
            st[i,j,l] = shear_traction_dc3d(ftype, u, λ, μ, mesh.dip)
        end
    end

    if fourier
        x1 = Vector{T}(undef, 2mesh.nx - 1)
        p1 = plan_rfft(x1, flags=fftw_flags)
        st_dft = Array{Complex{T},3}(undef, mesh.nx, mesh.nξ, mesh.nξ)
        for l ∈ 1:mesh.nξ, j ∈ 1:mesh.nξ
            # toeplitz matrices
            st_dft[:,j,l] .= p1 * [st[:,j,l]; reverse(st[2:end,j,l])]
        end
        return st_dft
    end
    return st
end

@inline unit_dislocation(::StrikeSlip, T=Float64) = [one(T), zero(T), zero(T)]

@inline function shear_traction_dc3d(::StrikeSlip, u::AbstractVector, λ::T, μ::T, dip::T) where T
    σxy = μ * (u[5] + u[7])
    σxz = μ * (u[6] + u[10])
    -σxy * sind(dip) + σxz * cosd(dip)
end

@inline function shear_traction_vol(::StrikeSlip, σ::AbstractVector, dip::T) where T
    # think carefully about the direction
    -σ[2] * sind(dip) + σ[3] * cosd(dip)
end

#
function stress_greens_function(
    mf::RectOkadaMesh, ma::BEMHex8Mesh,
    λ::T, μ::T;
    ftype::FaultType=StrikeSlip(),
    qtype::String="Gauss1",
    nrept::Integer=2, buffer_ratio::Real=0,
    ) where {T}

    @assert buffer_ratio ≥ 0 "Argument `buffer_ratio` must be ≥ 0."
    nElem = length(ma.cx)
    nDisl = mf.nx * mf.nξ
    i2s = CartesianIndices((mf.nx, mf.nξ))
    lrept = (buffer_ratio + one(T)) * (mf.Δx * mf.nx)
    α = (λ + μ) / (λ + 2μ)
    ud = unit_dislocation(ftype)
    localCoords, weights = quadrature(5, qtype) # 5 denotes Hex8
    weights ./= sum(weights)

    st = zeros(T, 6nElem, nDisl)
    @inbounds @threads for j ∈ 1: nDisl # source fault patch
        u = Vector{T}(undef, 12)
        q = i2s[j]
        for i ∈ 1: nElem # receiver mantle volume
            for w in eachindex(weights)
                lx = localCoords[3w - 2]
                ly = localCoords[3w - 1]
                lz = localCoords[3w]
                rx = ma.cx[i] + lx * ma.Δx[i] / 2
                ry = ma.cy[i] + ly * ma.Δy[i] / 2
                rz = ma.cz[i] + lz * ma.Δz[i] / 2
                fill!(u, zero(T))
                for p ∈ -nrept: nrept
                    u .+= dc3d(
                        rx, ry, rz,
                        α, mf.dep, mf.dip,
                        mf.ax[q[1]] .+ p * lrept,
                        mf.aξ[q[2]],
                        ud,
                        )
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

        @inbounds @threads for j in 1: nElem # source mantle volume
            temp = Vector{T}(undef, 6)
            jcol = (p - 1) * nElem + j
            for i in 1: nDisl # receiver fault patch
                q = i2s[i]
                stress_vol_hex8!(temp,
                    mf.x[q[1]], mf.y[q[2]], mf.z[q[2]], # receiver
                    ma.qx[j], ma.qy[j], ma.qz[j], # source
                    ma.Δy[j], ma.Δx[j], ma.Δz[j], # L-T-W <=> y-x-z
                    zero(T), # no rotation
                    epsv[1], epsv[2], epsv[3], epsv[4], epsv[5], epsv[6],
                    μ, ν,
                )
                st[i, jcol] = shear_traction_vol(ftype, temp, mf.dip)
            end
        end
    end
    return st
end

function stress_greens_function(
    mesh::BEMHex8Mesh,
    λ::T, μ::T;
    qtype::String="Gauss1",
    ) where {T}

    nElem = length(mesh.cx)
    st = zeros(T, 6nElem, 6nElem)
    ν = λ / 2 / (λ + μ)

    localCoords, weights = quadrature(5, qtype) # 5 denotes Hex8
    weights ./= sum(weights)

    for p ∈ 1:6 # xx, xy, xz, yy, yz, zz
        epsv = zeros(T, 6)
        epsv[p] = one(T)

        @threads for i ∈ 1:nElem # source
            temp = Vector{T}(undef, 6)
            icol = (p - 1) * nElem + i
            for j ∈ 1: nElem # receiver
                @fastmath @inbounds for q in eachindex(weights)
                    lx = localCoords[3q - 2]
                    ly = localCoords[3q - 1]
                    lz = localCoords[3q]
                    rx = mesh.cx[j] + lx * mesh.Δx[j] / 2
                    ry = mesh.cy[j] + ly * mesh.Δy[j] / 2
                    rz = mesh.cz[j] + lz * mesh.Δz[j] / 2
                    stress_vol_hex8!(temp,
                        rx, ry, rz, # receiver
                        mesh.qx[i], mesh.qy[i], mesh.qz[i], # source
                        mesh.Δy[i], mesh.Δx[i], mesh.Δz[i], # L-T-W <=> y-x-z
                        zero(T), # no rotation
                        epsv[1], epsv[2], epsv[3], epsv[4], epsv[5], epsv[6],
                        μ, ν,
                    )
                    @simd for k ∈ 1:6
                        st[(k - 1) * nElem + j, icol] += temp[k] * weights[q]
                    end
                end
            end
        end
    end
    return st
end

function quadrature(etype::Integer, qtype::String)
    @gmsh_do begin
        return gmsh.model.mesh.getIntegrationPoints(etype, qtype)
    end
end