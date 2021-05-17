# precompute Green's functions

abstract type FaultType end
struct StrikeSlip <: FaultType end

# multi-threading
function stress_greens_function(mesh::RectOkadaMesh, λ::T, μ::T;
    ftype::FaultType=StrikeSlip(), fourier::Bool=true,
    nrept::Integer=2, buffer_ratio::Real=0, fftw_flags::UInt32=FFTW.ESTIMATE
    ) where {T<:Real}

    @assert buffer_ratio ≥ 0 "Argument `buffer_ratio` must be ≥ 0."
    ud = unit_dislocation(ftype)
    lrept = (buffer_ratio + one(T)) * (mesh.Δx * mesh.nx)
    α = (λ + μ) / (λ + 2μ)

    st = Array{T, 3}(undef, mesh.nx, mesh.nξ, mesh.nξ)
    @inbounds @threads for l ∈ 1: mesh.nξ
        u = Vector{T}(undef, 12)
        for j ∈ 1: mesh.nξ, i ∈ 1: mesh.nx
            fill!(u, zero(T))
            for p ∈ -nrept: nrept
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
        st_dft = Array{Complex{T}, 3}(undef, mesh.nx, mesh.nξ, mesh.nξ)
        for l ∈ 1: mesh.nξ, j ∈ 1: mesh.nξ
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