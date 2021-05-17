# temp allocation
abstract type ODEAllocation end

struct TractionRateAllocFFTConv{T, U, P<:FFTW.Plan} <: ODEAllocation
    dims::Dims{2}
    dτ_dt::T # traction rate of interest
    relv::T # relative velocity including zero-padding
    relvnp::T # relative velocity excluding zero-padding area
    dτ_dt_dft::U # stress rate in discrete fourier domain
    relv_dft::U # relative velocity in discrete fourier domain
    dτ_dt_buffer::T # stress rate including zero-padding zone for fft
    pf::P # real-value-FFT forward operator
end

function gen_alloc(nx::I, nξ::I; T=Float64, fftw_flags=FFTW.PATIENT) where {I<:Integer}
    x1 = Matrix{T}(undef, 2 * nx - 1, nξ)
    p1 = plan_rfft(x1, 1, flags=fftw_flags)

    return TractionRateAllocFFTConv(
        (nx, nξ),
        Matrix{T}(undef, nx, nξ),
        zeros(T, 2nx-1, nξ), zeros(T, nx, nξ), # for relative velocity, including zero
        [Matrix{Complex{T}}(undef, nx, nξ) for _ in 1: 2]...,
        Matrix{T}(undef, 2nx-1, nξ),
        p1)
end

# ode parts
function relative_velocity!(alloc::TractionRateAllocFFTConv, vpl::T, v::AbstractMatrix{T}) where T
    @inbounds @fastmath @threads for j ∈ 1: alloc.dims[2]
        for i ∈ 1: alloc.dims[1]
            alloc.relv[i,j] = v[i,j] - vpl # there are zero paddings in `alloc.relv`
            alloc.relvnp[i,j] = alloc.relv[i,j] # copy-paste, useful for `LinearAlgebra.BLAS`
        end
    end
end

@inline function dτ_dt!(gf::AbstractArray{T, 3}, alloc::TractionRateAllocFFTConv) where {T<:Complex}
    mul!(alloc.relv_dft, alloc.pf, alloc.relv)
    fill!(alloc.dτ_dt_dft, zero(T))
    @inbounds @fastmath @threads for j ∈ 1: alloc.dims[2]
        for l ∈ 1: alloc.dims[2], i ∈ 1: alloc.dims[1]
            alloc.dτ_dt_dft[i,j] += gf[i,j,l] * alloc.relv_dft[i,l]
        end
    end
    ldiv!(alloc.dτ_dt_buffer, alloc.pf, alloc.dτ_dt_dft)
    @inbounds @fastmath @threads for j ∈ 1: alloc.dims[2]
        for i ∈ 1: alloc.dims[1]
            alloc.dτ_dt[i,j] = alloc.dτ_dt_buffer[i,j]
        end
    end
end

# build ode
function assemble(gf::AbstractArray, p::RateStateQuasiDynamicProperty, u0::ArrayPartition, tspan::NTuple{2};
    se::StateEvolutionLaw=DieterichStateLaw())

    alloc = gen_alloc(size(u0.x[1], 1), size(u0.x[1], 2))
    f! = (du, u, p, t) -> ode(du, u, p, t, alloc, gf, se)
    return ODEProblem(f!, u0, tspan, p)
end

function ode(du::ArrayPartition{T}, u::ArrayPartition{T}, p::RateStateQuasiDynamicProperty, t::U,
    alloc::TractionRateAllocFFTConv, gf::AbstractArray, se::StateEvolutionLaw,
    ) where {T, U}
    v, θ, _ = u.x
    dv, dθ, dδ = du.x
    clamp!(θ, zero(T), Inf)
    clamp!(v, zero(T), Inf)
    relative_velocity!(alloc, p.vpl, v)
    dτ_dt!(gf, alloc)
    @inbounds @fastmath @threads for i ∈ eachindex(v)
        ψ1 = exp((p.f0 + p.b[i] * log(p.v0 * θ[i] / p.L[i])) / p.a[i]) / 2p.v0
        ψ2 = p.σ[i] * ψ1 / hypot(1, v[i] * ψ1)
        dμ_dv = p.a[i] * ψ2
        dμ_dθ = p.b[i] / θ[i] * v[i] * ψ2
        dθ[i] = dθ_dt(se, v[i], θ[i], p.L[i])
        dv[i] = (alloc.dτ_dt[i] - dμ_dθ * dθ[i]) / (dμ_dv + p.η)
        dδ[i] = v[i]
    end
end

# evolution law
dθ_dt(::DieterichStateLaw, v::T, θ::T, L::T) where T = @fastmath 1 - v * θ / L