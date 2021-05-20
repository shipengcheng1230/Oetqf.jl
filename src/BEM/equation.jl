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

struct StressRateAllocMatrix{T} <: ODEAllocation
    reldϵ::T
end

function gen_alloc(::Val{:BEMFault}, nx::I, nξ::I; T=Float64, fftw_flags=FFTW.MEASURE) where {I<:Integer}
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

function gen_alloc(::Val{:BEMMantle}, n::Integer; T=Float64)
    StressRateAllocMatrix(Matrix{T}(undef, n, 6))
end

# ode parts
@inline function relative_velocity!(alloc::TractionRateAllocFFTConv, vpl::T, v::AbstractMatrix{T}) where T
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
function assemble(
    gf::AbstractArray,
    p::RateStateQuasiDynamicProperty,
    u0::ArrayPartition, tspan::NTuple{2};
    se::StateEvolutionLaw=DieterichStateLaw(), kwargs...)

    alloc = gen_alloc(Val(:BEMFault), size(u0.x[1], 1), size(u0.x[1], 2); kwargs...)
    f! = (du, u, p, t) -> ode(du, u, p, t, alloc, gf, se)
    return ODEProblem(f!, u0, tspan, p)
end

function assemble(
    gf₁₁::AbstractArray,
    gf₁₂::AbstractMatrix,
    gf₂₁::AbstractMatrix,
    gf₂₂::AbstractMatrix,
    pf::RateStateQuasiDynamicProperty,
    pa::ViscosityProperty,
    u0::ArrayPartition, tspan::NTuple{2};
    se::StateEvolutionLaw=DieterichStateLaw(), kwargs...)

    alloc1 = gen_alloc(Val(:BEMFault), size(u0.x[1])...; kwargs...)
    alloc2 = gen_alloc(Val(:BEMMantle), size(u0.x[3], 1))
    f! = (du, u, p, t) -> ode(du, u, p, t, alloc1, alloc2, gf₁₁, gf₁₂, gf₂₁, gf₂₂, se)
    return ODEProblem(f!, u0, tspan, (pf, pa))
end

function ode(du::ArrayPartition{T}, u::ArrayPartition{T}, p::RateStateQuasiDynamicProperty, t::U,
    alloc::TractionRateAllocFFTConv, gf::AbstractArray, se::StateEvolutionLaw,
    ) where {T, U}
    v, θ, _ = u.x
    dv, dθ, dδ = du.x
    # clamp!(θ, zero(T), Inf)
    # clamp!(v, zero(T), Inf)
    relative_velocity!(alloc, p.vpl, v)
    dτ_dt!(gf, alloc)
    update_fault!(p, alloc, v, θ, dv, dθ, dδ, se)
end

function ode(du::ArrayPartition{T}, u::ArrayPartition{T},
    p::Tuple{RateStateQuasiDynamicProperty, ViscosityProperty},
    t::U,
    alloc1::TractionRateAllocFFTConv,
    alloc2::StressRateAllocMatrix,
    gf₁₁::AbstractArray,
    gf₁₂::AbstractMatrix,
    gf₂₁::AbstractMatrix,
    gf₂₂::AbstractMatrix,
    se::StateEvolutionLaw,
    ) where {T, U}

    v, θ, _, σ, _ = u.x
    dv, dθ, dϵ, dσ, dδ = du.x
    pf, pa = p

    # clamp!(θ, zero(T), Inf)
    # clamp!(v, zero(T), Inf)
    relative_velocity!(alloc1, pf.vpl, v)
    update_strain_rate!(pa, σ, dϵ)
    relative_strain_rate!(alloc2, dϵ, pa.dϵ₀)
    dτ_dt!(gf₁₁, alloc1) # fault - fault
    mul!(vec(alloc1.dτ_dt), gf₂₁, vec(alloc2.reldϵ), true, true) # mantle - fault
    mul!(vec(dσ), gf₁₂, vec(alloc1.relvnp)) # fault - mantle
    mul!(vec(dσ), gf₂₂, vec(alloc2.reldϵ), true, true) # mantle - mantle
    update_fault!(pf, alloc1, v, θ, dv, dθ, dδ, se)
end

@inline function update_strain_rate!(p::ViscosityProperty, σ::T, dϵ::T) where T
    @inbounds @fastmath @threads for i ∈ 1: size(σ, 1)
        σkk = (σ[i,1] + σ[i,4] + σ[i,6]) / 3
        σxx = σ[i,1] - σkk
        σyy = σ[i,4] - σkk
        σzz = σ[i,6] - σkk
        σxy, σxz, σyz = σ[i,2], σ[i,3], σ[i,5]
        τⁿ = (sqrt(σxx^2 + σyy^2 + σzz^2 + 2 * (σxy^2 + σxz^2 + σyz^2))) ^ p.n[i]
        dϵ[i,1] = dϵ_dt(p, σxx, τⁿ, i)
        dϵ[i,2] = dϵ_dt(p, σxy, τⁿ, i)
        dϵ[i,3] = dϵ_dt(p, σxz, τⁿ, i)
        dϵ[i,4] = dϵ_dt(p, σyy, τⁿ, i)
        dϵ[i,5] = dϵ_dt(p, σyz, τⁿ, i)
        dϵ[i,6] = dϵ_dt(p, σzz, τⁿ, i)
    end
end

@inline function relative_strain_rate!(alloc::StressRateAllocMatrix, dϵ::AbstractMatrix, dϵ₀::AbstractVector)
    @inbounds for j ∈ 1: size(dϵ, 2)
        @fastmath @threads for i ∈ 1: size(dϵ, 1)
            alloc.reldϵ[i,j] = dϵ[i,j] - dϵ₀[j]
        end
    end
end

# rate-and-state update
@inline function update_fault!(
    p::RateStateQuasiDynamicProperty, alloc::TractionRateAllocFFTConv,
    v::T, θ::T, dv::T, dθ::T, dδ::T, se::StateEvolutionLaw) where T

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
@inline dθ_dt(::DieterichStateLaw, v::T, θ::T, L::T) where T = @fastmath 1 - v * θ / L

# viscosity law
Base.@propagate_inbounds dϵ_dt(p::PowerLawViscosityProperty, σ::T, τⁿ::T, i::I) where {T, I} = @fastmath p.γ[i] * σ * τⁿ
