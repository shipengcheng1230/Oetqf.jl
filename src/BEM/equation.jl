# temp allocation
abstract type ODEAllocation end

struct TractionRateAllocFFTConv{T, U, P<:FFTW.Plan} <: ODEAllocation
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

function gen_alloc(::Val{:BEMFault}, nx::I, nξ::I; T=Float64, fftw_flags::UInt32=FFTW.PATIENT) where {I<:Integer}
    x1 = Matrix{T}(undef, 2 * nx - 1, nξ)
    p1 = plan_rfft(x1, 1; flags=fftw_flags)

    return TractionRateAllocFFTConv(
        Matrix{T}(undef, nx, nξ),
        zeros(T, 2nx-1, nξ), zeros(T, nx, nξ), # for relative velocity, including zero
        [Matrix{Complex{T}}(undef, nx, nξ) for _ ∈ 1: 2]...,
        Matrix{T}(undef, 2nx-1, nξ),
        p1)
end

function gen_alloc(::Val{:BEMMantle}, n::Integer; T=Float64)
    StressRateAllocMatrix(Matrix{T}(undef, n, 6))
end

# ode parts
@inline function relative_velocity!(alloc::TractionRateAllocFFTConv, vpl::T, v::AbstractMatrix{T}) where T
    @inbounds @batch for j ∈ axes(v, 2)
        @simd for i ∈ axes(v, 1)
            alloc.relvnp[i,j] = v[i,j] - vpl # no zero paddings, used for `LinearAlgebra.BLAS`
            alloc.relv[i,j] = alloc.relvnp[i,j] # there exists zero paddings in `alloc.relv`
        end
    end
end

@inline function dτ_dt!(gf::AbstractArray{T, 3}, alloc::TractionRateAllocFFTConv) where {T<:Complex}
    mul!(alloc.relv_dft, alloc.pf, alloc.relv)
    fill!(alloc.dτ_dt_dft, zero(T))
    # potential simplification by using https://github.com/mcabbott/Tullio.jl
    @inbounds @batch for j ∈ axes(gf, 2)
        for l ∈ axes(gf, 3)
            @simd for i ∈ axes(gf, 1)
                alloc.dτ_dt_dft[i,j] += gf[i,j,l] * alloc.relv_dft[i,l]
            end
        end
    end
    ldiv!(alloc.dτ_dt_buffer, alloc.pf, alloc.dτ_dt_dft)
    @inbounds @batch for j ∈ axes(alloc.dτ_dt, 2)
        @simd for i ∈ axes(alloc.dτ_dt, 1)
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
    return ODEProblem{true}(ode, u0, tspan, (p, alloc, gf, se))
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

    alloc₁ = gen_alloc(Val(:BEMFault), size(u0.x[1])...; kwargs...)
    alloc₂ = gen_alloc(Val(:BEMMantle), size(u0.x[3], 1))
    return ODEProblem{true}(ode, u0, tspan, (pf, pa, alloc₁, alloc₂, gf₁₁, gf₁₂, gf₂₁, gf₂₂, se))
end

function ode(du::T, u::T, p::Tuple{P, AL, A, SE}, t::U
    ) where {T, U, P<:AbstractProperty, AL<:ODEAllocation, A, SE<:StateEvolutionLaw}

    v, θ, _ = u.x
    dv, dθ, dδ = du.x
    prop, alloc, gf, se = p

    relative_velocity!(alloc, prop.vpl, v)
    dτ_dt!(gf, alloc)
    update_fault!(prop, alloc, v, θ, dv, dθ, dδ, se)
end

function ode(du::T, u::T, p::Tuple{P1, P2, AL1, AL2, A, U, U, U, SE}, t::V
    ) where {T, U, V, A, SE<:StateEvolutionLaw, P1<:AbstractProperty, P2<:AbstractProperty, AL1<:ODEAllocation, AL2<:ODEAllocation}

    v, θ, _, σ, _ = u.x
    dv, dθ, dϵ, dσ, dδ = du.x
    pf, pa, alloc1, alloc2, gf₁₁, gf₁₂, gf₂₁, gf₂₂, se = p

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
    @inbounds @batch for i ∈ axes(σ, 1)
        σkk = (σ[i,1] + σ[i,4] + σ[i,6]) / 3
        σxx = σ[i,1] - σkk
        σyy = σ[i,4] - σkk
        σzz = σ[i,6] - σkk
        σxy, σxz, σyz = σ[i,2], σ[i,3], σ[i,5]
        τⁿ = (sqrt(σxx^2 + σyy^2 + σzz^2 + 2 * (σxy^2 + σxz^2 + σyz^2))) ^ p.n[i]
        dϵ[i,1] = dϵ_dt(p.γ[i], σxx, τⁿ)
        dϵ[i,2] = dϵ_dt(p.γ[i], σxy, τⁿ)
        dϵ[i,3] = dϵ_dt(p.γ[i], σxz, τⁿ)
        dϵ[i,4] = dϵ_dt(p.γ[i], σyy, τⁿ)
        dϵ[i,5] = dϵ_dt(p.γ[i], σyz, τⁿ)
        dϵ[i,6] = dϵ_dt(p.γ[i], σzz, τⁿ)
    end
end

@inline function relative_strain_rate!(alloc::StressRateAllocMatrix, dϵ::AbstractMatrix, dϵ₀::AbstractVector)
    @inbounds @batch for i ∈ axes(dϵ, 1)
        @simd for j ∈ axes(dϵ, 2)
            alloc.reldϵ[i,j] = dϵ[i,j] - dϵ₀[j]
        end
    end
end

# rate-and-state update
@inline function update_fault!(
    p::RateStateQuasiDynamicProperty, alloc::TractionRateAllocFFTConv,
    v::T, θ::T, dv::T, dθ::T, dδ::T, se::StateEvolutionLaw) where T

    @inbounds @batch for i ∈ eachindex(v)
        ψ1 = exp((p.f0 + p.b[i] * log(p.v0 * max(zero(eltype(θ)), θ[i]) / p.L[i])) / p.a[i]) / 2p.v0
        ψ2 = p.σ[i] * ψ1 / hypot(1, v[i] * ψ1)
        dμ_dv = p.a[i] * ψ2
        dμ_dθ = p.b[i] / θ[i] * v[i] * ψ2
        dθ[i] = dθ_dt(se, v[i], θ[i], p.L[i])
        dv[i] = (alloc.dτ_dt[i] - dμ_dθ * dθ[i]) / (dμ_dv + p.η)
        dδ[i] = v[i]
    end
end

# evolution law
@inline dθ_dt(::DieterichStateLaw, v::T, θ::T, L::T) where T = 1 - v * θ / L

# viscosity law
@inline dϵ_dt(γ::T, σ::T, τⁿ::T) where T = γ * σ * τⁿ
