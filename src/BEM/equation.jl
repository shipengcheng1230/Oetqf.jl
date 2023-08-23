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
    @batch for j ∈ axes(v, 2)
        for i ∈ axes(v, 1)
            alloc.relvnp[i,j] = v[i,j] - vpl # no zero paddings, used for `LinearAlgebra.BLAS`
            alloc.relv[i,j] = alloc.relvnp[i,j] # there exists zero paddings in `alloc.relv`
        end
    end
end

@inline function dτ_dt!(gf::AbstractArray{T, 3}, alloc::TractionRateAllocFFTConv) where {T<:Complex}
    mul!(alloc.relv_dft, alloc.pf, alloc.relv)
    fill!(alloc.dτ_dt_dft, zero(T))
    # potential simplification by using https://github.com/mcabbott/Tullio.jl
    @batch for j ∈ axes(gf, 2)
        for l ∈ axes(gf, 3)
            for i ∈ axes(gf, 1)
                alloc.dτ_dt_dft[i,j] += gf[i,j,l] * alloc.relv_dft[i,l]
            end
        end
    end
    ldiv!(alloc.dτ_dt_buffer, alloc.pf, alloc.dτ_dt_dft)
    @batch for j ∈ axes(alloc.dτ_dt, 2)
        for i ∈ axes(alloc.dτ_dt, 1)
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
    gf::AbstractArray,
    p::RateStateQuasiDynamicProperty,
    dila::DilatancyProperty,
    u0::ArrayPartition, tspan::NTuple{2};
    se::StateEvolutionLaw=DieterichStateLaw(), kwargs...)

    alloc = gen_alloc(Val(:BEMFault), size(u0.x[1], 1), size(u0.x[1], 2); kwargs...)
    return ODEProblem{true}(ode, u0, tspan, (p, dila, alloc, gf, se))
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

function assemble(
    gf₁₁::AbstractArray,
    gf₁₂::AbstractMatrix,
    gf₂₁::AbstractMatrix,
    gf₂₂::AbstractMatrix,
    pf::RateStateQuasiDynamicProperty,
    pa::ViscosityProperty,
    dila::DilatancyProperty,
    u0::ArrayPartition, tspan::NTuple{2};
    se::StateEvolutionLaw=DieterichStateLaw(), kwargs...)

    alloc₁ = gen_alloc(Val(:BEMFault), size(u0.x[1])...; kwargs...)
    alloc₂ = gen_alloc(Val(:BEMMantle), size(u0.x[3], 1))
    return ODEProblem{true}(ode, u0, tspan, (pf, pa, dila, alloc₁, alloc₂, gf₁₁, gf₁₂, gf₂₁, gf₂₂, se))
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

function ode(du::T, u::T, p::Tuple{P1, P2, AL, A, SE}, t::U
    ) where {
        T, U,
        P1<:RateStateQuasiDynamicProperty, P2<:DilatancyProperty,
        AL<:TractionRateAllocFFTConv,
        A, SE<:StateEvolutionLaw
    }

    v, θ, _, 𝓅 = u.x
    dv, dθ, dδ, d𝓅 = du.x
    prop, dila, alloc, gf, se = p

    relative_velocity!(alloc, prop.vpl, v)
    dτ_dt!(gf, alloc)
    update_fault_with_dilatancy!(prop, dila, alloc, v, θ, 𝓅, dv, dθ, dδ, d𝓅, se)
end

function ode(du::T, u::T, p::Tuple{P1, P2, AL1, AL2, A, U, U, U, SE}, t::V
    ) where {
        T, U, V, A,
        SE<:StateEvolutionLaw,
        P1<:RateStateQuasiDynamicProperty, P2<:ViscosityProperty,
        AL1<:TractionRateAllocFFTConv, AL2<:StressRateAllocMatrix
    }

    v, θ, _, σ, _ = u.x
    dv, dθ, dϵ, dσ, dδ = du.x
    pf, pa, alloc1, alloc2, gf₁₁, gf₁₂, gf₂₁, gf₂₂, se = p

    relative_velocity!(alloc1, pf.vpl, v)
    update_strain_rate!(pa, σ, dϵ)
    relative_strain_rate!(alloc2, dϵ, pa.dϵ₀)
    dτ_dt!(gf₁₁, alloc1) # fault - fault
    matvecmul!(vec(alloc1.dτ_dt), gf₂₁, vec(alloc2.reldϵ), true, true) # mantle - fault
    matvecmul!(vec(dσ), gf₁₂, vec(alloc1.relvnp)) # fault - mantle
    matvecmul!(vec(dσ), gf₂₂, vec(alloc2.reldϵ), true, true) # mantle - mantle
    update_fault!(pf, alloc1, v, θ, dv, dθ, dδ, se)
end

function ode(du::T, u::T, p::Tuple{P1, P2, Dila, AL1, AL2, A, U, U, U, SE}, t::V
    ) where {
        T, U, V, A,
        SE<:StateEvolutionLaw,
        P1<:RateStateQuasiDynamicProperty, P2<:ViscosityProperty, Dila<:DilatancyProperty,
        AL1<:TractionRateAllocFFTConv, AL2<:StressRateAllocMatrix
    }

    v, θ, _, σ, 𝓅, _ = u.x
    dv, dθ, dϵ, dσ, d𝓅, dδ = du.x
    pf, pa, dila, alloc1, alloc2, gf₁₁, gf₁₂, gf₂₁, gf₂₂, se = p

    relative_velocity!(alloc1, pf.vpl, v)
    update_strain_rate!(pa, σ, dϵ)
    relative_strain_rate!(alloc2, dϵ, pa.dϵ₀)
    dτ_dt!(gf₁₁, alloc1) # fault - fault
    matvecmul!(vec(alloc1.dτ_dt), gf₂₁, vec(alloc2.reldϵ), true, true) # mantle - fault
    matvecmul!(vec(dσ), gf₁₂, vec(alloc1.relvnp)) # fault - mantle
    matvecmul!(vec(dσ), gf₂₂, vec(alloc2.reldϵ), true, true) # mantle - mantle
    update_fault_with_dilatancy!(pf, dila, alloc1, v, θ, 𝓅, dv, dθ, dδ, d𝓅, se)
end


@inline function update_strain_rate!(p::ViscosityProperty, σ::T, dϵ::T) where T
    @batch for i ∈ axes(σ, 1)
        σkk = (σ[i,1] + σ[i,4] + σ[i,6]) / 3
        σxx = σ[i,1] - σkk
        σyy = σ[i,4] - σkk
        σzz = σ[i,6] - σkk
        σxy, σxz, σyz = σ[i,2], σ[i,3], σ[i,5]
        τnorm = sqrt(σxx^2 + σyy^2 + σzz^2 + 2 * (σxy^2 + σxz^2 + σyz^2))
        dϵ[i,1] = dϵ_dt(p, i, σxx, τnorm)
        dϵ[i,2] = dϵ_dt(p, i, σxy, τnorm)
        dϵ[i,3] = dϵ_dt(p, i, σxz, τnorm)
        dϵ[i,4] = dϵ_dt(p, i, σyy, τnorm)
        dϵ[i,5] = dϵ_dt(p, i, σyz, τnorm)
        dϵ[i,6] = dϵ_dt(p, i, σzz, τnorm)
    end
end

@inline function relative_strain_rate!(alloc::StressRateAllocMatrix, dϵ::AbstractMatrix, dϵ₀::AbstractVector)
    @batch for i ∈ axes(dϵ, 1)
        for j ∈ axes(dϵ, 2)
            alloc.reldϵ[i,j] = dϵ[i,j] - dϵ₀[j]
        end
    end
end

# rate-and-state update
@inline function update_fault!(
    p::RateStateQuasiDynamicProperty, alloc::TractionRateAllocFFTConv,
    v::T, θ::T, dv::T, dθ::T, dδ::T, se::StateEvolutionLaw) where T

    @batch for i ∈ eachindex(v)
        ψ1 = exp((p.f₀ + p.b[i] * log(p.v₀ * max(zero(eltype(θ)), θ[i]) / p.L[i])) / p.a[i]) / 2p.v₀
        ψ2 = p.σ[i] * ψ1 / hypot(1, v[i] * ψ1)
        dμ_dv = p.a[i] * ψ2
        dμ_dθ = p.b[i] / θ[i] * v[i] * ψ2
        dθ[i] = dθ_dt(se, v[i], θ[i], p.L[i])
        dv[i] = (alloc.dτ_dt[i] - dμ_dθ * dθ[i]) / (dμ_dv + p.η)
        dδ[i] = v[i]
    end
end

@inline function update_fault_with_dilatancy!(
    p::RateStateQuasiDynamicProperty, dila::DilatancyProperty,
    alloc::TractionRateAllocFFTConv,
    v::T, θ::T, 𝓅::T, dv::T, dθ::T, dδ::T, d𝓅::T, se::StateEvolutionLaw) where T

    @batch for i ∈ eachindex(v)
        dθ[i] = dθ_dt(se, v[i], θ[i], p.L[i])
        d𝓅[i] = d𝓅_dt(dila, i, 𝓅[i], θ[i], dθ[i])

        aᶠ = p.a[i] / p.f₀
        bᶠ = p.b[i] / p.f₀
        vᶠ = max(zero(eltype(v)), v[i] / p.v₀)
        θᶠ = max(zero(eltype(θ)), θ[i] * p.v₀ / p.L[i])
        vᶠᵃ⁻¹ = vᶠ ^ (aᶠ - 1)
        θᶠᵇ⁻¹ = θᶠ ^ (bᶠ - 1)
        vᶠᵃ = vᶠ ^ aᶠ
        θᶠᵇ = θᶠ ^ bᶠ

        dv[i] = (
            alloc.dτ_dt[i] +
            p.f₀ * d𝓅[i] * vᶠᵃ * θᶠᵇ -
            p.f₀ * (p.σ[i] - 𝓅[i]) * vᶠᵃ * θᶠᵇ⁻¹ * bᶠ * p.v₀ / p.L[i] * dθ[i]
        ) / (
            p.f₀ * (p.σ[i] - 𝓅[i]) * vᶠᵃ⁻¹ * θᶠᵇ * aᶠ / p.v₀
        )
        dδ[i] = v[i]
    end
end

# evolution law
@inline dθ_dt(::DieterichStateLaw, v::T, θ::T, L::T) where T = 1 - v * θ / L

# dilantancy law
@inline d𝓅_dt(dila::DilatancyProperty, i::Int, 𝓅::T, θ::T, dθ::T) where T = -(𝓅 - dila.p₀[i]) / dila.tₚ[i] + dila.ϵ[i] / dila.β[i] / θ * dθ

# viscosity law
@inline dϵ_dt(p::PowerLawViscosityProperty, i::Int, σ::T, τnorm::T) where T = p.γ[i] * σ * τnorm ^ p.n[i]
@inline function dϵ_dt(p::CompositePowerLawViscosityProperty, i::Int, σ::T, τnorm::T) where T
    ans = zero(T)
    for pp ∈ p.piter
        ans += dϵ_dt(pp, i, σ, τnorm)
    end
    ans
end