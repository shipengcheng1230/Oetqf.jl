import Base.fieldnames
import Base.==

abstract type StateEvolutionLaw end
struct DieterichStateLaw <: StateEvolutionLaw end

abstract type AbstractProperty end
abstract type ViscosityProperty <: AbstractProperty end

@with_kw struct RateStateQuasiDynamicProperty{T<:Real, U<:AbstractVecOrMat} <: AbstractProperty
    a::U # contrib from velocity
    b::U # contrib from state
    L::U # critical distance
    σ::U # effective normal stress
    η::T # radiation damping
    vpl::T # plate rate
    f₀::T = 0.6 # ref. frictional coeff
    v₀::T = 1e-6 # ref. velocity

    @assert size(a) == size(b) == size(L) == size(σ)
    @assert f₀ > 0
    @assert v₀ > 0
    @assert η > 0
    @assert vpl > 0
end

@with_kw struct DilatancyProperty{T} <: AbstractProperty
    tₚ::T # characteristic diffusion timescale
    ϵ::T # dilantancy coefficient
    β::T # fault gouge bulk compressibility
    p₀::T # ambient pore pressure
end

@with_kw struct PowerLawViscosityProperty{T, I, U} <: ViscosityProperty
    γ::T
    n::I # notice it is `power - 1`
    dϵ₀::U

    @assert length(dϵ₀) == 6
    @assert length(γ) == length(n)
end

@with_kw struct CompositePowerLawViscosityProperty{T<:AbstractVector, U} <: ViscosityProperty
    piter::T
    dϵ₀::U

    @assert length(dϵ₀) == 6
end

const _field_names = Dict(
    :RateStateQuasiDynamicProperty => (:a, :b, :L, :σ, :η, :vpl, :f₀, :v₀),
    :PowerLawViscosityProperty => (:γ, :n, :dϵ₀),
    :CompositePowerLawViscosityProperty => (:piter, :dϵ₀),
    :DilatancyProperty => (:tₚ, :ϵ, :β, :p₀),
)

# @assert mapreduce(Set, union, values(_field_names)) |> length == mapreduce(length, +, values(_field_names)) "Found duplicated property field names!"

for (nn, fn) ∈ _field_names
    @eval begin
        fieldnames(p::$(nn)) = $(fn)
        description(p::$(nn)) = String($(QuoteNode(nn)))
    end
end

function Base.:(==)(p1::P, p2::P) where P<:AbstractProperty
    reduce(&, [getfield(p1, name) == getfield(p2, name) for name ∈ fieldnames(p1)])
end

function struct_to_dict(p)
    Dict(name => getfield(p, name) for name ∈ fieldnames(p))
end

function struct_to_dict(p::CompositePowerLawViscosityProperty)
    Dict(:piter => [struct_to_dict(x) for x ∈ p.piter], :dϵ₀ => getfield(p, :dϵ₀))
end

function save_property(file::AbstractString, p::AbstractProperty)
    bson(file, struct_to_dict(p))
end

function save_property(file::AbstractString, p1::RateStateQuasiDynamicProperty, p2::ViscosityProperty)
    d = foldl(merge, map(struct_to_dict, (p1, p2)))
    bson(file, d)
end

function load_property(file::AbstractString, p::Symbol)
    d = BSON.load(file)
    load_property(d, p)
end

function load_property(d::AbstractDict, p::Symbol)
    @match p begin
        :RateStateQuasiDynamicProperty => RateStateQuasiDynamicProperty(
            d[:a], d[:b], d[:L], d[:σ], d[:η], d[:vpl], d[:f₀], d[:v₀],
        )
        :PowerLawViscosityProperty => PowerLawViscosityProperty(
            d[:γ], d[:n], d[:dϵ₀],
        )
        :CompositePowerLawViscosityProperty => CompositePowerLawViscosityProperty(
            [load_property(x, :PowerLawViscosityProperty) for x ∈ d[:piter]], d[:dϵ₀],
        )
        :DilatancyProperty => DilatancyProperty(
            d[:tₚ], d[:ϵ], d[:β], d[:p₀],
        )
    end
end