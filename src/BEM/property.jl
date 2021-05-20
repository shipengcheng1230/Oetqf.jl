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
    f0::T = 0.6 # ref. frictional coeff
    v0::T = 1e-6 # ref. velocity

    @assert size(a) == size(b)
    @assert size(b) == size(L)
    @assert size(L) == size(σ)
    @assert f0 > 0
    @assert v0 > 0
    @assert η > 0
    @assert vpl > 0
end

@with_kw struct PowerLawViscosityProperty{T, I, U} <: ViscosityProperty
    γ::T
    n::I
    dϵ₀::U
end

const _field_names = Dict(
    :RateStateQuasiDynamicProperty => (:a, :b, :L, :σ, :η, :vpl, :f0, :v0),
    :PowerLawViscosityProperty => (:γ, :n, :dϵ₀),
)

@assert mapreduce(Set, union, values(_field_names)) |> length == mapreduce(length, +, values(_field_names)) "Found duplicated property field names!"

for (nn, fn) in _field_names
    @eval begin
        fieldnames(p::$(nn)) = $(fn)
        description(p::$(nn)) = String($(QuoteNode(nn)))
    end
end

function Base.:(==)(p1::P, p2::P) where P<:AbstractProperty
    reduce(&, [getfield(p1, name) == getfield(p2, name) for name in fieldnames(p1)])
end

function struct_to_dict(p)
    Dict(name => getfield(p, name) for name ∈ fieldnames(p))
end

function save_property(file::AbstractString, p::AbstractProperty)
    bson(file, struct_to_dict(p))
end

function save_property(file::AbstractString, piter)
    d = foldl(merge, map(struct_to_dict, piter))
    bson(file, d)
end

function load_property(file::AbstractString, p::Symbol)
    d = BSON.load(file)
    @match p begin
        :RateStateQuasiDynamicProperty => RateStateQuasiDynamicProperty(
            d[:a], d[:b], d[:L], d[:σ], d[:η], d[:vpl], d[:f0], d[:v0],
        )
        :PowerLawViscosityProperty => PowerLawViscosityProperty(
            d[:γ], d[:n], d[:dϵ₀],
        )
    end
end