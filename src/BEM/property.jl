import Base.fieldnames
import Base.==

abstract type StateEvolutionLaw end
struct DieterichStateLaw <: StateEvolutionLaw end

abstract type AbstractProperty end

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


const prop_field_names = Dict(
    :RateStateQuasiDynamicProperty => ("a", "b", "L", "σ", "η", "vpl", "f0", "v0"),
)

for (nn, fn) in prop_field_names
    @eval begin
        fieldnames(p::$(nn)) = $(fn)
        description(p::$(nn)) = String($(QuoteNode(nn)))
    end
end

function Base.:(==)(p1::P, p2::P) where P<:AbstractProperty
    reduce(&, [getfield(p1, Symbol(name)) == getfield(p2, Symbol(name)) for name in fieldnames(p1)])
end