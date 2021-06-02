# https://github.com/JuliaLinearAlgebra/Octavian.jl/issues/83
function AmulB!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, α::Number, β::Number)
    @tturbo for m ∈ indices((A, y), 1)
        ym = zero(eltype(y))
        for n ∈ indices((A, x), (2, 1))
            ym += A[m, n] * x[n]
        end
        y[m] = ym * α + y[m] * β
    end
end

function AmulB!(y::AbstractVector{T}, A::AbstractMatrix{T}, x::AbstractVector{T}) where T
    AmulB!(y, A, x, true, false)
end