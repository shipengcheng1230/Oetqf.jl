function get_matvecmul!()
    return @load_preference("matvecmul!", "LinearAlgebra")
end

const _matvecmul! = get_matvecmul!()

function set_matvecmul!(backend::String)
    if backend ∉ ("LinearAlgebra", "Octavian")
        throw(ArgumentError("Invalid backend: $(backend)"))
    end
    @set_preferences!("matvecmul!" => backend)
    @info("New backend $(backend) set; restart your Julia session for this change to take effect!")
end

@static if _matvecmul! == "LinearAlgebra"
    const matvecmul! = mul! # OpenBLAS / MKL
elseif _matvecmul! == "Octavian"
    const matvecmul! = matmul! # Julia implementation of BLAS
else
    nothing
end
