# HDF5

mutable struct H5SaveBuffer{
    S<:AbstractString, D<:AbstractDict, I<:Integer, R<:Real,
    V<:AbstractVector, V1<:AbstractVector, V2<:Tuple, V3<:Tuple, UT<:AbstractUnitRange}
    file::S
    ubuffer::D
    tbuffer::V
    nstep::I
    count::I
    total::I
    uiter::UT
    ustrs::V1
    ushapes::V2
    idxs::V3
    tstop::R
    tstr::S
    stride::I
    stride_count::I
end

function create_h5buffer(file::AbstractString, ptrs::Tuple, ueltype::DataType, nstep::Integer, tstop::Real, ustrs, tstr; stride::Integer=1, append=false)
    @assert tstr ∉ ustrs "Duplicate name of $(tstr) in $(ustrs)."
    @assert length(ustrs) == length(ptrs) "Unmatched length between solution components and names."
    ubuffer = h5savebufferzone(ptrs, nstep, ustrs)
    tbuffer = Vector{ueltype}(undef, nstep)
    count, total = 1, 0
    uiter = eachindex(ptrs)
    ushapes = map(size, ptrs)
    f = x -> map(Base.Slice, axes(x))
    idxs = map(f, ptrs)
    if !append
        h5open(file, "w") do f
            create_dataset(f, tstr, datatype(typeof(tstop)), ((nstep,), (-1,)), chunk=(nstep,))
            for i ∈ uiter
                accusize = (ushapes[i]..., nstep)
                create_dataset(f, ustrs[i], datatype(ueltype), (accusize, (ushapes[i]..., -1,)), chunk=accusize)
            end
        end
    else
        total = h5open(file, "r") do f
            open_dataset(f, tstr) |> length
        end
    end
    return H5SaveBuffer(file, ubuffer, tbuffer, nstep, count, total, uiter, ustrs, ushapes, idxs, tstop, tstr, stride, 0)
end

h5savebufferzone(u::AbstractArray, nstep::Integer) = Array{eltype(u)}(undef, size(u)..., nstep)
h5savebufferzone(u::Tuple, nstep, names) = Dict(names[i] => h5savebufferzone(u[i], nstep) for i ∈ eachindex(u))

function h5savebuffercbkernel(u, t, integrator, b::H5SaveBuffer, getu::Function)
    if mod(b.stride_count, b.stride) == 0
        ptrs = getu(u, t, integrator)
        _trigger_copy(b, ptrs, t)
        (t == b.tstop || b.count > b.nstep) && _trigger_save(b)
    end
    b.stride_count += 1
end

function _trigger_copy(b::H5SaveBuffer, ptrs, t)
    b.tbuffer[b.count] = t
    for i ∈ b.uiter
        @strided b.ubuffer[b.ustrs[i]][b.idxs[i]..., b.count] .= ptrs[i]
    end
    b.count += 1
end

function _trigger_save(b::H5SaveBuffer)
    h5open(b.file, "r+") do f
        ht = open_dataset(f, b.tstr)
        HDF5.set_extent_dims(ht, (b.total + b.count - 1,))
        ht[b.total+1: b.total+b.count-1] = ifelse(b.count > b.nstep, b.tbuffer, selectdim(b.tbuffer, 1, 1: b.count-1))
        for i ∈ b.uiter
            hd = open_dataset(f, b.ustrs[i])
            HDF5.set_extent_dims(hd, (b.ushapes[i]..., b.total + b.count - 1))
            hd[b.idxs[i]..., b.total+1: b.total+b.count-1] = ifelse(b.count > b.nstep, b.ubuffer[b.ustrs[i]],
                view(b.ubuffer[b.ustrs[i]], b.idxs[i]..., 1: b.count-1))
        end
    end
    b.total += b.count - 1
    b.count = 1
end

# examples of function handlers to extract solutions for saving
𝐕𝚯𝚫(u::ArrayPartition, args...) = (u.x[1], u.x[2], u.x[3])
𝐕𝚯𝚬′𝚫(u::ArrayPartition, t, integrator) = (u.x[1], u.x[2], integrator(integrator.t, Val{1}).x[3], u.x[5])

"""
    wsolve(prob::ODEProblem, alg::OrdinaryDiffEqAlgorithm,
        file, nstep, getu, ustrs, tstr; kwargs...)

Write the solution to HDF5 file while solving the ODE. The interface
    is exactly the same as
    [`solve` an `ODEProblem`](https://diffeq.sciml.ai/dev/basics/common_solver_opts/)
    except a few more about the saving procedure. Notice, it will set
    `save_everystep=false` so to avoid memory blow up. The return code
    will be written as an attribute in `tstr` data group.

## Extra Arguments
- `file::AbstractString`: name of file to be saved
- `nstep::Integer`: number of steps after which a saving operation will be performed
- `getu::Function`: function handler to extract desired solution for saving, which should have the signature
    `getu(u, t, integrator)`, where `u` is the current solution, `t` is the current time,
    and `integrator` is the current integrator object. The output should be a tuple of
    arrays or vectors to be saved.
- `ustr::AbstractVector`: list of names to be assigned for each components, whose
    length must equal the length of `getu` output
- `tstr::AbstractString`: name of time data

## KWARGS
- `stride::Integer=1`: downsampling rate for saving outputs
- `append::Bool=false`: if true then append solution after the end of `file`
- `force::Bool=false`: force to overwrite the existing solution file

## Returns
- `sol::ODESolution`: the solution object of `OrdinaryDiffEq.jl`
"""
function wsolve(prob::ODEProblem, alg::OrdinaryDiffEq.OrdinaryDiffEqAlgorithm, file, nstep, getu, ustrs, tstr; stride::Integer=1, append::Bool=false, force::Bool=false, kwargs...)
    if isfile(file) && !force && !append
        @info "Overwrite existing file $(file) must set `force = true`."
        @info "Aborting computation."
        return
    end

    integrator = init(prob, alg)
    ptrs = getu(prob.u0, prob.tspan[1], integrator)
    bf = create_h5buffer(file, ptrs, eltype(prob.u0), nstep, prob.tspan[2], ustrs, tstr; stride=stride, append=append)
    cb = (u, t, integrator) -> h5savebuffercbkernel(u, t, integrator, bf, getu)
    fcb = FunctionCallingCallback(cb)
    sol = solve(prob, alg; save_everystep=false, callback=fcb, kwargs...)
    _trigger_save(bf) # in case `solve` terminates earlier
    return sol
end