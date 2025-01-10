"""
$(TYPEDEF)

object that shares information about the current quadrature point
(item number w.r.t. current AssemblyType, cell number when reasonable,
region number w.r.t. current AssemblyType, volume if the item,
normal when reasonable, current time, current world coordinates,
current reference coordinates, set of parameters)
"""
mutable struct QPInfos{Ti, Tv, Ttime, Tx, Txref, TvG, TiG, PT}
    item::Ti
    cell::Ti
    region::Ti
    volume::TvG
    normal::Vector{TvG}
    time::Ttime
    x::Vector{Tx}
    xref::Vector{Txref}
    grid::ExtendableGrid{TvG, TiG}
    params::PT
end


"""
$(TYPEDSIGNATURES)

constructor for QPInfos
"""
function QPInfos(xgrid::ExtendableGrid{Tv, Ti}; time = 1.0, dim = size(xgrid[Coordinates], 1), T = Tv, x = ones(T, dim), params = [], kwargs...) where {Tv, Ti}
    return QPInfos{Ti, Tv, typeof(time), T, T, Tv, Ti, typeof(params)}(Ti(1), Ti(1), Ti(1), Tv(1.0), zeros(Tv, dim), time, x, ones(T, dim), xgrid, params)
end


"""
$(TYPEDSIGNATURES)

standard kernel that just copies the input to the result
"""
function standard_kernel(result, input, qpinfo)
    result .= input
    return nothing
end

"""
$(TYPEDSIGNATURES)

a kernel that acts as the constant function one
"""
function constant_one_kernel(result, qpinfo)
    result .= 1
    return nothing
end
