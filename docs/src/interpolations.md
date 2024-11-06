
# Finite Element Interpolations

## Source functions and QPInfo

The functions that can be interpolated with the methods below are expected to have a certain interface, i.e.:
```julia
function f!(result, qpinfo) end
```
The qpinfo argument communicates vast information of the current quadrature point:

| qpinfo child       | Type               | Description         |
| :----------------  | :----------------  |  :---------------- |
| qpinfo.x           | Vector{Real}       | space coordinates of quadrature point |
| qpinfo.time        | Real               | current time |
| qpinfo.item        | Integer            | current item that contains qpinfo.x |
| qpinfo.region      | Integer            | region number of item |
| qpinfo.xref        | Vector{Real}       | reference coordinates within item of qpinfo.x |
| qpinfo.volume      | Real               | volume of item |
| qpinfo.params      | Vector{Any}        | parameters that can be transfered via keyword arguments |


## Standard Interpolations

Each finite element has its standard interpolator that can be applied to some user-defined source Function. Instead of interpolating on the full cells, the interpolation can be restricted to faces or edges via an AssemblyType.

```@docs
interpolate!
```

It is also possible to interpolate finite element functions on one grid onto a finite element function on another grid via the lazy_interpolate routine.

```@docs
lazy_interpolate!
```

The following function continuously interpolates finite element function into a H1Pk space by
point evaluations at the Lagrange nodes of the H1Pk element (averaged over all neighbours).

```@docs
continuify
```

## Nodal Evaluations

Usually, Plotters need nodal values, so there is a generic function that evaluates any finite element function at the nodes of the grids (possibly by averaging if discontinuous). In case of Identity evaluations of an H1-conforming finite element, the function nodevalues_view can generate a view into the coefficient field that avoids further allocations.


```@docs
nodevalues!
nodevalues
nodevalues_view
nodevalues_subset!
```



## Displace Mesh

Nodal values (e.g. of a FEVector that discretizes a displacement) can be used to displace the mesh.

```@docs
displace_mesh!
displace_mesh
```
