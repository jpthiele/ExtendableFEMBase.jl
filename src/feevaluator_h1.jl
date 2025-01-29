# IDENTITY H1
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:Identity, <:AbstractH1FiniteElement})
    # needs only to be updated if basis sets can change (e.g. for P3 in 2D/3D)
    if FEBE.subset_handler != NothingFunction
        subset = _update_subset!(FEBE)
        cvals = FEBE.cvals
        fill!(cvals, 0)
        refbasisvals = FEBE.refbasisvals
        for i in 1:size(cvals, 3), dof_i in 1:size(cvals, 2), k in 1:size(cvals, 1)
            cvals[k, dof_i, i] = refbasisvals[i][subset[dof_i], k]
        end
    end
    return nothing
end

# IDENTITYCOMPONENT H1
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:IdentityComponent{c}, <:AbstractH1FiniteElement}) where {c}
    if FEBE.subset_handler != NothingFunction
        subset = _update_subset!(FEBE)
        cvals = FEBE.cvals
        refbasisvals = FEBE.refbasisvals
        fill!(cvals, 0)
        for i in 1:size(cvals, 3)
            for dof_i in 1:size(cvals, 2)
                cvals[1, dof_i, i] = refbasisvals[i][subset[dof_i], c]
            end
        end
    end
    return nothing
end

# IDENTITY H1+COEFFICIENTS
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:Identity, <:AbstractH1FiniteElementWithCoefficients})
    subset = _update_subset!(FEBE)
    coefficients = _update_coefficients!(FEBE)
    cvals = FEBE.cvals
    refbasisvals = FEBE.refbasisvals
    fill!(cvals, 0)
    for i in 1:size(cvals, 3), dof_i in 1:size(cvals, 2), k in 1:size(cvals, 1)
        cvals[k, dof_i, i] = refbasisvals[i][subset[dof_i], k] * coefficients[k, dof_i]
    end
    return nothing
end

# IDENTITYCOMPONENT H1+COEFFICIENTS
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:IdentityComponent{c}, <:AbstractH1FiniteElementWithCoefficients}) where {c}
    subset = _update_subset!(FEBE)
    coefficients = _update_coefficients!(FEBE)
    cvals = FEBE.cvals
    refbasisvals = FEBE.refbasisvals
    fill!(cvals, 0)
    for i in 1:size(cvals, 3)
        for dof_i in 1:size(cvals, 2)
            cvals[1, dof_i, i] = refbasisvals[i][subset[dof_i], c] * coefficients[c, dof_i]
        end
    end
    return nothing
end

function apply_inverse_transform!(
        L2GAinv::Matrix{T1},
        cvals::Array{T2, 3},
        refvals::Array{T3, 3},
        dof_i::Int,
        compressiontargets,
        reference_offset::Int64,
        coeff::T4 = 1.0
    ) where {T1, T2, T3, T4}
    m = size(L2GAinv, 2)
    n = size(cvals, 3)
    # Reshape both local basis values (cvals)
    # and reference basis values as matrices to apply L2GAinv
    Y = reshape(view(cvals, compressiontargets, dof_i, :), m, n)
    X = reshape(view(refvals, reference_offset, :, :), m, n)
    # Y = coeff*L2GAinv*X + 1.0*Y
    mul!(Y, L2GAinv, X, coeff, 1.0)
    return nothing
end
# GRADIENT H1
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:Gradient, <:AbstractH1FiniteElement})
    L2GAinv = _update_trafo!(FEBE)
    subset = _update_subset!(FEBE)
    fill!(FEBE.cvals, 0)
    for c in 1:length(FEBE.offsets), dof_i in 1:size(FEBE.cvals, 2)
        apply_inverse_transform!(
            L2GAinv,
            FEBE.cvals,
            FEBE.refbasisderivvals,
            dof_i,
            (1:size(L2GAinv, 2)) .+ FEBE.offsets[c],
            subset[dof_i] + FEBE.offsets2[c]
        )
    end
    return nothing
end

# GRADIENT H1+COEFFICIENTS
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:Gradient, <:AbstractH1FiniteElementWithCoefficients})
    L2GAinv = _update_trafo!(FEBE)
    subset = _update_subset!(FEBE)
    coefficients = _update_coefficients!(FEBE)
    fill!(FEBE.cvals, 0)
    for c in 1:length(FEBE.offsets), dof_i in 1:size(FEBE.cvals, 2)
        apply_inverse_transform!(
            L2GAinv,
            FEBE.cvals,
            FEBE.refbasisderivvals,
            dof_i,
            (1:size(L2GAinv, 2)) .+ FEBE.offsets[c],
            subset[dof_i] + FEBE.offsets2[c],
            coefficients[c, dof_i]
        )
    end
    return nothing
end

# SYMMETRIC GRADIENT H1
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:SymmetricGradient{offdiagval}, <:AbstractH1FiniteElement}) where {offdiagval}
    L2GAinv = _update_trafo!(FEBE)
    subset = _update_subset!(FEBE)
    compression = FEBE.compressiontargets
    fill!(FEBE.cvals, 0)
    for dof_i in 1:size(FEBE.cvals, 2), c in 1:length(FEBE.offsets)
        compressiontargets = view(compression, (1:size(L2GAinv, 2)) .+ FEBE.offsets[c])
        apply_inverse_transform!(
            L2GAinv,
            FEBE.cvals,
            FEBE.refbasisderivvals,
            dof_i,
            compressiontargets,
            subset[dof_i] + FEBE.offsets2[c]
        )
    end
    # Multiply off diagonal values by offdiagval
    for c in 1:length(FEBE.offsets)
        # Due to compression only the upper right diagonal has to be multiplied
        for k in (c + 1):size(L2GAinv, 2)
            view(FEBE.cvals, compression[k + FEBE.offsets[c]], :, :) .*= offdiagval
        end
    end

    return nothing
end

# DIVERGENCE H1
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:Divergence, <:AbstractH1FiniteElement})
    L2GAinv = _update_trafo!(FEBE)
    subset = _update_subset!(FEBE)
    cvals = FEBE.cvals
    offsets2 = FEBE.offsets2
    refbasisderivvals = FEBE.refbasisderivvals
    fill!(cvals, 0)
    for dof_i in 1:size(cvals, 2)
        for i in 1:size(cvals, 3), k in 1:size(L2GAinv, 2), j in 1:size(L2GAinv, 1)
            cvals[1, dof_i, i] += L2GAinv[k, j] * refbasisderivvals[subset[dof_i] + offsets2[k], j, i]
        end
    end
    return nothing
end

# DIVERGENCE H1+COEFFICIENTS
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:Divergence, <:AbstractH1FiniteElementWithCoefficients})
    L2GAinv = _update_trafo!(FEBE)
    subset = _update_subset!(FEBE)
    coefficients = _update_coefficients!(FEBE)
    cvals = FEBE.cvals
    offsets2 = FEBE.offsets2
    refbasisderivvals = FEBE.refbasisderivvals
    fill!(cvals, 0)
    for i in 1:size(cvals, 3), dof_i in 1:size(cvals, 2), k in 1:size(L2GAinv, 2), j in 1:size(L2GAinv, 1)
        cvals[1, dof_i, i] += L2GAinv[k, j] * refbasisderivvals[subset[dof_i] + offsets2[k], j, i] * coefficients[k, dof_i]
    end
    return nothing
end

# NORMALFLUX H1 (ON_FACES, ON_BFACES)
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:NormalFlux, <:AbstractH1FiniteElement})
    # fetch normal of item
    normal = view(FEBE.coefficients_op, :, FEBE.citem[])
    subset = _update_subset!(FEBE)
    cvals = FEBE.cvals
    refbasisvals = FEBE.refbasisvals
    fill!(cvals, 0)
    for i in 1:size(cvals, 3), dof_i in 1:size(cvals, 2), k in 1:length(normal)
        cvals[1, dof_i, i] += refbasisvals[i][subset[dof_i], k] * normal[k]
    end
    return nothing
end

# NORMALFLUX H1+COEFFICIENTS (ON_FACES, ON_BFACES)
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:NormalFlux, <:AbstractH1FiniteElementWithCoefficients})
    # fetch normal of item
    normal = view(FEBE.coefficients_op, :, FEBE.citem[])
    subset = _update_subset!(FEBE)
    coefficients = _update_coefficients!(FEBE)
    cvals = FEBE.cvals
    refbasisvals = FEBE.refbasisvals
    fill!(cvals, 0)
    for i in 1:size(cvals, 3), dof_i in 1:size(cvals, 2), k in 1:length(normal)
        cvals[1, dof_i, i] += refbasisvals[i][subset[dof_i], k] * normal[k] * FEBE.coefficients[k, dof_i]
    end
    return nothing
end

# HESSIAN H1
# multiply tinverted jacobian of element trafo with gradient of basis function
# which yields (by chain rule) the gradient in x coordinates
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:Hessian, <:AbstractH1FiniteElement})
    L2GAinv = _update_trafo!(FEBE)
    subset = _update_subset!(FEBE)
    cvals = FEBE.cvals
    offsets = FEBE.offsets
    offsets2 = FEBE.offsets2
    refbasisderivvals = FEBE.refbasisderivvals
    ndofs = size(cvals, 2)
    edim = size(L2GAinv, 1)
    ncomponents = length(offsets)
    fill!(cvals, 0)
    for i in 1:size(cvals, 3)
        for dof_i in 1:ndofs
            for c in 1:ncomponents
                for k in 1:edim, l in 1:edim
                    # second derivatives partial^2 (x_k x_l)
                    for xi in 1:edim, xj in 1:edim
                        cvals[(c - 1) * edim^2 + (k - 1) * edim + l, dof_i, i] += L2GAinv[k, xi] * L2GAinv[l, xj] * refbasisderivvals[subset[dof_i] + offsets2[xi] * ncomponents + offsets2[c], xj, i]
                    end
                end
            end
        end
    end
    return nothing
end

# SYMMETRIC HESSIAN H1
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:SymmetricHessian{offdiagval}, <:AbstractH1FiniteElement}) where {offdiagval}
    L2GAinv = _update_trafo!(FEBE)
    subset = _update_subset!(FEBE)
    compression = FEBE.compressiontargets
    cvals = FEBE.cvals
    offsets = FEBE.offsets
    offsets2 = FEBE.offsets2
    refbasisderivvals = FEBE.refbasisderivvals
    ndofs = size(cvals, 2)
    edim = size(L2GAinv, 1)
    ncomponents = length(offsets)
    fill!(cvals, 0)
    for i in 1:size(cvals, 3)
        for dof_i in 1:ndofs
            for c in 1:ncomponents
                for k in 1:edim, l in k:edim
                    # compute second derivatives  ∂^2 (x_k x_l) and put it in the right spot of Voigt vector
                    # note: if l > k the derivative is multiplied with offdiagval
                    if k != l
                        for xi in 1:edim, xj in 1:edim
                            cvals[compression[(c - 1) * edim^2 + (k - 1) * edim + l], dof_i, i] += offdiagval * L2GAinv[k, xi] * L2GAinv[l, xj] * refbasisderivvals[subset[dof_i] + offsets2[xi] * ncomponents + offsets2[c], xj, i]
                        end
                    else
                        for xi in 1:edim, xj in 1:edim
                            cvals[compression[(c - 1) * edim^2 + (k - 1) * edim + l], dof_i, i] += L2GAinv[k, xi] * L2GAinv[l, xj] * refbasisderivvals[subset[dof_i] + offsets2[xi] * ncomponents + offsets2[c], xj, i]
                        end
                    end
                end
            end
        end
    end
    return nothing
end

# LAPLACIAN HESSIAN H1
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:Laplacian, <:AbstractH1FiniteElement})
    L2GAinv = _update_trafo!(FEBE)
    subset = _update_subset!(FEBE)
    cvals = FEBE.cvals
    offsets = FEBE.offsets
    offsets2 = FEBE.offsets2
    refbasisderivvals = FEBE.refbasisderivvals
    ndofs = size(cvals, 2)
    edim = size(L2GAinv, 1)
    ncomponents = length(offsets)
    fill!(cvals, 0)
    for i in 1:size(cvals, 3)
        for dof_i in 1:ndofs
            for c in 1:ncomponents
                for k in 1:edim
                    # second derivatives partial^2 (x_k x_l)
                    for xi in 1:edim, xj in 1:edim
                        cvals[c, dof_i, i] += L2GAinv[k, xi] * L2GAinv[k, xj] * refbasisderivvals[subset[dof_i] + offsets2[xi] * ncomponents + offsets2[c], xj, i]
                    end
                end
            end
        end
    end
    return nothing
end


# CURLSCALAR H1 : R1 -> R2
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:CurlScalar, <:AbstractH1FiniteElement})
    L2GAinv = _update_trafo!(FEBE)
    subset = _update_subset!(FEBE)
    cvals = FEBE.cvals
    offsets = FEBE.offsets
    offsets2 = FEBE.offsets2
    refbasisderivvals = FEBE.refbasisderivvals
    fill!(cvals, 0)
    for i in 1:size(cvals, 3)
        for dof_i in 1:size(cvals, 2)
            for j in 1:size(L2GAinv, 2)
                cvals[1, dof_i, i] -= L2GAinv[2, j] * refbasisderivvals[subset[dof_i], j, i] # -du/dy
                cvals[2, dof_i, i] += L2GAinv[1, j] * refbasisderivvals[subset[dof_i], j, i] # du/dx
            end
        end
    end
    return nothing
end

# CURL2D H1 : R2 -> R1
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:Curl2D, <:AbstractH1FiniteElement})
    L2GAinv = _update_trafo!(FEBE)
    subset = _update_subset!(FEBE)
    cvals = FEBE.cvals
    offsets = FEBE.offsets
    offsets2 = FEBE.offsets2
    refbasisderivvals = FEBE.refbasisderivvals
    fill!(cvals, 0)
    for i in 1:size(cvals, 3)
        for dof_i in 1:size(cvals, 2)
            for j in 1:size(L2GAinv, 2)
                cvals[1, dof_i, i] -= L2GAinv[2, j] * refbasisderivvals[subset[dof_i], j, i]  # -du1/dy
                cvals[1, dof_i, i] += L2GAinv[1, j] * refbasisderivvals[subset[dof_i] + offsets2[2], j, i]  # du2/dx
            end
        end
    end
    return nothing
end


# CURL3D H1
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:Curl3D, <:AbstractH1FiniteElement})
    L2GAinv = _update_trafo!(FEBE)
    subset = _update_subset!(FEBE)
    cvals = FEBE.cvals
    offsets = FEBE.offsets
    offsets2 = FEBE.offsets2
    refbasisderivvals = FEBE.refbasisderivvals
    fill!(cvals, 0)
    for i in 1:size(cvals, 3)
        for dof_i in 1:size(cvals, 2)
            for k in 1:3
                cvals[1, dof_i, i] += L2GAinv[2, k] * refbasisderivvals[subset[dof_i] + offsets2[3], k, i] # du3/dx2
                cvals[1, dof_i, i] -= L2GAinv[3, k] * refbasisderivvals[subset[dof_i] + offsets2[2], k, i] # - du2/dx3
                cvals[2, dof_i, i] += L2GAinv[3, k] * refbasisderivvals[subset[dof_i] + offsets2[1], k, i] # du1/dx3
                cvals[2, dof_i, i] -= L2GAinv[1, k] * refbasisderivvals[subset[dof_i] + offsets2[3], k, i] # - du3/dx1
                cvals[3, dof_i, i] += L2GAinv[1, k] * refbasisderivvals[subset[dof_i] + offsets2[2], k, i] # du2/dx1
                cvals[3, dof_i, i] -= L2GAinv[2, k] * refbasisderivvals[subset[dof_i] + offsets2[1], k, i] # - du1/dx2
            end
        end
    end
    return nothing
end

# TANGENTGRADIENT H1
function update_basis!(FEBE::SingleFEEvaluator{<:Real, <:Real, <:Integer, <:TangentialGradient, <:AbstractH1FiniteElement})
    L2GAinv = _update_trafo!(FEBE)
    subset = _update_subset!(FEBE)
    cvals = FEBE.cvals
    offsets = FEBE.offsets
    offsets2 = FEBE.offsets2
    refbasisderivvals = FEBE.refbasisderivvals
    fill!(cvals, 0)

    # compute tangent of item
    tangent = FEBE.iteminfo
    tangent[1] = FEBE.coefficients_op[2, FEBE.citem[]]
    tangent[2] = -FEBE.coefficients_op[1, FEBE.citem[]]

    for i in 1:size(cvals, 3)
        for dof_i in 1:size(cvals, 2)
            for c in 1:length(offsets), k in 1:size(L2GAinv, 1)
                for j in 1:size(L2GAinv, 2)
                    # compute duc/dxk
                    cvals[1, dof_i, i] += L2GAinv[k, j] * refbasisderivvals[subset[dof_i] + offsets2[c], j, i] * tangent[c]
                end
            end
        end
    end
    return nothing
end
