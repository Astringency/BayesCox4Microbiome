# Shared training-only preprocessing for validation and subject-level LOO.
function _training_zero_replacement(X_train::AbstractMatrix)
    positives = Float64[x for x in X_train if x > 0]
    isempty(positives) && throw(ArgumentError("training X contains no positive values."))
    return minimum(positives) / 2
end

function _validation_design_matrix(
    X_train::AbstractMatrix,
    X_val::AbstractMatrix,
    train_data::CoxPreparedData;
    requested_pseudocount,
)
    x = Matrix{Float64}(X_val)
    all(isfinite, x) || throw(ArgumentError("validation X must be finite."))
    all(>=(0), x) || throw(ArgumentError("validation X must be nonnegative."))

    z =
        if train_data.transform == :log
            pc = requested_pseudocount === nothing ? train_data.pseudocount : Float64(requested_pseudocount)
            if pc == 0.0
                if any(iszero, x)
                    replacement = train_data.pseudocount > 0 ? train_data.pseudocount : _training_zero_replacement(X_train)
                    x[x .== 0.0] .= replacement
                end
            else
                x .+= pc
            end
            if train_data.closure
                rowsums = vec(sum(x; dims = 2))
                all(>(0), rowsums) ||
                    throw(ArgumentError("each validation row sum after zero handling must be positive."))
                x ./= rowsums
            end
            log.(x)
        elseif train_data.transform == :none
            x
        else
            throw(ArgumentError("unsupported training transform $(train_data.transform)."))
        end

    if train_data.standardize
        z = (z .- reshape(train_data.col_mean, 1, :)) ./ reshape(train_data.col_scale, 1, :)
    end
    return z
end

