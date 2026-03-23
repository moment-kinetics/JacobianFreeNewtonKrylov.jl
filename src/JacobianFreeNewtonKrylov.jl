"""
A package implementing a Jacobian-free Newton-Krylov method [1-3] for solving nonlinear systems in serial.

This class of solvers use an outer Newton iteration to solve the system
   R(x) = 0,
where R is the residual function and x is the solution vector.
Each step of the Newton iteration requires a solution to the linearised system
   J.δx = -R(x^n)
for the nth iteration of the solution vector x^n, where J = δR/δx is the Jacobian.
Here, we use the (weighted) GMRES method [4-7] to solve the linear system,
which is a method that uses the Krylov subspace to obtain a solution
without an explicit expression for J. Instead, GMRES computes the δx which minimises
    r = || J.δx + R(x^n) ||
For this method, only the product J.v for a vector v is required.
We compute this product using the finite difference
    J.v = (R(x + e.v) - R(x))/e + O(e.v)
for e a suitably sized number for the estimate
    O(e.v) << J.v
to hold.

Useful references:
[1] D.A. Knoll, D.E. Keyes, "Jacobian-free Newton–Krylov methods: a survey of approaches and applications", Journal of Computational Physics, Volume 193, 2004, Pages 357-397, https://doi.org/10.1016/j.jcp.2003.08.010.
[2] V.A. Mousseau and D.A. Knoll, "Fully Implicit Kinetic Solution of Collisional Plasmas", Journal of Computational Physics 136, 308–323 (1997), https://doi.org/10.1006/jcph.1997.5736.
[3] V.A. Mousseau, "Fully Implicit Kinetic Modelling of Collisional Plasmas", PhD thesis, Idaho National Engineering Laboratory (1996), https://inis.iaea.org/collection/NCLCollectionStore/_Public/27/067/27067141.pdf.
[4] https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
[5] https://www.rikvoorhaar.com/blog/gmres
[6] E. Carson , J. Liesen, Z. Strakoš, "Towards understanding CG and GMRES through examples", Linear Algebra and its Applications 692, 241–291 (2024), https://doi.org/10.1016/j.laa.2024.04.003.
[7] Q. Zou, "GMRES algorithms over 35 years", Applied Mathematics and Computation 445, 127869 (2023), https://doi.org/10.1016/j.amc.2023.127869
"""
module JacobianFreeNewtonKrylov

export newton_solve!,
       NewtonKrylovSolverData

struct NewtonKrylovDiagnostics
    n_solves::Base.RefValue{Int64}
    nonlinear_iterations::Base.RefValue{Int64}
    linear_iterations::Base.RefValue{Int64}
end

struct NewtonKrylovSolverData{TFloat <: AbstractFloat}
    rtol::TFloat
    atol::TFloat
    nonlinear_max_iterations::Int64
    preconditioner_update_interval::Int64
    linear_rtol::TFloat
    linear_atol::TFloat
    krylov_subspace_max_size::Int64
    H::Array{TFloat,2}
    c::Array{TFloat,1}
    s::Array{TFloat,1}
    g::Array{TFloat,1}
    V::Array{TFloat,2}
    residual::Vector{TFloat}
    delta_x::Vector{TFloat}
    rhs_delta::Vector{TFloat}
    v::Vector{TFloat}
    w::Vector{TFloat}
    weight::Vector{TFloat}
    diagnostics::NewtonKrylovDiagnostics
    """
    """
    function NewtonKrylovSolverData(::Type{TFloat}, n_degrees_of_freedom::Int64;
                                    # relative tolerance for convergence of Newton iterations
                                    rtol::TFloatTol=1.0e-8,
                                    # absolute tolerance for convergence of Newton iterations
                                    atol::TFloatTol=1.0e-12,
                                    # max newton_solve! iterations
                                    nonlinear_max_iterations::Int64=20,
                                    # tolerance for GMRES linear solve, relative to the residual_norm
                                    # which is weighted with 1/(atol + rtol |x|), with x the solution vector.
                                    # GMRES_atol = 1 implies that the GMRES linear solve converges to
                                    # the same precision as the final Newton iteration.
                                    linear_rtol::TFloatTol=1.0e-3,
                                    linear_atol::TFloatTol=1.0,
                                    # (maximum) number of members of Krylov subspace in GMRES solve
                                    krylov_subspace_max_size::Int64=10,
                                    preconditioner_update_interval::Int64=300) where {
                                        TFloat <: AbstractFloat, TFloatTol <: AbstractFloat}
        # buffer arrays for Newton-Krylov-GMRES solve
        H = Array{TFloat,2}(undef, krylov_subspace_max_size + 1, krylov_subspace_max_size)
        c = Array{TFloat,1}(undef, krylov_subspace_max_size + 1)
        s = Array{TFloat,1}(undef, krylov_subspace_max_size + 1)
        g = Array{TFloat,1}(undef, krylov_subspace_max_size + 1)
        V = Array{TFloat,2}(undef, n_degrees_of_freedom, krylov_subspace_max_size + 1)
        residual = Vector{TFloat}(undef, n_degrees_of_freedom)
        delta_x = Vector{TFloat}(undef, n_degrees_of_freedom)
        rhs_delta = Vector{TFloat}(undef, n_degrees_of_freedom)
        v = Vector{TFloat}(undef, n_degrees_of_freedom)
        w = Vector{TFloat}(undef, n_degrees_of_freedom)
        weight = Vector{TFloat}(undef, n_degrees_of_freedom)
        return new{TFloat}(rtol, atol,
                nonlinear_max_iterations,
                preconditioner_update_interval,
                linear_rtol,
                linear_atol, krylov_subspace_max_size,
                H, c, s, g, V,
                residual, delta_x, rhs_delta, v, w, weight,
                NewtonKrylovDiagnostics(Ref(0), Ref(0), Ref(0)))
    end
end

"""
    newton_solve!(x, rhs_func!, nl_solver_params;
                  left_preconditioner=(x) -> nothing, right_preconditioner=(x) -> nothing)

`x` is the initial guess at the solution, and is overwritten by the result of the Newton
solve.

`rhs_func!(residual, x)` is the function we are trying to find a solution of. It calculates
```
residual = R(x)
```
where we are trying to solve `R(x)=0`.

`left_preconditioner` or `right_preconditioner` apply preconditioning. They should be
passed a function that solves `P.x = b` where `P` is the preconditioner matrix, `b`
is given by the values passed to the function as the argument, and the result `x` is
returned by overwriting the argument.

Tolerances
----------

Note that the meaning of the relative tolerance `rtol` and absolute tolerance `atol` is
very different for the outer Newton iteration and the inner GMRES iteration.

For the outer Newton iteration the residual `R(x^n)` measures the departure of the
system from the solution with a weight `rtol` that weights the residual with a relative
error compared to the solution at each grid point, and an absolute tolerance `atol`.

In contrast, GMRES is constructed to minimise the norm || . || of  `r_k = b - A . x_k`
where GMRES is solving the linear system `A . x = b`, `x_k` is the approximation
to the solution `x` at the `k`'th iteration and `r_k` is the residual at the
`k`'th iteration. There is no flexibility to measure error relative to `x` in any
sense. For GMRES, a `relative tolerance' is relative to the residual of the
right-hand-side `b`, which is the first iterate `x_0` (when no initial guess is
given). [Where a non-zero initial guess is given it might be better to use a different
stopping criterion, see Carson et al. section 3.8.]. The stopping criterion for the GMRES
iteration is therefore
```
|| r_k || < max(linear_rtol || r_0 ||, linear_atol) = max(linear_rtol} || b ||, linear_atol)
```
"""
function newton_solve!(solution_vector_x::TVector, residual_func!::TResidual,
            nl_solver_params::NewtonKrylovSolverData{TFloat};
            left_preconditioner::TPreconditionerLeft=(solution_vector_x) -> nothing,
            right_preconditioner::TPreconditionerRight=(solution_vector_x) -> nothing,
            recalculate_preconditioner::TPreconditionerUpdate=() -> nothing,
            diagnose::Bool=false) where {
                TFloat <: AbstractFloat,
                TVector <: AbstractArray{TFloat,1},
                TResidual <: Function,
                TPreconditionerLeft <: Function,
                TPreconditionerRight <: Function,
                TPreconditionerUpdate <: Function}
    rtol = nl_solver_params.rtol
    atol = nl_solver_params.atol
    residual = nl_solver_params.residual
    delta_x = nl_solver_params.delta_x
    v = nl_solver_params.v
    w = nl_solver_params.w
    weight = nl_solver_params.weight

    # N.B. the weights are proportional to 1/(atol + rtol * |x|)^2
    calculate_weight!(weight, nl_solver_params.atol, nl_solver_params.rtol, solution_vector_x)
    residual_func!(residual, solution_vector_x)
    # N.B. because weights ~ 1/(atol + rtol * |x|)^2 the size of the residual norm
    # is residual_norm ~ 1/(atol + rtol * |x|) >> 1 for the initial guess and
    # residual_norm ~ 1 for the converged solution vector x
    residual_norm = vector_norm(residual, weight)
    newton_iterations = 0
    GMRES_iterations = 0
    success = true
    while residual_norm > 1.0
        newton_iterations += 1

        # use the GMRES algoritm to find the approximate solution to:
        #   J δx = -RHS(x)
        krylov_subspace_size = linear_solve!(solution_vector_x, residual_func!, residual, delta_x, v, w, weight;
                                   GMRES_rtol=nl_solver_params.linear_rtol,
                                   GMRES_atol=nl_solver_params.linear_atol,
                                   max_krylov_subspace_size=nl_solver_params.krylov_subspace_max_size,
                                   left_preconditioner=left_preconditioner,
                                   right_preconditioner=right_preconditioner,
                                   H=nl_solver_params.H, c=nl_solver_params.c,
                                   s=nl_solver_params.s, g=nl_solver_params.g,
                                   V=nl_solver_params.V, rhs_delta=nl_solver_params.rhs_delta)
        GMRES_iterations += krylov_subspace_size

        # calculate the residual for the NaN diagnostic check
        @. w = solution_vector_x + delta_x
        residual_func!(residual, w)
        residual_norm = vector_norm(residual, weight)
        if isnan(residual_norm)
            error("NaN in Newton iteration at iteration $(newton_iterations)")
        end

        # update root estimate x_n+1 = x_n + delta_x
        @. solution_vector_x = w

        # update the weight for the inner product
        calculate_weight!(weight, nl_solver_params.atol, nl_solver_params.rtol, solution_vector_x)

        if newton_iterations % nl_solver_params.preconditioner_update_interval == 0
            # Update the preconditioner to accelerate convergence
            recalculate_preconditioner()
        end

        if newton_iterations > nl_solver_params.nonlinear_max_iterations
            println("maximum iteration limit reached")
            success = false
            break
        end
    end
    nl_solver_params.diagnostics.n_solves[] += 1
    nl_solver_params.diagnostics.nonlinear_iterations[] += newton_iterations
    nl_solver_params.diagnostics.linear_iterations[] += GMRES_iterations
    if diagnose
        println("Newton iterations: ", newton_iterations)
        println("Final residual_norm: ", residual_norm)
        println("Total linear (GMRES) iterations: ", GMRES_iterations)
        println("Linear (GMRES) iterations per Newton iteration: ", GMRES_iterations / newton_iterations)
        println()
    end

    return success
end

function calculate_weight!(weight::Vector{TFloat},
            atol::TFloat, rtol::TFloat, solution_vector_x::Vector{TFloat})  where TFloat <: AbstractFloat
    @inbounds for i in eachindex(solution_vector_x,weight)
        weight[i] = 1.0 / abs2(rtol * abs(solution_vector_x[i]) + atol)
    end
    # normalisation = length(weight)/sum(weight)
    # @. weight *= normalisation
    return nothing
end
function vector_norm(residual::Array{TFloat, 1},
            weight::Vector{TFloat}) where TFloat <: AbstractFloat
    return sqrt(vector_dot_product(residual, residual, weight))
end

function vector_dot_product(v::Array{TFloat, 1}, w::Array{TFloat, 1},
            weight::Vector{TFloat}) where TFloat <: AbstractFloat
    dot_product = 0.0
    @inbounds for i in eachindex(v,w)
        dot_product += v[i] * w[i] * weight[i]
    end
    dot_product = dot_product / length(v)
    return dot_product
end

function calculate_delta_x(delta_x::Array{TFloat, 1},
            V::Array{TFloat,2}, y::Vector{TFloat}) where TFloat <: AbstractFloat
    @. delta_x = 0.0
    @inbounds begin
        for iy in eachindex(y)
            for icoord in eachindex(delta_x)
                delta_x[icoord] += y[iy] * V[icoord,iy]
            end
        end
    end
    return nothing
end

"""
Apply the GMRES algorithm to solve the 'linear problem' J.δx^n = R(x^n), which is needed
at each step of the outer Newton iteration (in `newton_solve!()`).

Uses Givens rotations to reduce the upper Hessenberg matrix to an upper triangular form,
which allows conveniently finding the residual at each step, and computing the final
solution, without calculating a least-squares minimisation at each step. See 'algorithm 2
MGS-GMRES' in Zou (2023) [https://doi.org/10.1016/j.amc.2023.127869].
"""
function linear_solve!(solution_vector_x::TVector, residual_func!::TResidual,
            residual0::TVector, delta_x::TVector, v::TVector, w::TVector,
            weight::TVector; GMRES_rtol::TFloat, GMRES_atol::TFloat, max_krylov_subspace_size::Int64,
            left_preconditioner::TPreconditionerLeft,
            right_preconditioner::TPreconditionerRight,
            H::Array{TFloat,2}, c::TVector, s::TVector,
            g::TVector, V::Array{TFloat,2}, rhs_delta::TVector) where {
                        TFloat <: AbstractFloat,
                        TVector <: Vector{TFloat},
                        TResidual <: Function,
                        TPreconditionerLeft <: Function,
                        TPreconditionerRight <: Function}
    # use the GMRES algoritm to find the approximate solution to:
    #   J δx = residual0

    # We calculate the product J.v by finite difference using the relation
    #  J.v = ( R(x + e.v) - R(x))/ e
    # for e a suitably chosen number such that
    #  (R(x + e.v ) - R(x))/e = J.v + O(e.v)
    #
    # In the standard GMRES method where the vector v has entries of order unity,
    # v being computed from a vector u by `v = u / vector_norm(u, weight)`.
    # The number e should be << 1, but not so small as to cause rounding errors when
    # evaluating R(x + e.v) - R(x). A suitable choice for e in these circumstances is
    # e = `sqrt(eps())`, where `eps()` is machine precision for the floating point type.
    #
    # However, here we use the weighted GMRES method, and v is normalised with a large weight
    # in the definition of `vector_norm()` such that the entries of
    # `v = u / vector_norm(u, weight)` are small.
    # To avoid possible rounding errors from a very small e.v, we need to choose
    # e = `sqrt(eps())*vector_norm(ones(TFloat,length(x)), weight)`
    # so that the large weight in the normalisation of v is cancelled out.
    #
    # We define e = `Jv_scale_factor` below
    Jv_scale_factor = sqrt(eps(TFloat))*vector_norm(ones(TFloat,length(solution_vector_x)), weight)
    inv_Jv_scale_factor = 1.0 / Jv_scale_factor
    # the function computing J.v = ( R(x + e.v) - R(x))/ e
    function approximate_Jacobian_vector_product!(v::Vector{TFloat}) where TFloat <: AbstractFloat
        right_preconditioner(v)
        @. v = solution_vector_x + Jv_scale_factor * v
        residual_func!(rhs_delta, v)
        @. v = (rhs_delta - residual0) * inv_Jv_scale_factor
        left_preconditioner(v)
        return nothing
    end

    # To start with we use 'v' as a buffer to make a copy of residual0 to which we can apply
    # the left-preconditioner.
    @. v = residual0
    left_preconditioner(v)

    # Now we actually set 'w' as the first Krylov vector, and normalise it.
    @. w = -v
    beta = vector_norm(w, weight)
    @inbounds for i in eachindex(w)
        V[i,1] = w[i]/beta
    end
    g[1] = beta

    # Set tolerance for GMRES iteration to rtol times the initial residual, unless this is
    # so small that it is smaller than atol, in which case use atol instead.
    tol = max(GMRES_rtol * beta, GMRES_atol)

    krylov_subspace_size = 0
    # set H to zero to ensure lower-than-diagonal entries
    # of the upper Hessenberg matrix are zero
    @. H = 0.0
    @inbounds begin
        for i ∈ 1:max_krylov_subspace_size
            krylov_subspace_size = i

            # Compute next Krylov vector
            for k in eachindex(w)
                w[k] = V[k,i]
            end

            approximate_Jacobian_vector_product!(w)

            # Gram-Schmidt orthogonalization
            for j ∈ 1:i
                for k in eachindex(v)
                    v[k] = V[k,j]
                end
                w_dot_Vj = vector_dot_product(w, v, weight)

                H[j,i] = w_dot_Vj

                for k in eachindex(w)
                    w[k] = w[k] - H[j,i] * V[k,j]
                end
            end
            norm_w = vector_norm(w, weight)

            H[i+1,i] = norm_w

            for k in eachindex(w)
                V[k,i+1] = w[k]/H[i+1,i]
            end

            # apply Givens rotation to find new values of H and g
            for j ∈ 1:i-1
                gamma = c[j] * H[j,i] + s[j] * H[j+1,i]
                H[j+1,i] = -s[j] * H[j,i] + c[j] * H[j+1,i]
                H[j,i] = gamma
            end
            delta = sqrt(H[i,i]^2 + H[i+1,i]^2)
            s[i] = H[i+1,i] / delta
            c[i] = H[i,i] / delta
            H[i,i] = c[i] * H[i,i] + s[i] * H[i+1,i]
            H[i+1,i] = 0
            g[i+1] = -s[i] * g[i]
            g[i] = c[i] * g[i]

            residual = abs(g[i+1])

            if residual < tol
                break
            end
        end
    end
    i = krylov_subspace_size

    # finally, compute delta_x
    #################################

    @views y = H[1:i,1:i] \ g[1:i]

    # The following calculates
    #    delta_x .= sum(y[i] .* V[:,i] for i ∈ 1:length(y))
    calculate_delta_x(delta_x, V, y)
    right_preconditioner(delta_x)

    return krylov_subspace_size
end

end
