"""
A module implementing a Jacobian-free Newton-Krylov method for solving nonlinear systems in serial.

This class of solvers use an outer Newton iteration. Each step of the Newton iteration requires a
linear solve of the Jacobian. An 'inexact Jacobian' method is used, and the GMRES method
(GMRES is a type of Krylov solver) is used to (approximately) solve the (approximate)
linear system.

Useful references:
[1] V.A. Mousseau and D.A. Knoll, "Fully Implicit Kinetic Solution of Collisional Plasmas", Journal of Computational Physics 136, 308–323 (1997), https://doi.org/10.1006/jcph.1997.5736.
[2] V.A. Mousseau, "Fully Implicit Kinetic Modelling of Collisional Plasmas", PhD thesis, Idaho National Engineering Laboratory (1996), https://inis.iaea.org/collection/NCLCollectionStore/_Public/27/067/27067141.pdf.
[3] https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
[4] https://www.rikvoorhaar.com/blog/gmres
[5] E. Carson , J. Liesen, Z. Strakoš, "Towards understanding CG and GMRES through examples", Linear Algebra and its Applications 692, 241–291 (2024), https://doi.org/10.1016/j.laa.2024.04.003.
[6] Q. Zou, "GMRES algorithms over 35 years", Applied Mathematics and Computation 445, 127869 (2023), https://doi.org/10.1016/j.amc.2023.127869
"""
module JacobianFreeNewtonKrylov

export newton_solve!,
       NewtonKrylovSolverData

struct NewtonKrylovDiagnostics
    n_solves::Base.RefValue{Int64}
    nonlinear_iterations::Base.RefValue{Int64}
    linear_iterations::Base.RefValue{Int64}
    function NewtonKrylovDiagnostics()
        return new(Ref(0), Ref(0), Ref(0))
    end
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
    diagnostics::NewtonKrylovDiagnostics
    """
    """
    function NewtonKrylovSolverData(::Type{TFloat}, n_degrees_of_freedom::Int64;
                                    # relative tolerance for convergence of Newton iterations
                                    rtol::TFloatTol=1.0e-5,
                                    # absolute tolerance for convergence of Newton iterations
                                    atol::TFloatTol=1.0e-12,
                                    # max newton_solve! iterations
                                    nonlinear_max_iterations::Int64=20,
                                    # tolerance for GMRES linear solve
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
        return new{TFloat}(rtol, atol,
                nonlinear_max_iterations,
                preconditioner_update_interval,
                linear_rtol,
                linear_atol, krylov_subspace_max_size,
                H, c, s, g, V,
                residual, delta_x, rhs_delta, v, w,
                NewtonKrylovDiagnostics())
    end
end

"""
    newton_solve!(x, rhs_func!, nl_solver_params;
                  left_preconditioner=(x) -> nothing, right_preconditioner=(x) -> nothing)

`x` is the initial guess at the solution, and is overwritten by the result of the Newton
solve.

`rhs_func!(residual, x)` is the function we are trying to find a solution of. It calculates
```math
\\mathtt{residual} = R(\\mathtt{x})
```
where we are trying to solve \$R(x)=0\$.

`left_preconditioner` or `right_preconditioner` apply preconditioning. They should be
passed a function that solves \$P.x = b\$ where \$P\$ is the preconditioner matrix, \$b\$
is given by the values passed to the function as the argument, and the result \$x\$ is
returned by overwriting the argument.

Tolerances
----------

Note that the meaning of the relative tolerance `rtol` and absolute tolerance `atol` is
very different for the outer Newton iteration and the inner GMRES iteration.

For the outer Newton iteration the residual \$R(x^n)\$ measures the departure of the
system from the solution (at each grid point). Its size can be compared to the size of the
solution `x`, so it makes sense to define an `error norm' for \$R(x^n)\$ as
```math
E(x^n) = \\left\\lVert \\frac{R(x^n)}{\\mathtt{rtol} x^n \\mathtt{atol}} \\right\\rVert_2
```
where \$\\left\\lVert \\cdot \\right\\rVert\$ is the 'L2 norm' (square-root of sum of
squares). We can further try to define a grid-size independent error norm by dividing out
the number of grid points to get a root-mean-square (RMS) error rather than an L2 norm.
```math
E_{\\mathrm{RMS}}(x^n) = \\sqrt{ \\frac{1}{N} \\sum_i \\frac{R(x^n)_i}{\\mathtt{rtol} x^n_i \\mathtt{atol}} }
```
where \$N\$ is the total number of grid points.

In contrast, GMRES is constructed to minimise the L2 norm of \$r_k = b - A\\cdot x_k\$
where GMRES is solving the linear system \$A\\cdot x = b\$, \$x_k\$ is the approximation
to the solution \$x\$ at the \$k\$'th iteration and \$r_k\$ is the residual at the
\$k\$'th iteration. There is no flexibility to measure error relative to \$x\$ in any
sense. For GMRES, a `relative tolerance' is relative to the residual of the
right-hand-side \$b\$, which is the first iterate \$x_0\$ (when no initial guess is
given). [Where a non-zero initial guess is given it might be better to use a different
stopping criterion, see Carson et al. section 3.8.]. The stopping criterion for the GMRES
iteration is therefore
```
\\left\\lVert r_k \\right\\rVert < \\max(\\mathtt{linear\\_rtol} \\left\\lVert r_0 \\right\\rVert, \\mathtt{linear\\_atol}) = \\max(\\mathtt{linear\\_rtol} \\left\\lVert b \\right\\rVert, \\mathtt{linear\\_atol})
```
As the GMRES solve is only used to get the right `direction' for the next Newton step, it
is not necessary to have a very tight `linear_rtol` for the GMRES solve.
"""
function newton_solve!(x::TVector, residual_func!::TResidual,
            nl_solver_params::NewtonKrylovSolverData{TFloat};
            left_preconditioner::TPreconditionerLeft=(x) -> nothing,
            right_preconditioner::TPreconditionerRight=(x) -> nothing,
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

    norm_params = (nl_solver_params.rtol, nl_solver_params.atol, x)

    residual_func!(residual, x)
    residual_norm = vector_norm(residual, norm_params...)
    newton_iterations = 0
    GMRES_iterations = 0

    success = true
    while (newton_iterations < 1 && residual_norm > 1.0e-8) || residual_norm > 1.0
        newton_iterations += 1

        # use the GMRES algoritm to find the approximate solution to:
        #   J δx = -RHS(x)
        krylov_subspace_size = linear_solve!(x, residual_func!, residual, delta_x, v, w,
                                   norm_params;
                                   rtol=nl_solver_params.linear_rtol,
                                   atol=nl_solver_params.linear_atol,
                                   max_krylov_subspace_size=nl_solver_params.krylov_subspace_max_size,
                                   left_preconditioner=left_preconditioner,
                                   right_preconditioner=right_preconditioner,
                                   H=nl_solver_params.H, c=nl_solver_params.c,
                                   s=nl_solver_params.s, g=nl_solver_params.g,
                                   V=nl_solver_params.V, rhs_delta=nl_solver_params.rhs_delta)
        GMRES_iterations += krylov_subspace_size

        # calculate the residual for the NaN diagnostic check
        @. w = x + delta_x
        residual_func!(residual, w)
        residual_norm = vector_norm(residual, norm_params...)
        if isnan(residual_norm)
            error("NaN in Newton iteration at iteration $(newton_iterations)")
        end
        # update root estimate x_n+1 = x_n + delta_x
        @. x = w

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
        println("Final residual: ", residual_norm)
        println("Total linear (GMRES) iterations: ", GMRES_iterations)
        println("Linear (GMRES) iterations per Newton iteration: ", GMRES_iterations / newton_iterations)
        println()
    end

    return success
end

function vector_norm(residual::Array{TFloat, 1},
            rtol::TFloat, atol::TFloat, x::Vector{TFloat}) where TFloat <: AbstractFloat
    return sqrt(vector_dot_product(residual, residual, rtol, atol, x))
end

function vector_dot_product(v::Array{TFloat, 1}, w::Array{TFloat, 1},
            rtol::TFloat, atol::TFloat, x::Vector{TFloat}) where TFloat <: AbstractFloat
    dot_product = 0.0
    for i ∈ eachindex(v,w)
        dot_product += v[i] * w[i] / abs2(rtol * abs(x[i]) + atol)
    end
    dot_product = dot_product / length(v)
    return dot_product
end

function calculate_delta_x(delta_x::Array{TFloat, 1},
            V::Array{TFloat,2}, y::Vector{TFloat}) where TFloat <: AbstractFloat
    @. delta_x = 0.0
    for iy in eachindex(y)
        for icoord in eachindex(delta_x)
            delta_x[icoord] += y[iy] * V[icoord,iy]
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
function linear_solve!(x::TVector, residual_func!::TResidual,
            residual0::TVector, delta_x::TVector, v::TVector, w::TVector,
            norm_params; rtol, atol, max_krylov_subspace_size::Int64,
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

    Jv_scale_factor = 1.0e3
    inv_Jv_scale_factor = 1.0 / Jv_scale_factor

    # The vectors `v` that are passed to this function will be normalised so that
    # `vector_norm(v) == 1.0`. `vector_norm()` is defined - including the
    # relative and absolute tolerances from the Newton iteration - so that a vector with a
    # norm of 1.0 is 'small' in the sense that a vector with a norm of 1.0 is small enough
    # relative to `x` to consider the iteration converged. This means that `x+v` would be
    # very close to `x`, so R(x+v)-R(x) would be likely to be badly affected by rounding
    # errors, because `v` is so small, relative to `x`. We actually want to multiply `v`
    # by a large number `Jv_scale_factor` (in constrast to the small `epsilon` in the
    # 'usual' case where the norm does not include either reative or absolute tolerance)
    # to ensure that we get a reasonable estimate of J.v.
    function approximate_Jacobian_vector_product!(v::Vector{TFloat}) where TFloat <: AbstractFloat
        right_preconditioner(v)
        @. v = x + Jv_scale_factor * v
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
    beta = vector_norm(w, norm_params...)
    for i in eachindex(w)
        V[i,1] = w[i]/beta
    end
    g[1] = beta

    # Set tolerance for GMRES iteration to rtol times the initial residual, unless this is
    # so small that it is smaller than atol, in which case use atol instead.
    tol = max(rtol * beta, atol)

    krylov_subspace_size = 0
    # set H to zero to ensure lower-than-diagonal entries
    # of the upper Hessenberg matrix are zero
    @. H = 0.0
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
            w_dot_Vj = vector_dot_product(w, v, norm_params...)

            H[j,i] = w_dot_Vj

            for k in eachindex(w)
                w[k] = w[k] - H[j,i] * V[k,j]
            end
        end
        norm_w = vector_norm(w, norm_params...)

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
