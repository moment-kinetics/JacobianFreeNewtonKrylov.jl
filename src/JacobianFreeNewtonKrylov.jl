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

export reset_nonlinear_per_stage_counters!,
       newton_solve!,
       nl_solver_info
# promote type to user defined function?
jfnk_float = Float64
jfnk_int = Int64

struct nl_solver_info
    rtol::jfnk_float
    atol::jfnk_float
    nonlinear_max_iterations::jfnk_int
    linear_rtol::jfnk_float
    linear_atol::jfnk_float
    linear_restart::jfnk_int
    H::Array{jfnk_float,2}
    c::Array{jfnk_float,1}
    s::Array{jfnk_float,1}
    g::Array{jfnk_float,1}
    V::Array{jfnk_float,2}
    n_solves::Base.RefValue{jfnk_int}
    nonlinear_iterations::Base.RefValue{jfnk_int}
    linear_iterations::Base.RefValue{jfnk_int}
    precon_iterations::Base.RefValue{jfnk_int}
    solves_since_precon_update::Base.RefValue{jfnk_int}
    max_nonlinear_iterations_this_step::Base.RefValue{jfnk_int}
    max_linear_iterations_this_step::Base.RefValue{jfnk_int}
    preconditioner_update_interval::jfnk_int
    residual::Vector{jfnk_float}
    delta_x::Vector{jfnk_float}
    rhs_delta::Vector{jfnk_float}
    v::Vector{jfnk_float}
    w::Vector{jfnk_float}
    """
    """
    function nl_solver_info(n_degrees_of_freedom;
                                    # relative tolerance for convergence
                                    rtol=1.0e-5,
                                    # absolute tolerance for convergence
                                    atol=1.0e-12,
                                    # max newton_solve! iterations
                                    nonlinear_max_iterations=20,
                                    linear_rtol=1.0e-3,
                                    linear_atol=1.0,
                                    # number of members of Krylov subspace
                                    linear_restart=10,
                                    preconditioner_update_interval=300)
        H = Array{jfnk_float,2}(undef, linear_restart + 1, linear_restart)
        c = Array{jfnk_float,1}(undef, linear_restart + 1)
        s = Array{jfnk_float,1}(undef, linear_restart + 1)
        g = Array{jfnk_float,1}(undef, linear_restart + 1)
        V = Array{jfnk_float,2}(undef, n_degrees_of_freedom, linear_restart+1)
        # suspicious that we need to zero the dummy arrays
        H .= 0.0
        c .= 0.0
        s .= 0.0
        g .= 0.0
        V .= 0.0
        # buffer arrays, previously input parameters to newton_solve!()
        residual = Vector{jfnk_float}(undef, n_degrees_of_freedom)
        delta_x = Vector{jfnk_float}(undef, n_degrees_of_freedom)
        rhs_delta = Vector{jfnk_float}(undef, n_degrees_of_freedom)
        v = Vector{jfnk_float}(undef, n_degrees_of_freedom)
        w = Vector{jfnk_float}(undef, n_degrees_of_freedom)
        return new(jfnk_float(rtol), jfnk_float(atol),
                nonlinear_max_iterations,
                jfnk_float(linear_rtol),
                jfnk_float(linear_atol), linear_restart,
                H, c, s, g, V,
                Ref(0), Ref(0), Ref(0), Ref(0),
                Ref(preconditioner_update_interval),
                Ref(0), Ref(0),
                preconditioner_update_interval,
                residual, delta_x, rhs_delta, v, w)
    end
end

"""
    reset_nonlinear_per_stage_counters!(nl_solver_params::Union{nl_solver_info,Nothing})

Reset the counters that hold per-step totals or maximums in `nl_solver_params`.

Also increment `nl_solver_params.stage_counter[]`.
"""
function reset_nonlinear_per_stage_counters!(nl_solver_params::Union{nl_solver_info,Nothing})
    if nl_solver_params === nothing
        return nothing
    end

    nl_solver_params.max_nonlinear_iterations_this_step[] = 0
    nl_solver_params.max_linear_iterations_this_step[] = 0

    # Also increment the stage counter
    nl_solver_params.solves_since_precon_update[] += 1

    return nothing
end

"""
    newton_solve!(x, rhs_func!, nl_solver_params;
                  left_preconditioner=nothing, right_preconditioner=nothing)

`x` is the initial guess at the solution, and is overwritten by the result of the Newton
solve.

`rhs_func!(residual, x)` is the function we are trying to find a solution of. It calculates
```math
\\mathtt{residual} = F(\\mathtt{x})
```
where we are trying to solve \$F(x)=0\$.

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
function newton_solve!(x::TVector, residual_func!::TFunc,
            nl_solver_params::nl_solver_info;
            left_preconditioner=nothing,
            right_preconditioner=nothing,
            recalculate_preconditioner=nothing) where {
                TVector <: AbstractArray{jfnk_float,1},
                TFunc <: Function}
    rtol = nl_solver_params.rtol
    atol = nl_solver_params.atol
    residual = nl_solver_params.residual
    delta_x = nl_solver_params.delta_x
    v = nl_solver_params.v
    w = nl_solver_params.w
    if left_preconditioner === nothing
        left_preconditioner = identity
    end
    if right_preconditioner === nothing
        right_preconditioner = identity
    end

    norm_params = (nl_solver_params.rtol, nl_solver_params.atol, x)

    residual_func!(residual, x)
    residual_norm = vector_norm(residual, norm_params...)
    counter = 0
    linear_counter = 0

    # Would need this if delta_x was not set to zero within the Newton iteration loop
    # below.
    #@. delta_x = 0.0

    close_counter = -1
    close_linear_counter = -1
    success = true
    previous_residual_norm = residual_norm
old_precon_iterations = nl_solver_params.precon_iterations[]
    while (counter < 1 && residual_norm > 1.0e-8) || residual_norm > 1.0
        counter += 1
        #println("\nNewton ", counter)

        # Solve (approximately?):
        #   J δx = -RHS(x)
        linear_its = linear_solve!(x, residual_func!, residual, delta_x, v, w,
                                   norm_params;
                                   rtol=nl_solver_params.linear_rtol,
                                   atol=nl_solver_params.linear_atol,
                                   restart=nl_solver_params.linear_restart,
                                   left_preconditioner=left_preconditioner,
                                   right_preconditioner=right_preconditioner,
                                   H=nl_solver_params.H, c=nl_solver_params.c,
                                   s=nl_solver_params.s, g=nl_solver_params.g,
                                   V=nl_solver_params.V, rhs_delta=nl_solver_params.rhs_delta)
        linear_counter += linear_its

        # If the residual does not decrease, we will do a line search to find an update
        # that does decrease the residual. The value of `x` is used to define the
        # normalisation value with rtol that is used to calculate the residual, so do not
        # want to update it until the line search is completed (otherwise the norm changes
        # during the line search, which might make it fail to converge). So calculate the
        # updated value in the buffer `w` until the line search is completed, and only
        # then copy it into `x`.
        @. w = x + delta_x
        residual_func!(residual, w)

        # For the Newton iteration, we want the norm divided by the (sqrt of the) number
        # of grid points, so we can use a tolerance that is independent of the size of the
        # grid. This is unlike the norms needed in `linear_solve!()`.
        residual_norm = vector_norm(residual, norm_params...)
        if isnan(residual_norm)
            error("NaN in Newton iteration at iteration $counter")
        end
        @. x = w
        previous_residual_norm = residual_norm

        if recalculate_preconditioner !== nothing && counter % nl_solver_params.preconditioner_update_interval == 0
            # Have taken a large number of Newton iterations already - convergence must be
            # slow, so try updating the preconditioner.
            recalculate_preconditioner()
        end

        #println("Newton residual ", residual_norm, " ", linear_its, " $rtol $atol")

        if residual_norm < 0.1/rtol && close_counter < 0 && close_linear_counter < 0
            close_counter = counter
            close_linear_counter = linear_counter
        end

        if counter > nl_solver_params.nonlinear_max_iterations
            println("maximum iteration limit reached")
            success = false
            break
        end
    end
    nl_solver_params.n_solves[] += 1
    nl_solver_params.nonlinear_iterations[] += counter
    nl_solver_params.linear_iterations[] += linear_counter
    nl_solver_params.max_nonlinear_iterations_this_step[] =
        max(counter, nl_solver_params.max_nonlinear_iterations_this_step[])
    nl_solver_params.max_linear_iterations_this_step[] =
        max(linear_counter, nl_solver_params.max_linear_iterations_this_step[])
#    println("Newton iterations: ", counter)
#    println("Final residual: ", residual_norm)
#    println("Total linear iterations: ", linear_counter)
#    println("Linear iterations per Newton: ", linear_counter / counter)
#    precon_count = nl_solver_params.precon_iterations[] - old_precon_iterations
#    println("Total precon iterations: ", precon_count)
#    println("Precon iterations per linear: ", precon_count / linear_counter)
#
#    println("Newton iterations after close: ", counter - close_counter)
#    println("Total linear iterations after close: ", linear_counter - close_linear_counter)
#    println("Linear iterations per Newton after close: ", (linear_counter - close_linear_counter) / (counter - close_counter))
#    println()

    return success
end

function vector_norm(residual::Array{jfnk_float, 1},
                               rtol, atol, x)
    return sqrt(vector_dot_product(residual, residual, rtol, atol, x))
end

function vector_dot_product(v::Array{jfnk_float, 1}, w::Array{jfnk_float, 1},
                  rtol, atol, x)
    dot_product = 0.0
    for i ∈ eachindex(v,w)
        dot_product += v[i] * w[i] / (rtol * abs(x[i]) + atol)^2
    end
    dot_product = dot_product / length(v)
    return dot_product
end

function calculate_delta_x(delta_x::Array{jfnk_float, 1}, V, y)
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
function linear_solve!(x, residual_func!, residual0, delta_x, v, w,
                    norm_params; rtol, atol, restart,
                    left_preconditioner, right_preconditioner, H, c, s, g, V,
                    rhs_delta)
    # Solve (approximately?):
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
    function approximate_Jacobian_vector_product!(v, skip_first_precon::Bool=false)
        if !skip_first_precon
            right_preconditioner(v)
        end

        @. v = x + Jv_scale_factor * v
        residual_func!(rhs_delta, v)
        @. v = (rhs_delta - residual0) * inv_Jv_scale_factor
        left_preconditioner(v)
        return v
    end

    # To start with we use 'w' as a buffer to make a copy of residual0 to which we can apply
    # the left-preconditioner.
    @. v = 0.0
    left_preconditioner(residual0)

    # This function transforms the data stored in 'v' from δx to ≈J.δx
    # If initial δx is all-zero, we can skip a right-preconditioner evaluation because it
    # would just transform all-zero to all-zero.
    approximate_Jacobian_vector_product!(v, true)

    # Now we actually set 'w' as the first Krylov vector, and normalise it.
    @. w = -residual0 - v
    beta = vector_norm(w, norm_params...)
    for i in eachindex(w)
        V[i,1] = w[i]/beta
    end
    g[1] = beta

    # Set tolerance for GMRES iteration to rtol times the initial residual, unless this is
    # so small that it is smaller than atol, in which case use atol instead.
    tol = max(rtol * beta, atol)

    lsq_result = nothing
    residual = Inf
    counter = 0
    inner_counter = 0
    for i ∈ 1:restart
        inner_counter = i
        counter += 1
        #println("Linear ", counter)

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
    i = inner_counter

    # finally, compute delta_x
    #################################

    @views y = H[1:i,1:i] \ g[1:i]

    # The following calculates
    #    delta_x .= sum(y[i] .* V[:,i] for i ∈ 1:length(y))
    calculate_delta_x(delta_x, V, y)
    right_preconditioner(delta_x)

    return counter
end

end
