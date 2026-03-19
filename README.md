# JacobianFreeNewtonKrylov

A package implementing a Jacobian-free Newton-Krylov method [1-3] for solving nonlinear systems in serial.

This class of solvers use an outer Newton iteration to solve the system
```math
   R(x) = 0,
```
where $`R`$ is the residual function and $`x`$ is the solution vector.
Each step of the Newton iteration requires a solution to the linearised system
```math
   J\cdot \delta\! x = -R(x^n)
```
for the $`n^{\rm th}`$ iteration of the solution vector $`x^n`$, where
```math
   J = \frac{\partial R}{\partial x}
``` is the Jacobian.
Here, we use the GMRES method [4-7] to solve the linear system,
which is a method that uses the Krylov subspace to obtain a solution
without an explicit expression for J. Instead, GMRES computes the $`\delta\! x`$ which minimises
```math
    r = || J\cdot \delta\! x + R(x^n) ||
```
For this method, only the product $`J\cdot v`$ for a vector $`v`$ is required.
We compute this product using the finite difference
```math
    J\cdot v = \frac{R(x + \epsilon v) - R(x)}{\epsilon} + O( \epsilon v)
```
for $`\epsilon`$ a suitably sized number for the estimate
```math
    O(\epsilon v) << J\cdot v
```
to hold.

Useful references:
[1] D.A. Knoll, D.E. Keyes, "Jacobian-free Newton–Krylov methods: a survey of approaches and applications", Journal of Computational Physics, Volume 193, 2004, Pages 357-397, https://doi.org/10.1016/j.jcp.2003.08.010.
[2] V.A. Mousseau and D.A. Knoll, "Fully Implicit Kinetic Solution of Collisional Plasmas", Journal of Computational Physics 136, 308–323 (1997), https://doi.org/10.1006/jcph.1997.5736.
[3] V.A. Mousseau, "Fully Implicit Kinetic Modelling of Collisional Plasmas", PhD thesis, Idaho National Engineering Laboratory (1996), https://inis.iaea.org/collection/NCLCollectionStore/_Public/27/067/27067141.pdf.
[4] https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
[5] https://www.rikvoorhaar.com/blog/gmres
[6] E. Carson , J. Liesen, Z. Strakoš, "Towards understanding CG and GMRES through examples", Linear Algebra and its Applications 692, 241–291 (2024), https://doi.org/10.1016/j.laa.2024.04.003.
[7] Q. Zou, "GMRES algorithms over 35 years", Applied Mathematics and Computation 445, 127869 (2023), https://doi.org/10.1016/j.amc.2023.127869

# Usage

The interface to use the package is documented in `test/JacobianFreeNewtonKrylovTests.jl`. Here we provide a brief summary. It is assumed that the solution is contained in a `Vector{T} where T <: AbstractFloat`, i.e.,
```
# solution_vector
x = zeros(Float64,npoints)
```
where `npoints::Int64` is the number of degrees of freedom in the system. The user must write a function which evaluates the residual vector `residual::Vector{T}` e.g.,
```
function rhs_func!(residual, x)
    # compute the residual vector
    for i in eachindex(x)
    end
    return nothing
end
```
The buffer arrays and parameters necessary to control the Newton-Krylov solve must be initialised.
```
nl_solver_params = NewtonKrylovSolverData(
    typeof(x[1]), length(x),
    rtol = rtol, # The relative tolerance used in the Newton iteration
    atol = atol, # The absolute tolerance used in the Newton iteration
    krylov_subspace_max_size = 10, # The maximum number of members in the Krylov subspace used in GMRES.
    linear_rtol=1.0e-3, # tolerance for the GMRES solve, relative to size of the 1st Krylov vector
    linear_atol=1.0, # tolerance for the GMRES solve, relative
    nonlinear_max_iterations = nonlinear_max_iterations)
```

We can define functions for the preconditioner as follows
```
function recalculate_preconditioner()
    # calculate the preconditioner
    return nothing
end
# function to apply the left preconditioner in place
function left_preconditioner(x)
    # apply the preconditioner
    return nothing
end
# function to apply the right preconditioner in place
function right_preconditioner(x)
    # apply the preconditioner
    return nothing
end
```
Finally, we impose some initial condition on $`x`$ and we execute the Newton iteration.
```
@. x = initial_condition
newton_solve!(x, rhs_func!, nl_solver_params;
    left_preconditioner=left_preconditioner,
    right_preconditioner=right_preconditioner,
    recalculate_preconditioner=recalculate_preconditioner,
    diagnose = false)
```

