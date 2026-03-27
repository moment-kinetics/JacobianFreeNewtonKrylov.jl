# JacobianFreeNewtonKrylov

A package implementing a Jacobian-free Newton-Krylov method [1-3](#references) for solving nonlinear systems in serial. Newton-Krylov solvers use an outer Newton iteration to solve the system
```math
   R(x) = 0, \tag{1}
```
where $`R`$ is the residual function and $`x`$ is the solution vector.
Starting with an initial approximation of the root $`x^n`$, we linearise (1)
```math
    R(x) = R(x^n + \delta\! x) = J \cdot \delta\! x + R(x^n) + O(\delta\! x^2),
```
where
```math
   J = \frac{\partial R}{\partial x}
```
is the Jacobian matrix. The next approximation to the root is therefore given by 
```math
    x^{n+1} = x^n + \delta\! x
```
Each step of the Newton iteration requires a solution to the linearised system
```math
   J\cdot \delta\! x = -R(x^n) \tag{2}
```
for the $`n^{\rm th}`$ iteration of the solution vector $`x^n`$
Here, we use the weighted GMRES method [4-7](#references) to solve the linear system (2), which is a method that uses the Krylov subspace to obtain a solution without an explicit expression for $`J`$. Instead, GMRES computes the $`\delta\! x`$ which minimises
```math
    ||r(\delta\!x)||_W = || J\cdot \delta\! x + R(x^n) ||_W,
```
Where $`r = J\cdot \delta\! x + R(x^n)`$ is the GMRES residual, the norm $`|| \cdot ||_W`$ is defined through the weighted inner product $`(u,v)_W`$ by
```math
|| u ||_W = \sqrt{(u,u)_W},
```
with
```math
(u,v)_W = \sum_j \frac{u_j v_j W_j}{N},
```
and $`N`$ is the length of $`u`$. See Algorithm 2 of reference [7](#references), in particular, section 3.3 for a description of weighted GMRES.

In the GMRES method, only the product $`J\cdot v`$ for a vector $`v`$ is required. We compute this product using the finite difference
```math
    J\cdot v = \frac{R(x + \epsilon v) - R(x)}{\epsilon} + O( \epsilon v)
```
for $`\epsilon`$ a suitably sized number for the estimate
```math
    O(\epsilon v) << J\cdot v
```
to hold.

# Choice of weights, convergence, and definition of tolerances in the Newton method

We can choose the weight vector $`W`$ to introduce a concept of pointwise relative error into the norm $`||\cdot||_W`$. For the Newton method, we define
 - an absolute tolerance $`a_{tol}`$=`atol`
 - a relative tolerance $`r_{tol}`$=`rtol`

and we choose the weight to be defined by
```math
 W_j = \frac{1}{(a_{tol} + r_{tol}|x_j|)^2}
```
where $`|x_j|`$ is the absolute value of $`x_j`$. For this choice of $`W`$, the convergence criterion for the Newton method is
```math
    || R(x^n) ||_{W^{n-1}} < 1
```
where we indicate that the solution vector $`x^{n-1}`$ is used to define the weight $`W^{n-1}`$ that is used in the norm of the residual of the $`n^{\rm th}`$ iteration.

# Tolerances in the GMRES linear solve

The GMRES calculation of $`\delta\! x`$ is also controlled by tolerances. In notation similar to [7](#references) we identify $`A = J`$, $`b = R(x^n)`$ and $`u = \delta\! x`$. GMRES is constructed to solve
```math
    A \cdot u = b
```
by minimising
```math
   || r ||_W = ||b - A \cdot u||_W
```
In this system, when the initial guess for $`u=0`$, and the size of $`u`$ cannot be easily anticipated, we can define a convergence criterion by comparing the size of the residual after each GMRES iteration. The initial norm of the residual $`r`$ for $`u^0=0`$ is
```math
    \beta = || r^0 ||_W = || b||_W,
```
and subsequent residuals are denoted by $`r^m`$ for the inclusion of $`m`$ additional members of the Krylov subspace -- the first member is $`v^0 = b / \beta`$. 

We define the tolerances
 - $`a_{GMRES}`$=`linear_atol`
 - $`r_{GMRES}`$=`linear_rtol`

and we say that the GMRES iteration is converged when
```math
|| r^m ||_W < {\rm max}(a_{GMRES},r_{GMRES}\beta).
```
Since the norm $`|| \cdot  ||_W`$ contains the factor
```math
\frac{1}{|(a_{tol} + r_{tol}|x|)|} >> 1
```
convergence in the GMRES calculation can be reached for $`a_{GMRES}`$ = 1 and the relative tolerance $`r_{GMRES}\ll 1`$.

# A note on the Jacobian product $`J\cdot v`$

In the GMRES algorithm, the vectors used to construct the Krylov subspace that are used to represent $`u = \delta\!x`$ are normalised and orthogonalised using the Modified-Gram-Schmidt (MGS) procedure [7](#references). For example, the first member of the Krylov subspace is
```math
    v^0 = b / || b ||_W.
```
Noting our choice of weight, whilst we have that $`|| v^0 |_W = 1`$, the actual entries of the vector $`v^0_j`$ are numerically small. To avoid rounding errors in the finite difference
```math
    J\cdot v^0 = \frac{R(x + \epsilon v^0) - R(x)}{\epsilon}
```
we choose $`\epsilon = q ||1||_W`$, with $`q`$=`sqrt(eps(TFloat)) where TFloat <: AbstractFloat` and $`1`$ a vector of ones so that the product $`\epsilon v^0`$ has entries of size
```math
  \epsilon v^0_j \sim \verb+\sqrt(\eps())+.
```
For `Float64`, this would give entries of size `1.0e-8`.

<a name="references"/>

 # References:

 - [1] D.A. Knoll, D.E. Keyes, "Jacobian-free Newton–Krylov methods: a survey of approaches and applications", Journal of Computational Physics, Volume 193, 2004, Pages 357-397, https://doi.org/10.1016/j.jcp.2003.08.010.
 - [2] V.A. Mousseau and D.A. Knoll, "Fully Implicit Kinetic Solution of Collisional Plasmas", Journal of Computational Physics 136, 308–323 (1997), https://doi.org/10.1006/jcph.1997.5736.
 - [3] V.A. Mousseau, "Fully Implicit Kinetic Modelling of Collisional Plasmas", PhD thesis, Idaho National Engineering Laboratory (1996), https://inis.iaea.org/collection/NCLCollectionStore/_Public/27/067/27067141.pdf.
 - [4] https://en.wikipedia.org/wiki/Generalized_minimal_residual_method
 - [5] https://www.rikvoorhaar.com/blog/gmres
 - [6] E. Carson , J. Liesen, Z. Strakoš, "Towards understanding CG and GMRES through examples", Linear Algebra and its Applications 692, 241–291 (2024), https://doi.org/10.1016/j.laa.2024.04.003.
 - [7] Q. Zou, "GMRES algorithms over 35 years", Applied Mathematics and Computation 445, 127869 (2023), https://doi.org/10.1016/j.amc.2023.127869

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
    linear_rtol=1.0e-3, # tolerance for the GMRES solve, relative to size of the 1st Krylov vector, must be << 1
    linear_atol=1.0, # tolerance for the GMRES solve, must be <= 1
    nonlinear_max_iterations = nonlinear_max_iterations # Maximum number of Newton iterations
    )
```

We can define functions for the preconditioner as follows
```
function recalculate_preconditioner()
    # calculate the preconditioner
    return nothing
end
# function to apply the left preconditioner in place
function left_preconditioner(x)
    # apply the preconditioner to x in place
    return nothing
end
# function to apply the right preconditioner in place
function right_preconditioner(x)
    # apply the preconditioner to x in place
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
Setting the flag `diagnose = true` prints diagnostic information about the number of interations and the final residual.

