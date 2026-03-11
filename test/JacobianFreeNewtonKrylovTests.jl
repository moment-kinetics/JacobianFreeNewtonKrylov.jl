module JacobianFreeNewtonKrylovTests

using Test: @testset, @test
using JacobianFreeNewtonKrylov

@enum PreconditionerOptionType begin
    use_right_preconditioner
    use_left_preconditioner
    no_preconditioner
end

function linear_test(; n=16, max_nkrylov = 12, atol=1.0e-13, atol_expected=1.0e-5, nonlinear_max_iterations=100)
    println("    - linear test")
    @testset "linear test " begin
        # Test represents constant-coefficient diffusion, in 1D steady state, with a
        # central finite-difference discretisation of the second derivative.
        #
        # Note, need to use newton_solve!() here even though it is a linear problem,
        # because the inexact Jacobian-vector product we use in linear_solve!() means
        # linear_solve!() on its own does not converge to the correct answer.

        #n = 16
        restart = max_nkrylov
        #atol = 1.0e-13

        # the grid z
        z = collect(0:n-1) ./ (n-1)
        # grid spacing
        Dz = z[2] - z[1]
        # source s = - b > 0
        b = @. - z * (1.0 - z)
        # diffusion coefficient
        Ddiffuse = 0.05
        # including Dz spacing
        Dk = (Ddiffuse / Dz^2)

        A = zeros(n,n)
        i = 1
        # boundary condition row
        # residual = 0 when x[i] = 1.0
        A[i,i] = 1.0
        b[i] = 1.0
        # central differences d^2/d z^2 for interior points
        for i ∈ 2:n-1
            A[i,i-1] = 1.0 * Dk
            A[i,i] = -2.0 * Dk
            A[i,i+1] = 1.0 * Dk
        end
        i = n
        # boundary condition row
        # residual = 0 when x[i] = 1.0
        A[i,i] = 1.0
        b[i] = 1.0

        function rhs_func!(residual, x)
            residual .= A * x - b
            return nothing
        end

        x = Array{Float64,1}(undef,n)
        # initial guess (respects b.c. x[1] = x[n] = 1.0)
        x .= 1.0

        nl_solver_params = nl_solver_info(
            length(x),
            rtol = 0.0,
            atol = atol,
            linear_restart = restart,
            linear_rtol=0.0,
            nonlinear_max_iterations = nonlinear_max_iterations)

        newton_solve!(x, rhs_func!, nl_solver_params,
            diagnose = true)

        x_direct = A \ b

        @test isapprox(x, x_direct; atol=100.0*atol)

        x_expected = deepcopy(z)

        @. x_expected = 1.0 + (1.0/(12.0 * Ddiffuse))*(z + z^4 - 2.0 * z^3)

        @test isapprox(x_direct, x_expected; atol=atol_expected)
    end
end

function nonlinear_test(;
            preconditioner_option::PreconditionerOptionType=no_preconditioner,
            nonlinear_max_iterations=100)
    println("    - non-linear test: $preconditioner_option")
    @testset "non-linear test: $preconditioner_option" begin
        # Test represents constant-coefficient diffusion, in 1D steady state, with a
        # central finite-difference discretisation of the second derivative.
        #
        # Note, need to use newton_solve!() here even though it is a linear problem,
        # because the inexact Jacobian-vector product we use in linear_solve!() means
        # linear_solve!() on its own does not converge to the correct answer.

        n = 64
        restart = 12
        atol = 1.0e-15

        # the grid z
        z = collect(0:n-1) ./ (n-1)
        # grid spacing
        Dz = z[2] - z[1]
        # Delta t
        Dt = 100.0
        # RHS > 0
        b = @. Dt * z * (1.0 - z)

        function rhs_func!(residual, x)
            i = 1
            D = 1.0 + abs(x[i])^2.5
            residual[i] = x[i] - (Dt / Dz^2) * D * (- 2.0 * x[i] + x[i+1]) - b[i]
            for i ∈ 2:n-1
                D = 1.0 + abs(x[i])^2.5
                residual[i] = x[i] - (Dt / Dz^2) * D * (x[i-1] - 2.0 * x[i] + x[i+1]) - b[i]
            end
            i = n
            D = 1.0 + abs(x[i])^2.5
            residual[i] = x[i] - (Dt / Dz^2) * D * (x[i-1] - 2.0 * x[i]) / Dz^2 - b[i]
            return nothing
        end

        A = zeros(n,n)
        i = 1
        A[i,i] = -2.0
        A[i,i+1] = 1.0
        for i ∈ 2:n-1
            A[i,i-1] = 1.0
            A[i,i] = -2.0
            A[i,i+1] = 1.0
        end
        i = n
        A[i,i-1] = 1.0
        A[i,i] = -2.0
        # a preconditioner
        P = zeros(n,n)
        for i in eachindex(z)
            P[i,i] = 1.0
            @views @. P[i,:] -= (Dt / Dz^2) * A[i,:]
        end
        dummyx = Array{Float64,1}(undef,n)
        # function to apply the preconditioner in place
        function preconditioner!(x)
            dummyx .= x
            x .= P \ dummyx
            return nothing
        end

        # choose whether to use a right or left preconditioner in test
        if preconditioner_option == use_left_preconditioner
            left_preconditioner = preconditioner!
            right_preconditioner = identity
        elseif preconditioner_option == use_right_preconditioner
            left_preconditioner = identity
            right_preconditioner = preconditioner!
        elseif preconditioner_option == no_preconditioner
            left_preconditioner = identity
            right_preconditioner = identity
        end

        # the solution vector
        x = Array{Float64,1}(undef,n)
        # initial condition
        x .= 0.0

        nl_solver_params = nl_solver_info(
            length(x),
            rtol = 0.0,
            atol = atol,
            linear_restart = restart,
            nonlinear_max_iterations = nonlinear_max_iterations)

        newton_solve!(x, rhs_func!, nl_solver_params;
            left_preconditioner=left_preconditioner,
            right_preconditioner=right_preconditioner,
            diagnose = true)

        rhs_func!(nl_solver_params.residual, x)

        @test isapprox(nl_solver_params.residual, zeros(n); atol=4.0*atol)
    end
end

function runtests()
    @testset "non-linear solvers" begin
        println("non-linear solver tests")
        linear_test(n=16, atol_expected=1.0e-2)
        linear_test(n=32, atol_expected=(1.0/2.0)*1.0e-2)
        linear_test(n=64, atol_expected=(1.0/4.0)*1.0e-2, max_nkrylov=16)
        linear_test(n=128, atol_expected=(1.0/8.0)*1.0e-2, atol=1.0e-12, max_nkrylov=32)
        #nonlinear_test(preconditioner_option=no_preconditioner, nonlinear_max_iterations=100)
        #nonlinear_test(preconditioner_option=use_left_preconditioner, nonlinear_max_iterations=100)
        #nonlinear_test(preconditioner_option=use_right_preconditioner, nonlinear_max_iterations=100)
    end
end

end # JacobianFreeNewtonKrylovTests
using .JacobianFreeNewtonKrylovTests
JacobianFreeNewtonKrylovTests.runtests()
