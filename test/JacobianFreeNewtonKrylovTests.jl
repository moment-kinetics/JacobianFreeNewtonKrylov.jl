module JacobianFreeNewtonKrylovTests

using Test: @testset, @test
using JacobianFreeNewtonKrylov

function linear_test()
    println("    - linear test")
    @testset "linear test " begin
        # Test represents constant-coefficient diffusion, in 1D steady state, with a
        # central finite-difference discretisation of the second derivative.
        #
        # Note, need to use newton_solve!() here even though it is a linear problem,
        # because the inexact Jacobian-vector product we use in linear_solve!() means
        # linear_solve!() on its own does not converge to the correct answer.

        n = 16
        restart = 8
        atol = 1.0e-10

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

        z = collect(0:n-1) ./ (n-1)
        b = @. - z * (1.0 - z)

        function rhs_func!(residual, x; krylov=false)
            residual .= A * x - b
            return nothing
        end

        x = Array{Float64,1}(undef,n)

        x .= 0.0

        nl_solver_params = nl_solver_info(
            length(x),
            rtol = 0.0,
            atol = atol,
            linear_restart = restart,
            )

        newton_solve!(x, rhs_func!, nl_solver_params)

        x_direct = A \ b

        @test isapprox(x, x_direct; atol=100.0*atol)
    end
end

function nonlinear_test()
    println("    - non-linear test")
    @testset "non-linear test" begin
        # Test represents constant-coefficient diffusion, in 1D steady state, with a
        # central finite-difference discretisation of the second derivative.
        #
        # Note, need to use newton_solve!() here even though it is a linear problem,
        # because the inexact Jacobian-vector product we use in linear_solve!() means
        # linear_solve!() on its own does not converge to the correct answer.

        n = 16
        restart = 10
        atol = 1.0e-10

        z = collect(0:n-1) ./ (n-1)
        b = @. - z * (1.0 - z)

        function rhs_func!(residual, x; krylov=false)
            i = 1
            D = abs(x[i])^2.5
            residual[i] = D * (- 2.0 * x[i] + x[i+1]) - b[i]
            for i ∈ 2:n-1
                D = abs(x[i])^2.5
                residual[i] = D * (x[i-1] - 2.0 * x[i] + x[i+1]) - b[i]
            end
            i = n
            D = abs(x[i])^2.5
            residual[i] = D * (x[i-1] - 2.0 * x[i]) - b[i]
            return nothing
        end

        x = Array{Float64,1}(undef,n)

        x .= 1.0

        nl_solver_params = nl_solver_info(
            length(x),
            rtol = 0.0,
            atol = atol,
            linear_restart = restart,
            nonlinear_max_iterations = 100)

        newton_solve!(x, rhs_func!, nl_solver_params)

        rhs_func!(nl_solver_params.residual, x)

        @test isapprox(nl_solver_params.residual, zeros(n); atol=4.0*atol)
    end
end

function runtests()
    @testset "non-linear solvers" begin
        println("non-linear solver tests")
        linear_test()
        nonlinear_test()
    end
end

end # JacobianFreeNewtonKrylovTests
using .JacobianFreeNewtonKrylovTests
JacobianFreeNewtonKrylovTests.runtests()
