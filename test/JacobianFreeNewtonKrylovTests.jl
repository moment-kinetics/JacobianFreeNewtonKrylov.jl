module JacobianFreeNewtonKrylovTests

using Test: @testset, @test
using JacobianFreeNewtonKrylov

@enum PreconditionerOptionType begin
    use_right_preconditioner
    use_left_preconditioner
    no_preconditioner
end

function linear_test(; n=16, max_nkrylov = 12, atol=1.0e-13, atol_expected=1.0e-5, nonlinear_max_iterations=100)
    println("    - linear test n=$n")
    @testset "linear test n=$n" begin
        # Test represents constant-coefficient diffusion, in 1D steady state, with a
        # central finite-difference discretisation of the second derivative.
        # solves:
        # Ddiffuse d^2 T / d z^2 + s = 0
        # with x = T and s = -b
        # Note, need to use newton_solve!() here even though it is a linear problem,
        # because the inexact Jacobian-vector product we use in linear_solve!() means
        # linear_solve!() on its own does not converge to the correct answer.

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

        nl_solver_params = NewtonKrylovSolverData(
            typeof(x[1]), length(x),
            rtol = 0.0,
            atol = atol,
            krylov_subspace_max_size = max_nkrylov,
            linear_rtol=0.0,
            nonlinear_max_iterations = nonlinear_max_iterations)

        @test newton_solve!(x, rhs_func!, nl_solver_params,
            diagnose = false)

        x_direct = A \ b

        @test isapprox(x, x_direct; atol=100.0*atol)

        x_expected = deepcopy(z)

        @. x_expected = 1.0 + (1.0/(12.0 * Ddiffuse))*(z + z^4 - 2.0 * z^3)

        @test isapprox(x_direct, x_expected; atol=atol_expected)
    end
end

function nonlinear_test(; n = 16 , atol = 1.0e-14, rtol=0.0, max_nkrylov = 12,
            preconditioner_option::PreconditionerOptionType=no_preconditioner,
            nonlinear_max_iterations=100)
    println("    - non-linear test: $preconditioner_option n=$n rtol=$rtol")
    @testset "non-linear test: $preconditioner_option n=$n rtol=$rtol" begin
        # Test represents constant-coefficient diffusion, in 1D steady state, with a
        # central finite-difference discretisation of the second derivative.
        # solves:
        # Ddiffuse T^(5/2) d^2 T / d z^2 + s = 0
        # with x = T and s = -b

        # the grid z
        z = collect(0:n-1) ./ (n-1)
        # grid spacing
        Dz = z[2] - z[1]
        # diffusion coefficient
        Ddiffuse = 0.05
        # including Dz spacing
        Dk = (Ddiffuse / Dz^2)
        # source s = - b = Ddiffuse*(1 + z - z^2)^5/2
        b = @. - 2.0*Ddiffuse*( 1.0 + z * (1.0 - z))^2.5

        # boundary conditions
        b[1] = 1.0
        b[n] = 1.0

        # the solution vector
        x = Array{Float64,1}(undef,n)
        # initial condition (respects b.c. x[1]=x[n]=1.0)
        x .= 1.0

        # residual function
        function rhs_func!(residual, x)
            # boundary condition row
            i = 1
            residual[i] = x[i] - b[i]
            # central differences d^2/d z^2 for interior points
            for i ∈ 2:n-1
                Dx = abs(x[i])^2.5
                residual[i] = Dk * Dx * (x[i-1] - 2.0 * x[i] + x[i+1]) - b[i]
            end
            # boundary condition row
            i = n
            residual[i] = x[i] - b[i]
            return nothing
        end

        # a preconditioner
        A = zeros(n,n)
        function calculate_preconditioner!()
            # boundary condition
            i = 1
            A[i,i] = 1.0
            for i ∈ 2:n-1
                Dx = abs(x[i])^2.5
                A[i,i-1] = 1.0 * Dk * Dx
                A[i,i] = -2.0 * Dk * Dx
                A[i,i+1] = 1.0 * Dk * Dx
            end
            # boundary condition
            i = n
            A[i,i] = 1.0
            return nothing
        end
        # calculate the initial preconditioner
        calculate_preconditioner!()
        dummyx = Array{Float64,1}(undef,n)
        # function to apply the preconditioner in place
        function preconditioner!(x)
            dummyx .= x
            x .= A \ dummyx
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

        nl_solver_params = NewtonKrylovSolverData(
            typeof(x[1]), length(x),
            rtol = rtol,
            atol = atol,
            krylov_subspace_max_size = max_nkrylov,
            linear_rtol=0.0,
            nonlinear_max_iterations = nonlinear_max_iterations)

        @test newton_solve!(x, rhs_func!, nl_solver_params;
            left_preconditioner=left_preconditioner,
            right_preconditioner=right_preconditioner,
            recalculate_preconditioner=calculate_preconditioner!,
            diagnose = false)

        rhs_func!(nl_solver_params.residual, x)

        # check the residual is small
        @test maximum(abs.(nl_solver_params.residual)) < 10.0*maximum(atol .+ rtol*abs.(x))

        # compare against the manufactured solution
        x_expected = deepcopy(x)
        @. x_expected = 1.0 + z*(1-z)
        @test maximum(abs.(x .- x_expected)) < 4.0*maximum(atol .+ rtol*abs.(x))
    end
end

function runtests()
    @testset "JacobianFreeNewtonKrylov Tests" begin
        println("JacobianFreeNewtonKrylov Tests")
        linear_test(n=16, atol_expected=1.0e-2)
        linear_test(n=32, atol_expected=(1.0/2.0)*1.0e-2)
        linear_test(n=64, atol_expected=(1.0/4.0)*1.0e-2, max_nkrylov=16)
        linear_test(n=128, atol_expected=(1.0/8.0)*1.0e-2, atol=1.0e-12, max_nkrylov=32)

        nonlinear_test(preconditioner_option=no_preconditioner, nonlinear_max_iterations=100)
        nonlinear_test(n=32, atol=2.0e-14, preconditioner_option=no_preconditioner, nonlinear_max_iterations=100)
        nonlinear_test(n=64, atol=8.0e-14, preconditioner_option=no_preconditioner, nonlinear_max_iterations=250)
        nonlinear_test(n=128, atol=32.0e-14, preconditioner_option=no_preconditioner, nonlinear_max_iterations=850)

        nonlinear_test(preconditioner_option=use_left_preconditioner, nonlinear_max_iterations=100)
        nonlinear_test(n=32, atol=2.0e-14, preconditioner_option=use_left_preconditioner, nonlinear_max_iterations=100)
        nonlinear_test(n=64, atol=8.0e-14, preconditioner_option=use_left_preconditioner, nonlinear_max_iterations=100)
        nonlinear_test(n=128, atol=32.0e-14, preconditioner_option=use_left_preconditioner, nonlinear_max_iterations=100)

        nonlinear_test(preconditioner_option=use_right_preconditioner, nonlinear_max_iterations=100)
        nonlinear_test(n=32, atol=2.0e-14, preconditioner_option=use_right_preconditioner, nonlinear_max_iterations=100)
        nonlinear_test(n=64, atol=8.0e-14, preconditioner_option=use_right_preconditioner, nonlinear_max_iterations=100)
        nonlinear_test(n=128, atol=32.0e-14, preconditioner_option=use_right_preconditioner, nonlinear_max_iterations=100)

        # speed up convergence with relative tolerances
        nonlinear_test(n=128, atol=32.0e-14, rtol=1.0e-3, preconditioner_option=use_right_preconditioner, nonlinear_max_iterations=3, max_nkrylov = 4)
        nonlinear_test(n=128, atol=32.0e-14, rtol=1.0e-4, preconditioner_option=use_right_preconditioner, nonlinear_max_iterations=4, max_nkrylov = 4)
        nonlinear_test(n=128, atol=32.0e-14, rtol=1.0e-6, preconditioner_option=use_right_preconditioner, nonlinear_max_iterations=5, max_nkrylov = 6)
        nonlinear_test(n=128, atol=32.0e-14, rtol=1.0e-8, preconditioner_option=use_right_preconditioner, nonlinear_max_iterations=6, max_nkrylov = 8)
    end
end

end # JacobianFreeNewtonKrylovTests
using .JacobianFreeNewtonKrylovTests
JacobianFreeNewtonKrylovTests.runtests()
