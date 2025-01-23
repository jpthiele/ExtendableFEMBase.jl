function run_feevaluator_tests()
    return @testset "FEEvaluator" begin
        println("\n")
        println("===================================")
        println("Testing FEEvaluators for H1P1 in 2D")
        println("===================================")

        # unit square with 2 elements
        X = LinRange(0, 1, 2)
        xgrid = simplexgrid(X, X)
        EG = xgrid[UniqueCellGeometries][1]
        qf = QuadratureRule{Float64, EG}(1)

        # Test scalar valued version
        FES = FESpace{H1P1{1}}(xgrid)
        FEBasis_∇ = FEEvaluator(FES, Gradient, qf)
        FEBasis_∇.citem[] = 1
        update_basis!(FEBasis_∇)
        @test FEBasis_∇.cvals == [-1.0 1.0 0.0; 0.0 -1.0 1.0;;;]
        FEBasis_∇.citem[] = 2
        update_basis!(FEBasis_∇)
        @test FEBasis_∇.cvals == [1.0 -1.0 0.0; 0.0 1.0 -1.0;;;]

        # Test vector valued version
        FES_vec = FESpace{H1P1{2}}(xgrid)
        FEBasis_∇_vec = FEEvaluator(FES_vec, Gradient, qf)
        FEBasis_∇_vec.citem[] = 1
        update_basis!(FEBasis_∇_vec)
        @test FEBasis_∇_vec.cvals == [-1.0 1.0 0.0 0.0 0.0 0.0; 0.0 -1.0 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 -1.0 1.0 0.0; 0.0 0.0 0.0 0.0 -1.0 1.0;;;]
        FEBasis_∇_vec.citem[] = 2
        update_basis!(FEBasis_∇_vec)
        @test FEBasis_∇_vec.cvals == [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 1.0 -1.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0 -1.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0;;;]

    end
end
