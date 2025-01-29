function run_feevaluator_tests()
    return @testset "FEEvaluator" begin
        println("\n")
        println("====================================")
        println("Testing FEEvaluators for H1 Elements")
        println("====================================")

        # unit square with 2 elements
        X = LinRange(0, 1, 2)
        xgrid = simplexgrid(X, X)
        EG = xgrid[UniqueCellGeometries][1]
        qf = QuadratureRule{Float64, EG}(1)

        @info "Scalar H1P1 function gradient"
        FES = FESpace{H1P1{1}}(xgrid)
        FEBasis_∇ = FEEvaluator(FES, Gradient, qf)
        FEBasis_∇.citem[] = 1
        update_basis!(FEBasis_∇)
        @test FEBasis_∇.cvals == [-1.0 1.0 0.0; 0.0 -1.0 1.0;;;]
        FEBasis_∇.citem[] = 2
        allocs = @allocated update_basis!(FEBasis_∇)
        @test FEBasis_∇.cvals == [1.0 -1.0 0.0; 0.0 1.0 -1.0;;;]
        @test allocs == 0

        @info "Vector valued H1P1 function gradient"
        FES_vec = FESpace{H1P1{2}}(xgrid)
        FEBasis_∇_vec = FEEvaluator(FES_vec, Gradient, qf)
        FEBasis_∇_vec.citem[] = 1
        update_basis!(FEBasis_∇_vec)
        @test FEBasis_∇_vec.cvals == [-1.0 1.0 0.0 0.0 0.0 0.0; 0.0 -1.0 1.0 0.0 0.0 0.0; 0.0 0.0 0.0 -1.0 1.0 0.0; 0.0 0.0 0.0 0.0 -1.0 1.0;;;]
        FEBasis_∇_vec.citem[] = 2
        allocs = @allocated update_basis!(FEBasis_∇_vec)
        @test FEBasis_∇_vec.cvals == [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 1.0 -1.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0 -1.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0;;;]
        @test allocs == 0

        @info "Vector valued H1P1 sym gradient"
        FES_vec = FESpace{H1P1{2}}(xgrid)
        FEBasis_sym∇ = FEEvaluator(FES_vec, SymmetricGradient{0.5}, qf)
        FEBasis_sym∇.citem[] = 1
        update_basis!(FEBasis_sym∇)
        @test FEBasis_sym∇.cvals == [-1.0 1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 -1.0 1.0; 0.0 -0.5 0.5 -0.5 0.5 0.0;;;]
        FEBasis_sym∇.citem[] = 2
        allocs = @allocated update_basis!(FEBasis_sym∇)
        @test FEBasis_sym∇.cvals == [1.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 -1.0; 0.0 0.5 -0.5 0.5 -0.5 0.0;;;]
        @test allocs == 0

        @info "Vector valued H1P1 function divergence"
        FEBasis_div = FEEvaluator(FES_vec, Divergence, qf)
        FEBasis_div.citem[] = 1
        update_basis!(FEBasis_div)
        @test FEBasis_div.cvals == [-1.0 1.0 0.0 0.0 -1.0 1.0;;;]
        FEBasis_div.citem[] = 2
        allocs = @allocated update_basis!(FEBasis_div)
        @test FEBasis_div.cvals == [1.0 -1.0 0.0 0.0 1.0 -1.0;;;]
        @test allocs == 0


        @info "Vector valued H1BR function gradient"
        qf = QuadratureRule{Float64, EG}(3)
        FES_BR = FESpace{H1BR{2}}(xgrid)
        FEBasis_∇_BR = FEEvaluator(FES_BR, Gradient, qf)
        FEBasis_∇_BR.citem[] = 1
        update_basis!(FEBasis_∇_BR)
        @test FEBasis_∇_BR.cvals ≈
            [
            -1.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0713523695816984 0.7575605255714951;
            0.0 -1.0 1.0 0.0 0.0 0.0 -0.0 -0.14104621525160543 -2.8272543712414024;
            0.0 0.0 0.0 -1.0 1.0 0.0 -3.0680353217581158 0.0 -0.7575605255714951;
            0.0 0.0 0.0 0.0 -1.0 1.0 3.998341476088209 -0.0 2.8272543712414024;;;
            -1.0 1.0 0.0 0.0 0.0 0.0 0.0 3.998341476088209 2.8272543712414024;
            0.0 -1.0 1.0 0.0 0.0 0.0 -0.0 -3.0680353217581158 -0.7575605255714953;
            0.0 0.0 0.0 -1.0 1.0 0.0 -0.14104621525160543 0.0 -2.8272543712414024;
            0.0 0.0 0.0 0.0 -1.0 1.0 1.0713523695816984 -0.0 0.7575605255714953;;;
            -1.0 1.0 0.0 0.0 0.0 0.0 -0.0 0.45018666133564855 0.31833004103016876;
            0.0 -1.0 1.0 0.0 0.0 0.0 -0.0 3.419507184334259 -1.1880238867000752;
            0.0 0.0 0.0 -1.0 1.0 0.0 2.1895743526754634 0.0 -0.31833004103016876;
            0.0 0.0 0.0 0.0 -1.0 1.0 1.6801194929944439 0.0 1.1880238867000752;;;
            -1.0 1.0 0.0 0.0 0.0 0.0 -0.0 1.680119492994444 1.1880238867000754;
            0.0 -1.0 1.0 0.0 0.0 0.0 -0.0 2.1895743526754634 -0.3183300410301689;
            0.0 0.0 0.0 -1.0 1.0 0.0 3.4195071843342584 0.0 -1.1880238867000754;
            0.0 0.0 0.0 0.0 -1.0 1.0 0.4501866613356489 0.0 0.3183300410301689
        ]
        FEBasis_∇_BR.citem[] = 2
        allocs = @allocated update_basis!(FEBasis_∇_BR)
        @test FEBasis_∇_BR.cvals ≈
            [
            1.0 -1.0 0.0 0.0 0.0 0.0 -0.0 1.0713523695816984 -0.7575605255714951;
            0.0 1.0 -1.0 0.0 0.0 0.0 0.0 -0.14104621525160543 2.8272543712414024;
            0.0 0.0 0.0 1.0 -1.0 0.0 -3.0680353217581158 -0.0 0.7575605255714951;
            0.0 0.0 0.0 0.0 1.0 -1.0 3.998341476088209 0.0 -2.8272543712414024;;;
            1.0 -1.0 0.0 0.0 0.0 0.0 -0.0 3.998341476088209 -2.8272543712414024;
            0.0 1.0 -1.0 0.0 0.0 0.0 0.0 -3.0680353217581158 0.7575605255714953;
            0.0 0.0 0.0 1.0 -1.0 0.0 -0.14104621525160543 -0.0 2.8272543712414024;
            0.0 0.0 0.0 0.0 1.0 -1.0 1.0713523695816984 0.0 -0.7575605255714953;;;
            1.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.45018666133564855 -0.31833004103016876;
            0.0 1.0 -1.0 0.0 0.0 0.0 0.0 3.419507184334259 1.1880238867000752;
            0.0 0.0 0.0 1.0 -1.0 0.0 2.1895743526754634 -0.0 0.31833004103016876;
            0.0 0.0 0.0 0.0 1.0 -1.0 1.6801194929944439 -0.0 -1.1880238867000752;;;
            1.0 -1.0 0.0 0.0 0.0 0.0 0.0 1.680119492994444 -1.1880238867000754;
            0.0 1.0 -1.0 0.0 0.0 0.0 0.0 2.1895743526754634 0.3183300410301689;
            0.0 0.0 0.0 1.0 -1.0 0.0 3.4195071843342584 -0.0 1.1880238867000754;
            0.0 0.0 0.0 0.0 1.0 -1.0 0.4501866613356489 -0.0 -0.3183300410301689
        ]
        @test allocs == 0
    end
end
