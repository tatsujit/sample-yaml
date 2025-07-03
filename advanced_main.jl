include("bandit.jl")
include("advanced_experiment_design.jl")

using Dates

function main()
    println("高度なパラメータ探索システム")
    println("=" ^ 50)
    
    # パラメータ探索の例
    examples = [
        ("parameter_study_example.yaml", "グリッドサーチの例"),
        ("distribution_study_example.yaml", "分布サンプリングの例")
    ]
    
    for (config_file, description) in examples
        if isfile(config_file)
            println("\n実行中: $description")
            println("設定ファイル: $config_file")
            
            try
                results, df = run_parameter_study(config_file)
                
                println("実験完了: $(length(results)) 件の実験を実行")
                println("結果の統計:")
                println("  最高最終報酬: $(maximum([r["final_reward"] for r in results]))")
                println("  最低最終後悔: $(minimum([r["final_regret"] for r in results]))")
                
            catch e
                println("エラーが発生しました: $e")
            end
        else
            println("設定ファイルが見つかりません: $config_file")
        end
    end
    
    println("\n" * "=" ^ 50)
    println("パラメータ探索完了")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end