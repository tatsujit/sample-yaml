include("bandit.jl")

function main()
    config_files = [
        "experiment1.yaml",
        "experiment2.yaml"
    ]
    
    println("バンディット問題シミュレーション開始")
    println("=" ^ 50)
    
    all_results = run_multiple_experiments(config_files)
    
    println("=" ^ 50)
    println("全ての実験が完了しました！")
    println("結果は各実験設定ファイル名_results.png として保存されています。")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end