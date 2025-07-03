include("bandit.jl")
using Test

function test_gaussian_bandit()
    println("Testing GaussianBandit...")
    
    means = [0.0, 0.5, 1.0]
    stds = [0.1, 0.2, 0.3]
    bandit = GaussianBandit(means, stds)
    
    @test bandit.k == 3
    @test bandit.means == means
    @test bandit.stds == stds
    @test get_optimal_arm(bandit) == 3
    @test get_optimal_value(bandit) == 1.0
    
    Random.seed!(42)
    reward = pull_arm(bandit, 1)
    @test isa(reward, Float64)
    
    rewards = [pull_arm(bandit, 1) for _ in 1:1000]
    @test abs(mean(rewards) - means[1]) < 0.1
    @test abs(std(rewards) - stds[1]) < 0.1
    
    @test_throws AssertionError GaussianBandit([1.0, 2.0], [0.1])
    @test_throws AssertionError pull_arm(bandit, 0)
    @test_throws AssertionError pull_arm(bandit, 4)
    
    println("✓ GaussianBandit tests passed")
end

function test_epsilon_greedy()
    println("Testing EpsilonGreedy...")
    
    epsilon = 0.1
    k = 3
    alg = EpsilonGreedy(epsilon, k)
    
    @test alg.epsilon == epsilon
    @test alg.k == k
    @test length(alg.q_values) == k
    @test length(alg.action_counts) == k
    @test all(alg.q_values .== 0.0)
    @test all(alg.action_counts .== 0)
    
    Random.seed!(42)
    actions = [select_action(alg) for _ in 1:1000]
    @test all(1 .<= actions .<= k)
    
    update!(alg, 1, 1.0)
    @test alg.q_values[1] == 1.0
    @test alg.action_counts[1] == 1
    
    update!(alg, 1, 0.0)
    @test alg.q_values[1] == 0.5
    @test alg.action_counts[1] == 2
    
    println("✓ EpsilonGreedy tests passed")
end

function test_ucb()
    println("Testing UCB...")
    
    c = 2.0
    k = 3
    alg = UCB(c, k)
    
    @test alg.c == c
    @test alg.k == k
    @test alg.t == 0
    @test length(alg.q_values) == k
    @test length(alg.action_counts) == k
    @test all(alg.q_values .== 0.0)
    @test all(alg.action_counts .== 0)
    
    action1 = select_action(alg)
    @test 1 <= action1 <= k
    @test alg.t == 1
    
    update!(alg, action1, 1.0)
    @test alg.q_values[action1] == 1.0
    @test alg.action_counts[action1] == 1
    
    action2 = select_action(alg)
    @test 1 <= action2 <= k
    @test alg.t == 2
    
    println("✓ UCB tests passed")
end

function test_simulation_result()
    println("Testing SimulationResult...")
    
    steps = 100
    result = SimulationResult(steps)
    
    @test length(result.rewards) == steps
    @test length(result.cumulative_rewards) == steps
    @test length(result.regrets) == steps
    @test length(result.cumulative_regrets) == steps
    @test length(result.optimal_action_percentage) == steps
    @test all(result.rewards .== 0.0)
    
    println("✓ SimulationResult tests passed")
end

function test_run_simulation()
    println("Testing run_simulation...")
    
    means = [0.0, 0.5, 1.0]
    stds = [0.1, 0.1, 0.1]
    bandit = GaussianBandit(means, stds)
    
    epsilon = 0.1
    algorithm = EpsilonGreedy(epsilon, bandit.k)
    
    steps = 100
    result = run_simulation(bandit, algorithm, steps, seed=42)
    
    @test length(result.rewards) == steps
    @test length(result.cumulative_rewards) == steps
    @test length(result.regrets) == steps
    @test length(result.cumulative_regrets) == steps
    @test length(result.optimal_action_percentage) == steps
    
    @test result.cumulative_rewards[1] == result.rewards[1]
    @test result.cumulative_rewards[end] ≈ sum(result.rewards)
    @test result.cumulative_regrets[1] == result.regrets[1]
    @test result.cumulative_regrets[end] ≈ sum(result.regrets)
    
    @test all(0 .<= result.optimal_action_percentage .<= 100)
    
    println("✓ run_simulation tests passed")
end

function test_yaml_config()
    println("Testing YAML configuration...")
    
    test_config = """
environment:
  type: gaussian
  means: [0.1, 0.2, 0.3]
  stds: [1.0, 1.0, 1.0]

simulation:
  steps: 100
  runs: 2
  seed: 42

algorithms:
  - name: "test_epsilon_greedy"
    type: epsilon_greedy
    epsilon: 0.1
  - name: "test_ucb"
    type: ucb
    c: 2.0
"""
    
    open("test_config.yaml", "w") do f
        write(f, test_config)
    end
    
    config = load_config("test_config.yaml")
    @test config["environment"]["type"] == "gaussian"
    @test config["environment"]["means"] == [0.1, 0.2, 0.3]
    @test config["simulation"]["steps"] == 100
    @test length(config["algorithms"]) == 2
    
    bandit = create_bandit_from_config(config["environment"])
    @test isa(bandit, GaussianBandit)
    @test bandit.k == 3
    @test bandit.means == [0.1, 0.2, 0.3]
    
    alg1 = create_algorithm_from_config(config["algorithms"][1], bandit.k)
    @test isa(alg1, EpsilonGreedy)
    @test alg1.epsilon == 0.1
    
    alg2 = create_algorithm_from_config(config["algorithms"][2], bandit.k)
    @test isa(alg2, UCB)
    @test alg2.c == 2.0
    
    rm("test_config.yaml")
    
    println("✓ YAML configuration tests passed")
end

function test_average_results()
    println("Testing average_results...")
    
    steps = 10
    result1 = SimulationResult(steps)
    result2 = SimulationResult(steps)
    
    result1.rewards = collect(1:steps)
    result2.rewards = collect(steps:-1:1)
    
    result1.cumulative_rewards = cumsum(result1.rewards)
    result2.cumulative_rewards = cumsum(result2.rewards)
    
    result1.optimal_action_percentage = collect(1:steps) .* 10
    result2.optimal_action_percentage = collect(steps:-1:1) .* 10
    
    averaged = average_results([result1, result2])
    
    expected_rewards = (collect(1:steps) + collect(steps:-1:1)) / 2
    @test averaged.rewards ≈ expected_rewards
    
    expected_percentage = (collect(1:steps) .* 10 + collect(steps:-1:1) .* 10) / 2
    @test averaged.optimal_action_percentage ≈ expected_percentage
    
    single_result = average_results([result1])
    @test single_result === result1
    
    println("✓ average_results tests passed")
end

function test_experiment_integration()
    println("Testing experiment integration...")
    
    test_config = """
environment:
  type: gaussian
  means: [0.1, 0.9]
  stds: [0.5, 0.5]

simulation:
  steps: 50
  runs: 2
  seed: 123

algorithms:
  - name: "test_epsilon"
    type: epsilon_greedy
    epsilon: 0.1
  - name: "test_ucb"
    type: ucb
    c: 1.0
"""
    
    open("test_experiment.yaml", "w") do f
        write(f, test_config)
    end
    
    results, steps = run_experiment("test_experiment.yaml")
    
    @test length(results) == 2
    @test steps == 50
    
    for (name, result) in results
        @test isa(result, SimulationResult)
        @test length(result.rewards) == steps
        @test length(result.cumulative_rewards) == steps
        @test length(result.regrets) == steps
        @test length(result.cumulative_regrets) == steps
        @test length(result.optimal_action_percentage) == steps
    end
    
    rm("test_experiment.yaml")
    
    println("✓ experiment integration tests passed")
end

function test_error_handling()
    println("Testing error handling...")
    
    @test_throws AssertionError GaussianBandit([1.0, 2.0], [0.1])
    @test_throws AssertionError pull_arm(GaussianBandit([1.0], [0.1]), 2)
    
    invalid_config = Dict("type" => "unknown")
    @test_throws ErrorException create_bandit_from_config(invalid_config)
    
    invalid_alg_config = Dict("type" => "unknown")
    @test_throws ErrorException create_algorithm_from_config(invalid_alg_config, 3)
    
    println("✓ error handling tests passed")
end

function run_all_tests()
    println("Running comprehensive tests for bandit simulation...")
    println("=" ^ 60)
    
    test_gaussian_bandit()
    test_epsilon_greedy()
    test_ucb()
    test_simulation_result()
    test_run_simulation()
    test_yaml_config()
    test_average_results()
    test_experiment_integration()
    test_error_handling()
    
    println("=" ^ 60)
    println("All tests passed successfully! ✅")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
end