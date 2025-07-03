using Random
using Statistics
using YAML
using CairoMakie

abstract type BanditEnvironment end

struct GaussianBandit <: BanditEnvironment
    means::Vector{Float64}
    stds::Vector{Float64}
    k::Int
    
    function GaussianBandit(means::Vector{Float64}, stds::Vector{Float64})
        @assert length(means) == length(stds) "means and stds must have the same length"
        new(means, stds, length(means))
    end
end

function pull_arm(bandit::GaussianBandit, arm::Int)
    @assert 1 <= arm <= bandit.k "arm must be between 1 and $(bandit.k)"
    return randn() * bandit.stds[arm] + bandit.means[arm]
end

function get_optimal_arm(bandit::GaussianBandit)
    return argmax(bandit.means)
end

function get_optimal_value(bandit::GaussianBandit)
    return maximum(bandit.means)
end

abstract type BanditAlgorithm end

mutable struct EpsilonGreedy <: BanditAlgorithm
    epsilon::Float64
    q_values::Vector{Float64}
    action_counts::Vector{Int}
    k::Int
    
    function EpsilonGreedy(epsilon::Float64, k::Int)
        new(epsilon, zeros(k), zeros(Int, k), k)
    end
end

function select_action(alg::EpsilonGreedy)
    if rand() < alg.epsilon
        return rand(1:alg.k)
    else
        return argmax(alg.q_values)
    end
end

function update!(alg::EpsilonGreedy, action::Int, reward::Float64)
    alg.action_counts[action] += 1
    alg.q_values[action] += (reward - alg.q_values[action]) / alg.action_counts[action]
end

mutable struct UCB <: BanditAlgorithm
    c::Float64
    q_values::Vector{Float64}
    action_counts::Vector{Int}
    k::Int
    t::Int
    
    function UCB(c::Float64, k::Int)
        new(c, zeros(k), zeros(Int, k), k, 0)
    end
end

function select_action(alg::UCB)
    alg.t += 1
    
    if any(alg.action_counts .== 0)
        return findfirst(alg.action_counts .== 0)
    end
    
    ucb_values = alg.q_values .+ alg.c .* sqrt.(log(alg.t) ./ alg.action_counts)
    return argmax(ucb_values)
end

function update!(alg::UCB, action::Int, reward::Float64)
    alg.action_counts[action] += 1
    alg.q_values[action] += (reward - alg.q_values[action]) / alg.action_counts[action]
end

mutable struct SimulationResult
    rewards::Vector{Float64}
    cumulative_rewards::Vector{Float64}
    regrets::Vector{Float64}
    cumulative_regrets::Vector{Float64}
    optimal_action_percentage::Vector{Float64}
    
    function SimulationResult(steps::Int)
        new(zeros(steps), zeros(steps), zeros(steps), zeros(steps), zeros(steps))
    end
end

function run_simulation(bandit::BanditEnvironment, algorithm::BanditAlgorithm, steps::Int; seed::Union{Int, Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end
    
    result = SimulationResult(steps)
    optimal_arm = get_optimal_arm(bandit)
    optimal_value = get_optimal_value(bandit)
    
    optimal_actions = 0
    
    for t in 1:steps
        action = select_action(algorithm)
        reward = pull_arm(bandit, action)
        update!(algorithm, action, reward)
        
        result.rewards[t] = reward
        result.cumulative_rewards[t] = t == 1 ? reward : result.cumulative_rewards[t-1] + reward
        
        regret = optimal_value - reward
        result.regrets[t] = regret
        result.cumulative_regrets[t] = t == 1 ? regret : result.cumulative_regrets[t-1] + regret
        
        if action == optimal_arm
            optimal_actions += 1
        end
        result.optimal_action_percentage[t] = optimal_actions / t * 100
    end
    
    return result
end

function load_config(config_path::String)
    return YAML.load_file(config_path)
end

function create_bandit_from_config(config::Dict)
    bandit_type = get(config, "type", "gaussian")
    
    if bandit_type == "gaussian"
        means = config["means"]
        stds = get(config, "stds", ones(length(means)))
        return GaussianBandit(means, stds)
    else
        error("Unknown bandit type: $bandit_type")
    end
end

function create_algorithm_from_config(config::Dict, k::Int)
    alg_type = config["type"]
    
    if alg_type == "epsilon_greedy"
        epsilon = config["epsilon"]
        return EpsilonGreedy(epsilon, k)
    elseif alg_type == "ucb"
        c = config["c"]
        return UCB(c, k)
    else
        error("Unknown algorithm type: $alg_type")
    end
end

function plot_results(results::Vector{Tuple{String, SimulationResult}}, steps::Int)
    fig = Figure(size = (1200, 800))
    
    ax1 = Axis(fig[1, 1], title="Cumulative Rewards", xlabel="Steps", ylabel="Cumulative Reward")
    ax2 = Axis(fig[1, 2], title="Cumulative Regrets", xlabel="Steps", ylabel="Cumulative Regret")
    ax3 = Axis(fig[2, 1], title="Average Rewards", xlabel="Steps", ylabel="Average Reward")
    ax4 = Axis(fig[2, 2], title="Optimal Action Percentage", xlabel="Steps", ylabel="Optimal Action %")
    
    for (name, result) in results
        lines!(ax1, 1:steps, result.cumulative_rewards, label=name)
        lines!(ax2, 1:steps, result.cumulative_regrets, label=name)
        lines!(ax3, 1:steps, result.cumulative_rewards ./ (1:steps), label=name)
        lines!(ax4, 1:steps, result.optimal_action_percentage, label=name)
    end
    
    axislegend(ax1, position=:rb)
    axislegend(ax2, position=:rb)
    axislegend(ax3, position=:rb)
    axislegend(ax4, position=:rb)
    
    return fig
end

function run_experiment(config_path::String)
    config = load_config(config_path)
    
    bandit = create_bandit_from_config(config["environment"])
    steps = config["simulation"]["steps"]
    runs = get(config["simulation"], "runs", 1)
    seed = get(config["simulation"], "seed", nothing)
    
    results = Tuple{String, SimulationResult}[]
    
    for alg_config in config["algorithms"]
        alg_name = alg_config["name"]
        println("Running experiment with algorithm: $alg_name")
        
        all_results = SimulationResult[]
        for run in 1:runs
            current_seed = seed !== nothing ? seed + run - 1 : nothing
            algorithm = create_algorithm_from_config(alg_config, bandit.k)
            result = run_simulation(bandit, algorithm, steps, seed=current_seed)
            push!(all_results, result)
        end
        
        averaged_result = average_results(all_results)
        push!(results, (alg_name, averaged_result))
    end
    
    return results, steps
end

function average_results(results::Vector{SimulationResult})
    if length(results) == 1
        return results[1]
    end
    
    steps = length(results[1].rewards)
    avg_result = SimulationResult(steps)
    
    for i in 1:steps
        avg_result.rewards[i] = mean([r.rewards[i] for r in results])
        avg_result.cumulative_rewards[i] = mean([r.cumulative_rewards[i] for r in results])
        avg_result.regrets[i] = mean([r.regrets[i] for r in results])
        avg_result.cumulative_regrets[i] = mean([r.cumulative_regrets[i] for r in results])
        avg_result.optimal_action_percentage[i] = mean([r.optimal_action_percentage[i] for r in results])
    end
    
    return avg_result
end

function run_multiple_experiments(config_paths::Vector{String})
    all_results = []
    
    for config_path in config_paths
        println("Running experiment: $config_path")
        results, steps = run_experiment(config_path)
        push!(all_results, (config_path, results, steps))
        
        fig = plot_results(results, steps)
        save("$(splitext(config_path)[1])_results.png", fig)
        println("Results saved to $(splitext(config_path)[1])_results.png")
    end
    
    return all_results
end