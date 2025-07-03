using Distributions
using Iterators
using Random
using DataFrames
using CSV

abstract type ParameterSampler end

struct GridSampler <: ParameterSampler
    param_ranges::Dict{String, Vector}
end

struct DistributionSampler <: ParameterSampler
    param_distributions::Dict{String, Distribution}
    n_samples::Int
end

struct RandomSampler <: ParameterSampler
    param_ranges::Dict{String, Vector}
    n_samples::Int
end

struct LatinHypercubeSampler <: ParameterSampler
    param_ranges::Dict{String, Tuple{Float64, Float64}}
    n_samples::Int
end

function generate_parameter_combinations(sampler::GridSampler)
    param_names = collect(keys(sampler.param_ranges))
    param_values = collect(values(sampler.param_ranges))
    
    combinations = []
    for combo in Iterators.product(param_values...)
        param_dict = Dict(zip(param_names, combo))
        push!(combinations, param_dict)
    end
    
    return combinations
end

function generate_parameter_combinations(sampler::DistributionSampler)
    combinations = []
    for i in 1:sampler.n_samples
        param_dict = Dict{String, Any}()
        for (param_name, distribution) in sampler.param_distributions
            param_dict[param_name] = rand(distribution)
        end
        push!(combinations, param_dict)
    end
    
    return combinations
end

function generate_parameter_combinations(sampler::RandomSampler)
    combinations = []
    for i in 1:sampler.n_samples
        param_dict = Dict{String, Any}()
        for (param_name, values) in sampler.param_ranges
            param_dict[param_name] = rand(values)
        end
        push!(combinations, param_dict)
    end
    
    return combinations
end

function generate_parameter_combinations(sampler::LatinHypercubeSampler)
    param_names = collect(keys(sampler.param_ranges))
    n_params = length(param_names)
    
    combinations = []
    for i in 1:sampler.n_samples
        param_dict = Dict{String, Any}()
        for (j, param_name) in enumerate(param_names)
            lower, upper = sampler.param_ranges[param_name]
            lhs_value = lower + (upper - lower) * (i - 1 + rand()) / sampler.n_samples
            param_dict[param_name] = lhs_value
        end
        push!(combinations, param_dict)
    end
    
    return combinations
end

struct ExperimentDesign
    environments::Vector{Dict{String, Any}}
    algorithms::Vector{Dict{String, Any}}
    simulation_settings::Dict{String, Any}
    parameter_sampler::Union{ParameterSampler, Nothing}
    cross_product::Bool
    
    function ExperimentDesign(environments, algorithms, simulation_settings; 
                            parameter_sampler=nothing, cross_product=true)
        new(environments, algorithms, simulation_settings, parameter_sampler, cross_product)
    end
end

function generate_experiment_grid(design::ExperimentDesign)
    experiments = []
    
    if design.parameter_sampler !== nothing
        parameter_combinations = generate_parameter_combinations(design.parameter_sampler)
        
        for param_combo in parameter_combinations
            for env_config in design.environments
                for alg_config in design.algorithms
                    experiment = Dict{String, Any}()
                    
                    experiment["environment"] = merge(env_config, 
                        get(param_combo, "environment", Dict()))
                    
                    experiment["algorithm"] = merge(alg_config, 
                        get(param_combo, "algorithm", Dict()))
                    
                    experiment["simulation"] = merge(design.simulation_settings, 
                        get(param_combo, "simulation", Dict()))
                    
                    experiment["parameters"] = param_combo
                    
                    push!(experiments, experiment)
                end
            end
        end
    else
        if design.cross_product
            for env_config in design.environments
                for alg_config in design.algorithms
                    experiment = Dict{String, Any}()
                    experiment["environment"] = env_config
                    experiment["algorithm"] = alg_config
                    experiment["simulation"] = design.simulation_settings
                    experiment["parameters"] = Dict()
                    push!(experiments, experiment)
                end
            end
        else
            min_length = min(length(design.environments), length(design.algorithms))
            for i in 1:min_length
                experiment = Dict{String, Any}()
                experiment["environment"] = design.environments[i]
                experiment["algorithm"] = design.algorithms[i]
                experiment["simulation"] = design.simulation_settings
                experiment["parameters"] = Dict()
                push!(experiments, experiment)
            end
        end
    end
    
    return experiments
end

function run_experiment_grid(experiments::Vector{Dict{String, Any}})
    results = []
    
    for (i, experiment) in enumerate(experiments)
        println("Running experiment $i/$(length(experiments))")
        
        try
            bandit = create_bandit_from_config(experiment["environment"])
            algorithm = create_algorithm_from_config(experiment["algorithm"], bandit.k)
            
            steps = experiment["simulation"]["steps"]
            runs = get(experiment["simulation"], "runs", 1)
            seed = get(experiment["simulation"], "seed", nothing)
            
            experiment_results = []
            for run in 1:runs
                current_seed = seed !== nothing ? seed + run - 1 : nothing
                result = run_simulation(bandit, algorithm, steps, seed=current_seed)
                push!(experiment_results, result)
            end
            
            averaged_result = average_results(experiment_results)
            
            result_record = Dict{String, Any}()
            result_record["experiment_id"] = i
            result_record["environment"] = experiment["environment"]
            result_record["algorithm"] = experiment["algorithm"]
            result_record["simulation"] = experiment["simulation"]
            result_record["parameters"] = experiment["parameters"]
            result_record["result"] = averaged_result
            result_record["final_reward"] = averaged_result.cumulative_rewards[end]
            result_record["final_regret"] = averaged_result.cumulative_regrets[end]
            result_record["final_optimal_percentage"] = averaged_result.optimal_action_percentage[end]
            
            push!(results, result_record)
            
        catch e
            println("Error in experiment $i: $e")
            continue
        end
    end
    
    return results
end

function save_experiment_results(results::Vector{Dict{String, Any}}, filename::String)
    df_data = []
    
    for result in results
        row = Dict{String, Any}()
        row["experiment_id"] = result["experiment_id"]
        
        for (key, value) in result["environment"]
            row["env_$key"] = value
        end
        
        for (key, value) in result["algorithm"]
            row["alg_$key"] = value
        end
        
        for (key, value) in result["simulation"]
            row["sim_$key"] = value
        end
        
        for (key, value) in result["parameters"]
            row["param_$key"] = value
        end
        
        row["final_reward"] = result["final_reward"]
        row["final_regret"] = result["final_regret"]
        row["final_optimal_percentage"] = result["final_optimal_percentage"]
        
        push!(df_data, row)
    end
    
    df = DataFrame(df_data)
    CSV.write(filename, df)
    
    return df
end

function create_parameter_study_yaml(filename::String)
    parameter_study_config = """
parameter_study:
  type: "grid_search"  # or "distribution_sampling", "random_sampling", "latin_hypercube"
  
  # Grid search example
  grid_parameters:
    algorithm:
      epsilon: [0.01, 0.05, 0.1, 0.2]
      c: [0.5, 1.0, 2.0, 4.0]
    environment:
      noise_std: [0.1, 0.5, 1.0]
    simulation:
      steps: [500, 1000, 2000]
  
  # Distribution sampling example
  distribution_parameters:
    n_samples: 100
    distributions:
      algorithm:
        epsilon: 
          type: "Beta"
          alpha: 1.0
          beta: 9.0  # 平均0.1のBeta分布
        c:
          type: "Gamma"
          shape: 2.0
          scale: 1.0
      environment:
        noise_std:
          type: "LogNormal"
          mu: 0.0
          sigma: 0.5
  
  # Random sampling example
  random_parameters:
    n_samples: 50
    ranges:
      algorithm:
        epsilon: [0.01, 0.05, 0.1, 0.15, 0.2]
        c: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
      environment:
        noise_std: [0.1, 0.3, 0.5, 0.7, 1.0]

environments:
  - name: "easy_bandit"
    type: "gaussian"
    means: [0.1, 0.2, 0.3, 0.4, 0.5]
    stds: [1.0, 1.0, 1.0, 1.0, 1.0]
  
  - name: "hard_bandit"
    type: "gaussian"
    means: [0.45, 0.5, 0.55]
    stds: [1.0, 1.0, 1.0]
    
  - name: "sparse_bandit"
    type: "gaussian"
    means: [0.0, 0.0, 0.0, 0.0, 1.0]
    stds: [0.5, 0.5, 0.5, 0.5, 0.5]

algorithms:
  - name: "epsilon_greedy"
    type: "epsilon_greedy"
    epsilon: 0.1  # デフォルト値（パラメータ探索で上書きされる）
    
  - name: "ucb"
    type: "ucb"
    c: 2.0  # デフォルト値（パラメータ探索で上書きされる）

simulation:
  steps: 1000
  runs: 10
  seed: 42
"""
    
    open(filename, "w") do f
        write(f, parameter_study_config)
    end
end

function load_parameter_study_config(filename::String)
    config = load_config(filename)
    
    study_type = config["parameter_study"]["type"]
    
    if study_type == "grid_search"
        grid_params = config["parameter_study"]["grid_parameters"]
        
        all_param_ranges = Dict{String, Vector}()
        for (category, params) in grid_params
            for (param_name, values) in params
                all_param_ranges["$(category)_$(param_name)"] = values
            end
        end
        
        sampler = GridSampler(all_param_ranges)
        
    elseif study_type == "distribution_sampling"
        dist_params = config["parameter_study"]["distribution_parameters"]
        n_samples = dist_params["n_samples"]
        
        all_distributions = Dict{String, Distribution}()
        for (category, params) in dist_params["distributions"]
            for (param_name, dist_config) in params
                dist_type = dist_config["type"]
                if dist_type == "Beta"
                    dist = Beta(dist_config["alpha"], dist_config["beta"])
                elseif dist_type == "Gamma"
                    dist = Gamma(dist_config["shape"], dist_config["scale"])
                elseif dist_type == "LogNormal"
                    dist = LogNormal(dist_config["mu"], dist_config["sigma"])
                else
                    error("Unknown distribution type: $dist_type")
                end
                all_distributions["$(category)_$(param_name)"] = dist
            end
        end
        
        sampler = DistributionSampler(all_distributions, n_samples)
        
    elseif study_type == "random_sampling"
        random_params = config["parameter_study"]["random_parameters"]
        n_samples = random_params["n_samples"]
        
        all_param_ranges = Dict{String, Vector}()
        for (category, params) in random_params["ranges"]
            for (param_name, values) in params
                all_param_ranges["$(category)_$(param_name)"] = values
            end
        end
        
        sampler = RandomSampler(all_param_ranges, n_samples)
        
    else
        error("Unknown parameter study type: $study_type")
    end
    
    design = ExperimentDesign(
        config["environments"],
        config["algorithms"],
        config["simulation"],
        parameter_sampler=sampler,
        cross_product=true
    )
    
    return design
end

function run_parameter_study(config_file::String)
    println("Loading parameter study configuration...")
    design = load_parameter_study_config(config_file)
    
    println("Generating experiment grid...")
    experiments = generate_experiment_grid(design)
    println("Generated $(length(experiments)) experiments")
    
    println("Running experiments...")
    results = run_experiment_grid(experiments)
    
    println("Saving results...")
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    results_file = "parameter_study_results_$timestamp.csv"
    df = save_experiment_results(results, results_file)
    
    println("Results saved to: $results_file")
    println("Total experiments completed: $(length(results))")
    
    return results, df
end