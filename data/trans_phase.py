import os
import math
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model

# -------------------------
# Parser JSSP
# -------------------------
def parse_jssp_file(filename):
    jobs_data = []
    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    num_jobs, num_machines = map(int, lines[0].split())
    for i in range(1, num_jobs + 1):
        data = list(map(int, lines[i].split()))
        job = []
        for k in range(0, len(data), 2):
            machine = data[k]
            duration = data[k + 1]
            job.append((machine, duration))
        jobs_data.append(job)
    return jobs_data, num_jobs, num_machines

# -------------------------
# Lower bound
# -------------------------
def compute_lower_bound(jobs_data, num_machines):
    lb_job = max(sum(d for _, d in job) for job in jobs_data)
    machine_load = [0] * num_machines
    for job in jobs_data:
        for m, d in job:
            machine_load[m] += d
    lb_machine = max(machine_load)
    return max(lb_job, lb_machine)

# -------------------------
# Résolution de décision
# -------------------------
def solve_jssp_decision(jobs_data, num_machines, T, time_limit=2.0, random_seed=None):
    horizon = sum(d for job in jobs_data for _, d in job)
    model = cp_model.CpModel()
    start, end, interval = {}, {}, {}

    for j, job in enumerate(jobs_data):
        for t, (m, d) in enumerate(job):
            start[j,t] = model.NewIntVar(0, horizon, f"s_{j}_{t}")
            end[j,t]   = model.NewIntVar(0, horizon, f"e_{j}_{t}")
            interval[j,t] = model.NewIntervalVar(start[j,t], d, end[j,t], f"int_{j}_{t}")

    # Ordre des tâches
    for j, job in enumerate(jobs_data):
        for t in range(len(job)-1):
            model.Add(start[j,t+1] >= end[j,t])

    # Contrainte machine
    for m in range(num_machines):
        intervals = []
        for j, job in enumerate(jobs_data):
            for t, (mach, _) in enumerate(job):
                if mach == m:
                    intervals.append(interval[j,t])
        model.AddNoOverlap(intervals)

    # Makespan ≤ T
    for j, job in enumerate(jobs_data):
        model.Add(end[j, len(job)-1] <= T)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    if random_seed is not None:
        solver.parameters.random_seed = random_seed
    solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH

    status = solver.Solve(model)

    return {
        "feasible": status in (cp_model.OPTIMAL, cp_model.FEASIBLE),
        "branches": solver.NumBranches()
    }

# -------------------------
# Analyse probabiliste et graphes séparés
# -------------------------
def analyze_probabilistic_separate(folder_path, repetitions=5):
    selected_files = ["abz5.txt", "ft06.txt", "la07.txt", "la20.txt", "yn1.txt"]
    alphas = [1.00 + i*0.02 for i in range(40)]  # 1.00 → 1.40

    # Stockage résultats pour toutes les instances
    prob_feas_all = {}
    mean_branches_all = {}

    for file in selected_files:
        filepath = os.path.join(folder_path, file)
        jobs_data, num_jobs, num_machines = parse_jssp_file(filepath)
        LB = compute_lower_bound(jobs_data, num_machines)

        prob_feas = []
        mean_branches = []

        print(f"\n--- Instance {file} --- LB = {LB}")

        for alpha in alphas:
            T = math.ceil(alpha * LB)
            feas_count = 0
            branches_list = []

            for rep in range(repetitions):
                res = solve_jssp_decision(jobs_data, num_machines, T, time_limit=2.0, random_seed=rep)
                if res["feasible"]:
                    feas_count += 1
                branches_list.append(res["branches"])

            prob_feas.append(feas_count / repetitions)
            mean_branches.append(sum(branches_list)/repetitions)
            print(f"α={alpha:.2f} | Prob_feasible={feas_count}/{repetitions} | Mean branches={mean_branches[-1]:.1f}")

        prob_feas_all[file] = prob_feas
        mean_branches_all[file] = mean_branches

    # -------------------------
    # Graphe probabilité de faisabilité (points uniquement)
    # -------------------------
    plt.figure(figsize=(10,5))
    markers = ['o', 's', 'D', '^', 'v']  # markers différents pour chaque instance

    for i, file in enumerate(selected_files):
        plt.scatter(alphas, prob_feas_all[file], marker=markers[i], label=file)
    plt.xlabel("α = T / LB")
    plt.ylabel("Probabilité de faisabilité")
    plt.title("Probabilité de faisabilité - Toutes instances")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()

    # -------------------------
    # Graphe branches moyennes (points uniquement)
    # -------------------------
    plt.figure(figsize=(10,5))
    for i, file in enumerate(selected_files):
        plt.scatter(alphas, mean_branches_all[file], marker=markers[i], label=file)
    plt.xlabel("α = T / LB")
    plt.ylabel("Branches moyennes")
    plt.title("Branches moyennes - Toutes instances")
    plt.grid(True)
    plt.legend()
    plt.show()


# -------------------------
# Lancer l'analyse
# -------------------------
if __name__ == "__main__":
    folder_path = "instances_jsp"
    analyze_probabilistic_separate(folder_path, repetitions=5)
