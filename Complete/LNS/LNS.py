import os
import time
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import config
from ortools.sat.python import cp_model

# ==========================================
# 0. CONFIGURATION
# ==========================================

INSTANCE_DIR = config.INSTANCE_DIR
OUTPUT_DIR = "../plot/DecisionRepair"  # Nouveau dossier
INSTANCES = ["ft06", "ft10", "abz6", "la27", "ta31"]
TIME_LIMIT = 300.0
LNS_SUB_TIME_LIMIT = 1.0  # Temps max par "réparation" (très court !)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# ==========================================
# 1. PARSER ROBUSTE
# ==========================================
def parse_instance(file_path):
    jobs_data = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        iterator = iter(lines)
        num_jobs = 0
        num_machines = 0
        found_header = False

        for line in iterator:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('#'): continue
            parts = clean_line.split()
            if len(parts) >= 2:
                try:
                    num_jobs = int(parts[0])
                    num_machines = int(parts[1])
                    found_header = True
                    break
                except ValueError:
                    continue

        if not found_header: return None, None

        for line in iterator:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('#'): continue
            try:
                parts = list(map(int, clean_line.split()))
                job = []
                for i in range(0, len(parts), 2):
                    m = parts[i]
                    d = parts[i + 1]
                    job.append((m, d))
                jobs_data.append(job)
            except ValueError:
                continue

        return jobs_data, num_machines
    except Exception as e:
        print(f"❌ Erreur lecture {file_path}: {e}")
        return None, None


# ==========================================
# 2. SOLVEUR TYPE "DECISION REPAIR" (LNS)
# ==========================================

def solve_subproblem(jobs_data, num_machines, fixed_jobs, previous_solution, horizon, time_limit):
    """
    Crée et résout un sous-problème où certains jobs sont figés.
    """
    model = cp_model.CpModel()
    start, end, interval = {}, {}, {}

    # Création des variables
    for j, job in enumerate(jobs_data):
        for t, (m, d) in enumerate(job):
            start[j, t] = model.NewIntVar(0, horizon, f"s_{j}_{t}")
            end[j, t] = model.NewIntVar(0, horizon, f"e_{j}_{t}")
            interval[j, t] = model.NewIntervalVar(start[j, t], d, end[j, t], f"int_{j}_{t}")

            # --- LA PARTIE REPAIR ---
            # Si ce job fait partie des "jobs figés", on impose la valeur précédente
            if j in fixed_jobs and previous_solution is not None:
                prev_start = previous_solution[(j, t)]
                model.Add(start[j, t] == prev_start)

    makespan = model.NewIntVar(0, horizon, "makespan")

    # Contraintes Précédence
    for j, job in enumerate(jobs_data):
        for t in range(len(job) - 1):
            model.Add(start[j, t + 1] >= end[j, t])
        model.Add(makespan >= end[j, len(job) - 1])

    # Contraintes NoOverlap (On utilise la puissance globale ici pour réparer vite)
    # Note: On pourrait aussi utiliser la version CycleCut, mais pour le LNS,
    # on veut de la vitesse pure, donc NoOverlap est mieux adapté.
    for m in range(num_machines):
        tasks_on_machine = []
        for j, job in enumerate(jobs_data):
            for t, (mach, d) in enumerate(job):
                if mach == m:
                    tasks_on_machine.append(interval[j, t])
        model.AddNoOverlap(tasks_on_machine)

    # Objectif
    model.Minimize(makespan)

    # Si on a une solution précédente, on l'utilise comme borne supérieure (upper bound)
    # On cherche strictement mieux, ou au moins aussi bien pour valider
    if previous_solution and "makespan" in previous_solution:
        model.Add(makespan <= previous_solution["makespan"])

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    #solver.parameters.num_workers = 1  # Single thread pour les petites itérations

    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        new_sol = {}
        for j, job in enumerate(jobs_data):
            for t, _ in enumerate(job):
                new_sol[(j, t)] = solver.Value(start[j, t])
        new_sol["makespan"] = solver.Value(makespan)
        return new_sol, status

    return None, status


def solve_jssp_decision_repair(jobs_data, num_machines, time_limit=30.0):
    start_time = time.time()
    horizon = sum(d for job in jobs_data for _, d in job)

    best_solution = None
    best_makespan = float('inf')

    # Stats
    iterations = 0
    improvements = 0

    # 1. SOLUTION INITIALE (Rapide)
    # On ne fixe rien (fixed_jobs=[]), on laisse le solveur trouver une première base
    print("   -> Recherche solution initiale...", end="")
    current_solution, status = solve_subproblem(jobs_data, num_machines, [], None, horizon, 2.0)

    if current_solution:
        best_solution = current_solution
        best_makespan = current_solution["makespan"]
        print(f" Trouvée (Makespan: {best_makespan})")
    else:
        print(" Echec solution initiale (Temps trop court ou infaisable).")
        return {"status": "UNKNOWN", "time": time.time() - start_time, "makespan": None, "schedule": []}

    # 2. BOUCLE DE REPAIR (LNS)
    while (time.time() - start_time) < time_limit:
        iterations += 1
        remaining_time = time_limit - (time.time() - start_time)

        # --- STRATEGIE DE VOISINAGE (NEIGHBORHOOD) ---
        # On choisit aléatoirement 50% des jobs à FIGER.
        # Les autres seront "libérés" et le solveur essaiera de les réarranger.
        all_jobs_indices = list(range(len(jobs_data)))
        num_to_fix = max(1, len(jobs_data) // 2)
        fixed_jobs = random.sample(all_jobs_indices, num_to_fix)

        # Résolution du sous-problème
        # On donne un temps très court (LNS_SUB_TIME_LIMIT) pour chaque réparation
        sol, stat = solve_subproblem(jobs_data, num_machines, fixed_jobs, best_solution, horizon, LNS_SUB_TIME_LIMIT)

        if sol:
            new_mksp = sol["makespan"]
            if new_mksp < best_makespan:
                print(f"   -> Amélioration itération {iterations}: {best_makespan} -> {new_mksp}")
                best_solution = sol
                best_makespan = new_mksp
                improvements += 1
            # Même si pas mieux, on pourrait accepter la solution pour diversifier (Recuit Simulé),
            # mais ici on fait un simple Hill Climbing (Descente pure).

    total_time = time.time() - start_time

    # Reconstruction du format pour le plot
    schedule = []
    if best_solution:
        for j, job in enumerate(jobs_data):
            for t, (m, d) in enumerate(job):
                s_val = best_solution[(j, t)]
                schedule.append({
                    "Job": j, "Task": t, "Machine": m,
                    "Start": s_val, "Duration": d, "Finish": s_val + d
                })

    return {
        "status": "LNS_OPTIMIZED",  # Statut custom
        "time": total_time,
        "branches": iterations,  # On compte les itérations LNS au lieu des branches
        "conflicts": improvements,  # On détourne ce champ pour compter les améliorations
        "makespan": best_makespan,
        "schedule": schedule
    }


# ==========================================
# 3. PLOTTING
# ==========================================
def plot_gantt(schedule, makespan, instance_name):
    if not schedule: return
    df = pd.DataFrame(schedule)
    machines = sorted(df['Machine'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(df['Job'].unique())))

    fig, ax = plt.subplots(figsize=(12, 6))
    for _, row in df.iterrows():
        m = row['Machine']
        start = row['Start']
        duration = row['Duration']
        job_id = row['Job']
        rect = patches.Rectangle((start, m - 0.4), duration, 0.8,
                                 edgecolor='black', facecolor=colors[job_id % len(colors)], alpha=0.8)
        ax.add_patch(rect)
        if duration > 2:
            ax.text(start + duration / 2, m, f"J{job_id}", ha='center', va='center', color='white', fontsize=8,
                    fontweight='bold')

    ax.set_yticks(machines)
    ax.set_yticklabels([f"M{m}" for m in machines])
    ax.set_xlabel("Time")
    ax.set_title(f"Gantt (LNS Repair) - {instance_name} (Makespan: {makespan})")
    ax.set_xlim(0, makespan * 1.05)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"gantt_{instance_name}.png"))
    plt.close()


def plot_stats(results_df):
    # Plot 1: Temps vs Makespan
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(len(results_df))
    ax1.set_xlabel('Instance')
    ax1.set_ylabel('Temps (s)', color='tab:blue')
    ax1.bar(x, results_df['time'], color='tab:blue', alpha=0.6, label='Temps Total')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['instance'], rotation=45)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Best Makespan', color='tab:red')
    ax2.plot(x, results_df['makespan'], color='tab:red', marker='o', linewidth=2, label='Makespan')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    plt.title("Performance LNS (Decision Repair)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stats_repair_performance.png"))
    plt.close()

    # Plot 2: Itérations vs Améliorations
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['branches'], results_df['conflicts'],
                s=100, c=results_df['time'], cmap='plasma', edgecolors='k')
    for i, txt in enumerate(results_df['instance']):
        plt.annotate(txt, (results_df['branches'][i], results_df['conflicts'][i]), xytext=(5, 5),
                     textcoords='offset points')
    plt.colorbar(label='Temps Total (s)')
    plt.xlabel('Nombre total de réparations tentées (Itérations)')
    plt.ylabel('Nombre d\'améliorations trouvées')
    plt.title("Efficacité du Repair (Itérations vs Succès)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stats_repair_efficiency.png"))
    plt.close()


# ==========================================
# 4. MAIN
# ==========================================
def main():
    print(f"--- Benchmark DECISION REPAIR (LNS) ---")
    print(f"Instances : {INSTANCES}")
    print(f"Time Limit : {TIME_LIMIT}s\n")

    summary_results = []

    for name in INSTANCES:
        file_path = os.path.join(INSTANCE_DIR, f"{name}.txt")
        if not os.path.exists(file_path):
            print(f"⚠️  Fichier {name}.txt introuvable.")
            continue

        print(f"Traitement de {name}...", end=" ", flush=True)
        jobs_data, num_machines = parse_instance(file_path)

        if not jobs_data: continue

        res = solve_jssp_decision_repair(jobs_data, num_machines, time_limit=TIME_LIMIT)

        mksp = res['makespan'] if res['makespan'] else "N/A"
        print(f" -> Final: {mksp} | Itérations: {res['branches']} | Améliorations: {res['conflicts']}")

        summary_results.append({
            "instance": name,
            "status": res['status'],
            "time": res['time'],
            "makespan": res['makespan'] if res['makespan'] else 0,
            "branches": res['branches'],  # Itérations LNS
            "conflicts": res['conflicts']  # Améliorations
        })

        if res['schedule']:
            plot_gantt(res['schedule'], res['makespan'], name)

    if summary_results:
        df = pd.DataFrame(summary_results)
        print("\n--- Résumé LNS ---")
        print(df[["instance", "makespan", "time", "branches", "conflicts"]])
        df.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"), index=False)
        plot_stats(df)
        print(f"\n✅ Résultats sauvegardés dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()