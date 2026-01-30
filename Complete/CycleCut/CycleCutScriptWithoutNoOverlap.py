import os
import time
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
# 0. CONFIGURATION & UTILITAIRES
# ==========================================

INSTANCE_DIR = config.INSTANCE_DIR
OUTPUT_DIR = "../plot/CycleCut"
INSTANCES = ["ft06", "ft10", "abz6", "la27", "ta31"]
TIME_LIMIT = 300.0

# Création du dossier de sortie
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def parse_instance(file_path):
    """
    Lit une instance JSSP en ignorant les commentaires (#) et les lignes vides.
    Cherche la première ligne valide contenant 'NbJobs NbMachines'.
    """
    jobs_data = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        iterator = iter(lines)
        num_jobs = 0
        num_machines = 0
        found_header = False

        # 1. Recherche de l'en-tête (Dimensions)
        # On ignore tout tant qu'on ne trouve pas deux entiers
        for line in iterator:
            clean_line = line.strip()
            # Ignorer lignes vides ou commentaires
            if not clean_line or clean_line.startswith('#'):
                continue

            parts = clean_line.split()
            # On tente de lire les dimensions
            if len(parts) >= 2:
                try:
                    num_jobs = int(parts[0])
                    num_machines = int(parts[1])
                    found_header = True
                    break
                except ValueError:
                    continue  # Ce n'était pas des entiers, on continue

        if not found_header:
            print(f"⚠️  Impossible de trouver les dimensions dans {file_path}")
            return None, None

        # 2. Lecture des Jobs
        for line in iterator:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('#'):
                continue

            try:
                parts = list(map(int, clean_line.split()))
                # Format: Machine Durée Machine Durée ...
                job = []
                for i in range(0, len(parts), 2):
                    m = parts[i]
                    d = parts[i + 1]
                    job.append((m, d))
                jobs_data.append(job)
            except ValueError:
                continue  # Ligne malformée, on ignore

        return jobs_data, num_machines

    except Exception as e:
        print(f"❌ Erreur lecture {file_path}: {e}")
        return None


# ==========================================
# 2. SOLVEUR "PUR" (SANS NoOverlap)
# ==========================================

def solve_jssp_cycle_cutset(jobs_data, num_machines, time_limit=30.0):
    horizon = sum(d for job in jobs_data for _, d in job)
    model = cp_model.CpModel()

    start, end = {}, {}
    tasks_per_machine = {m: [] for m in range(num_machines)}

    # Variables de Base
    for j, job in enumerate(jobs_data):
        for t, (m, d) in enumerate(job):
            start[j, t] = model.NewIntVar(0, horizon, f"s_{j}_{t}")
            end[j, t] = model.NewIntVar(0, horizon, f"e_{j}_{t}")

            # Note: Sans AddNoOverlap, 'NewIntervalVar' ne sert qu'à lier start/end/duration
            # On le garde pour la forme, mais il ne déclenche plus de propagation magique
            model.NewIntervalVar(start[j, t], d, end[j, t], f"int_{j}_{t}")

            tasks_per_machine[m].append((j, t))

    makespan = model.NewIntVar(0, horizon, "makespan")

    # Contraintes de Précédence Jobs (Classique)
    for j, job in enumerate(jobs_data):
        for t in range(len(job) - 1):
            model.Add(start[j, t + 1] >= end[j, t])
        model.Add(makespan >= end[j, len(job) - 1])

    # --- Approche CYCLE-CUTSET PURE (Sans Filet) ---
    cutset_vars = []

    for m in range(num_machines):
        tasks = tasks_per_machine[m]

        # ⚠️ SUPPRESSION DE L'AIDE OR-TOOLS
        # model.AddNoOverlap(...)  <-- ON L'ENLÈVE ICI

        # On doit gérer la disjonction manuellement à 100%
        for i in range(len(tasks)):
            for k in range(i + 1, len(tasks)):
                j1, t1 = tasks[i]
                j2, t2 = tasks[k]

                # Le booléen qui décide qui passe avant qui
                prec_bool = model.NewBoolVar(f"prec_m{m}_{j1}_{j2}")
                cutset_vars.append(prec_bool)

                # Si prec_bool est Vrai => Tâche 1 finit avant Tâche 2
                model.Add(end[j1, t1] <= start[j2, t2]).OnlyEnforceIf(prec_bool)

                # Si prec_bool est Faux => Tâche 2 finit avant Tâche 1
                model.Add(end[j2, t2] <= start[j1, t1]).OnlyEnforceIf(prec_bool.Not())

    # Stratégie de Recherche Fixe
    # On oblige le solveur à fixer nos booléens en premier
    model.AddDecisionStrategy(cutset_vars, cp_model.CHOOSE_LOWEST_MIN, cp_model.SELECT_MIN_VALUE)

    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.search_branching = cp_model.FIXED_SEARCH  # Pas d'intelligence artificielle cachée

    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    # Extraction des résultats
    schedule = []
    final_makespan = None

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        final_makespan = solver.Value(makespan)
        for j, job in enumerate(jobs_data):
            for t, (m, d) in enumerate(job):
                s_val = solver.Value(start[j, t])
                schedule.append({
                    "Job": j, "Task": t, "Machine": m,
                    "Start": s_val, "Duration": d, "Finish": s_val + d
                })

    return {
        "status": solver.StatusName(status),
        "time": elapsed,
        "branches": solver.NumBranches(),
        "conflicts": solver.NumConflicts(),
        "makespan": final_makespan,
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
            ax.text(start + duration / 2, m, f"J{job_id}",
                    ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    ax.set_yticks(machines)
    ax.set_yticklabels([f"M{m}" for m in machines])
    ax.set_xlabel("Time")
    ax.set_title(f"Gantt - {instance_name} (Makespan: {makespan})")
    ax.set_xlim(0, makespan * 1.05)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"gantt_{instance_name}.png"))
    plt.close()


def plot_stats(results_df):
    # Plot 1: Temps & Makespan
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(len(results_df))
    ax1.set_xlabel('Instance')
    ax1.set_ylabel('Temps (s)', color='tab:blue')
    ax1.bar(x, results_df['time'], color='tab:blue', alpha=0.6, label='Temps')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['instance'], rotation=45)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Makespan', color='tab:red')
    ax2.plot(x, results_df['makespan'], color='tab:red', marker='o', linewidth=2, label='Makespan')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title("Performance: Temps vs Makespan (Sans aide OrTools)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stats_time_makespan.png"))
    plt.close()

    # Plot 2: Complexité
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['conflicts'], results_df['branches'],
                s=100, c=results_df['time'], cmap='viridis', edgecolors='k')

    for i, txt in enumerate(results_df['instance']):
        plt.annotate(txt, (results_df['conflicts'][i], results_df['branches'][i]),
                     xytext=(5, 5), textcoords='offset points')

    plt.colorbar(label='Temps (s)')
    plt.xlabel('Conflits')
    plt.ylabel('Branches')
    plt.title('Complexité de Recherche (Pure Cutset)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stats_search_complexity.png"))
    plt.close()


# ==========================================
# 4. MAIN
# ==========================================

def main():
    print(f"--- Benchmark Cycle Cutset (SANS NoOverlap) ---")
    print(f"Instances : {INSTANCES}")
    print(f"Time Limit : {TIME_LIMIT}s\n")

    summary_results = []

    for name in INSTANCES:
        file_path = os.path.join(INSTANCE_DIR, f"{name}.txt")

        if not os.path.exists(file_path):
            print(f"⚠️  Fichier {name}.txt introuvable. Passage.")
            continue

        print(f"Traitement de {name}...", end=" ", flush=True)

        jobs_data, num_machines = parse_instance(file_path)

        if not jobs_data:
            print("❌ Echec parsing.")
            continue

        res = solve_jssp_cycle_cutset(jobs_data, num_machines, time_limit=TIME_LIMIT)

        mksp = res['makespan'] if res['makespan'] else "N/A"
        print(f"[{res['status']}] Makespan: {mksp} | Temps: {res['time']:.2f}s | Branches: {res['branches']}")

        summary_results.append({
            "instance": name,
            "status": res['status'],
            "time": res['time'],
            "makespan": res['makespan'] if res['makespan'] else 0,
            "branches": res['branches'],
            "conflicts": res['conflicts']
        })

        if res['schedule']:
            plot_gantt(res['schedule'], res['makespan'], name)

    if summary_results:
        df = pd.DataFrame(summary_results)
        print("\n--- Résumé (Sans NoOverlap) ---")
        print(df[["instance", "status", "makespan", "time", "branches", "conflicts"]])
        df.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"), index=False)
        plot_stats(df)
        print(f"\n✅ Résultats sauvegardés dans {OUTPUT_DIR}")
    else:
        print("\nAucun résultat.")


if __name__ == "__main__":
    main()