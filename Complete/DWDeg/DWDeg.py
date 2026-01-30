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
OUTPUT_DIR = "../plot/DomOverWDeg"  # Dossier spécifique pour cette méthode
INSTANCES = ["ft06", "ft10", "abz6", "la27", "ta31"]  # Liste des instances à tester
TIME_LIMIT = 300.0  # Temps limite

# Création du dossier de sortie
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def parse_instance(file_path):
    """
    Lit une instance JSSP de manière robuste.
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
        for line in iterator:
            clean_line = line.strip()
            if not clean_line or clean_line.startswith('#'):
                continue

            parts = clean_line.split()
            if len(parts) >= 2:
                try:
                    num_jobs = int(parts[0])
                    num_machines = int(parts[1])
                    found_header = True
                    break
                except ValueError:
                    continue

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
                continue

        return jobs_data, num_machines

    except Exception as e:
        print(f"❌ Erreur lecture {file_path}: {e}")
        return None


# ==========================================
# 1. LE SOLVEUR (CP-SAT + Heuristique Custom)
# ==========================================

def solve_jssp_heuristic(jobs_data, num_machines, time_limit=30.0):
    """
    Résout le JSSP avec CP-SAT et une stratégie de recherche custom.
    """
    # -------------------------
    # Horizon
    # -------------------------
    horizon = sum(d for job in jobs_data for _, d in job)

    # -------------------------
    # Modèle CP-SAT
    # -------------------------
    model = cp_model.CpModel()
    start, end, interval = {}, {}, {}

    # -------------------------
    # Variables
    # -------------------------
    for j, job in enumerate(jobs_data):
        for t, (m, d) in enumerate(job):
            start[j, t] = model.NewIntVar(0, horizon, f"s_{j}_{t}")
            end[j, t] = model.NewIntVar(0, horizon, f"e_{j}_{t}")
            interval[j, t] = model.NewIntervalVar(
                start[j, t], d, end[j, t], f"int_{j}_{t}"
            )

    makespan = model.NewIntVar(0, horizon, "makespan")

    # -------------------------
    # Contraintes d’ordre (jobs)
    # -------------------------
    for j, job in enumerate(jobs_data):
        for t in range(len(job) - 1):
            model.Add(start[j, t + 1] >= end[j, t])
        # Lien avec le makespan
        model.Add(makespan >= end[j, len(job) - 1])

    # -------------------------
    # Contraintes machines (NoOverlap)
    # -------------------------
    for m in range(num_machines):
        machine_intervals = []
        for j, job in enumerate(jobs_data):
            for t, (machine, _) in enumerate(job):
                if machine == m:
                    machine_intervals.append(interval[j, t])
        model.AddNoOverlap(machine_intervals)

    # -------------------------
    # Heuristique DomOverWDeg-like (Code fourni)
    # -------------------------
    # On privilégie : Les tâches en fin de job, tâches longues, variables petit domaine
    critical_vars = []

    for j, job in enumerate(jobs_data):
        for t, (m, d) in enumerate(job):
            # Criticité = position + durée relative
            criticality = t + (d / float(horizon))  # Cast float pour sûreté
            critical_vars.append((criticality, start[j, t]))

    # Trier par criticité décroissante
    critical_vars.sort(reverse=True, key=lambda x: x[0])

    ordered_vars = [v for _, v in critical_vars]

    model.AddDecisionStrategy(
        ordered_vars,
        cp_model.CHOOSE_MIN_DOMAIN_SIZE,  # Dom
        cp_model.SELECT_MIN_VALUE  # Valeur la plus tôt possible
    )

    # -------------------------
    # Objectif
    # -------------------------
    model.Minimize(makespan)

    # -------------------------
    # Solveur & Paramètres
    # -------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.search_branching = cp_model.FIXED_SEARCH  # Important pour activer la stratégie custom
    solver.parameters.cp_model_presolve = False
    solver.parameters.num_search_workers = 1

    # -------------------------
    # Résolution
    # -------------------------
    t0 = time.time()
    status = solver.Solve(model)
    elapsed = time.time() - t0

    # Extraction Résultats pour Plotting
    schedule = []
    final_makespan = None

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        final_makespan = solver.Value(makespan)
        for j, job in enumerate(jobs_data):
            for t, (m, d) in enumerate(job):
                s_val = solver.Value(start[j, t])
                schedule.append({
                    "Job": j,
                    "Task": t,
                    "Machine": m,
                    "Start": s_val,
                    "Duration": d,
                    "Finish": s_val + d
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
# 2. FONCTIONS DE PLOT
# ==========================================

def plot_gantt(schedule, makespan, instance_name):
    """Génère un diagramme de Gantt (version corrigée avec cast int)."""
    if not schedule:
        return

    df = pd.DataFrame(schedule)

    # Cast explicite pour éviter les erreurs d'indexation numpy
    df['Job'] = df['Job'].astype(int)
    df['Start'] = df['Start'].astype(float)
    df['Duration'] = df['Duration'].astype(float)

    machines = sorted(df['Machine'].unique())
    unique_jobs = df['Job'].unique()
    nb_colors = len(unique_jobs)
    colors = plt.cm.tab20(np.linspace(0, 1, nb_colors))

    fig, ax = plt.subplots(figsize=(12, 6))

    for _, row in df.iterrows():
        m = row['Machine']
        start = row['Start']
        duration = row['Duration']
        job_id = int(row['Job'])  # Force int

        color_idx = job_id % len(colors)

        rect = patches.Rectangle((start, m - 0.4), duration, 0.8,
                                 edgecolor='black',
                                 facecolor=colors[color_idx],
                                 alpha=0.8)
        ax.add_patch(rect)

        if duration > 0:
            ax.text(start + duration / 2, m, f"J{job_id}",
                    ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    ax.set_yticks(machines)
    ax.set_yticklabels([f"Machine {m}" for m in machines])
    ax.set_xlabel("Temps")
    ax.set_title(f"Gantt Heuristic - {instance_name} (Makespan: {makespan})")

    if makespan:
        ax.set_xlim(0, makespan * 1.05)

    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"gantt_{instance_name}.png"))
    plt.close()


def plot_stats(results_df):
    """Génère les plots comparatifs"""
    if results_df.empty:
        return

    # 1. Makespan & Temps
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = range(len(results_df))
    ax1.set_xlabel('Instance')
    ax1.set_ylabel('Temps (s)', color='tab:blue')
    bars = ax1.bar(x, results_df['time'], color='tab:blue', alpha=0.6, label='Temps')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['instance'], rotation=45)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Makespan', color='tab:red')
    line = ax2.plot(x, results_df['makespan'], color='tab:red', marker='o', linewidth=2, label='Makespan')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title("Performance Heuristic: Temps vs Makespan")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stats_time_makespan.png"))
    plt.close()

    # 2. Complexity (Branches vs Conflicts)
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['conflicts'], results_df['branches'],
                s=100, c=results_df['time'], cmap='viridis', edgecolors='k')

    for i, txt in enumerate(results_df['instance']):
        plt.annotate(txt, (results_df['conflicts'][i], results_df['branches'][i]),
                     xytext=(5, 5), textcoords='offset points')

    plt.colorbar(label='Temps (s)')
    plt.xlabel('Conflits')
    plt.ylabel('Branches')
    plt.title('Complexité de Recherche (CP-SAT Custom Strategy)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stats_complexity.png"))
    plt.close()


# ==========================================
# 3. MAIN LOOP
# ==========================================

def main():
    print(f"--- Lancement Benchmark CP-SAT (Heuristique Custom) ---")
    print(f"Sortie des plots : {OUTPUT_DIR}")
    print(f"Time Limit : {TIME_LIMIT}s\n")

    summary_results = []

    for name in INSTANCES:
        file_path = os.path.join(INSTANCE_DIR, f"{name}.txt")

        if not os.path.exists(file_path):
            print(f"⚠️  Fichier {name}.txt introuvable dans {INSTANCE_DIR}. Passage.")
            continue

        print(f"Traitement de {name}...", end=" ", flush=True)

        jobs_data, num_machines = parse_instance(file_path)
        if not jobs_data:
            continue

        # Résolution
        res = solve_jssp_heuristic(jobs_data, num_machines, time_limit=TIME_LIMIT)

        mks_disp = res['makespan'] if res['makespan'] else "N/A"
        print(f"[{res['status']}] Makespan: {mks_disp} | Temps: {res['time']:.2f}s | Branches: {res['branches']}")

        # Sauvegarde stats
        summary_results.append({
            "instance": name,
            "status": res['status'],
            "time": res['time'],
            "makespan": res['makespan'] if res['makespan'] else 0,
            "branches": res['branches'],
            "conflicts": res['conflicts']
        })

        # Plot Gantt individuel
        if res['schedule']:
            plot_gantt(res['schedule'], res['makespan'], name)

    # Plot global
    if summary_results:
        df = pd.DataFrame(summary_results)
        print("\n--- Résumé ---")
        print(df[["instance", "status", "makespan", "time", "branches", "conflicts"]])

        # Sauvegarde CSV
        df.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"), index=False)

        plot_stats(df)
        print(f"\n✅ Terminé. Tous les plots sont dans {OUTPUT_DIR}")
    else:
        print("\nAucun résultat à afficher.")


if __name__ == "__main__":
    main()