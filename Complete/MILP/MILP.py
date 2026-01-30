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
from ortools.linear_solver import pywraplp

# ==========================================
# 0. CONFIGURATION & UTILITAIRES
# ==========================================

INSTANCE_DIR = config.INSTANCE_DIR  # Dossier contenant les .txt
OUTPUT_DIR = "../plot/MILP"  # Dossier spécifique pour les résultats MILP
INSTANCES = ["ft06", "ft10", "abz6", "la27", "ta31"] # Ajoute tes instances ici
TIME_LIMIT = 300.0  # Temps limite en secondes (ex: 300s = 5min)

# Création du dossier de sortie
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def parse_instance(file_path):
    """
    Lit une instance JSSP (format standard) de manière robuste.
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
# 1. LE SOLVEUR MILP (Adapté du Notebook)
# ==========================================

def solve_milp_jsp(jobs_data, num_machines, time_limit=300.0):
    """
    Résout le JSSP via MILP (SCIP) avec formulation Big-M.
    Retourne un dictionnaire standardisé pour les plots.
    """
    num_jobs = len(jobs_data)

    # Création du solveur SCIP
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("❌ Solveur SCIP introuvable.")
        return {"status": "ERROR", "time": 0, "makespan": 0, "schedule": []}

    # Calcul de l'horizon (Big-M)
    horizon = sum(task[1] for job in jobs_data for task in job)

    # --- 1. Variables ---
    # starts[j][t] : temps de début de la t-ème tâche du job j
    starts = {}
    for j in range(num_jobs):
        for t in range(len(jobs_data[j])):
            starts[(j, t)] = solver.IntVar(0.0, float(horizon), f'start_{j}_{t}')

    # makespan : variable à minimiser
    makespan = solver.IntVar(0.0, float(horizon), 'makespan')

    # Helper pour retrouver les tâches par machine
    # task_on_machine[m] = liste de (job_index, task_index_in_job, duration)
    task_on_machine = {m: [] for m in range(num_machines)}
    for j in range(num_jobs):
        for t, (mach, dur) in enumerate(jobs_data[j]):
            task_on_machine[mach].append((j, t, dur))

    # --- 2. Contraintes ---

    # A. Contraintes de précédence (Intra-Job)
    for j in range(num_jobs):
        for t in range(len(jobs_data[j]) - 1):
            dur = jobs_data[j][t][1]
            # start[t+1] >= start[t] + dur
            solver.Add(starts[(j, t + 1)] >= starts[(j, t)] + dur)

        # Contrainte pour le Makespan (après la dernière tâche)
        last_t = len(jobs_data[j]) - 1
        last_dur = jobs_data[j][last_t][1]
        solver.Add(makespan >= starts[(j, last_t)] + last_dur)

    # B. Contraintes Disjonctives (Inter-Job sur même machine) - BIG-M
    M = float(horizon)
    num_binary_vars = 0

    for m in range(num_machines):
        tasks = task_on_machine[m]
        for idx1 in range(len(tasks)):
            for idx2 in range(idx1 + 1, len(tasks)):
                j1, t1, dur1 = tasks[idx1]
                j2, t2, dur2 = tasks[idx2]

                # Variable binaire Y : 1 si j1 passe AVANT j2, 0 sinon
                y = solver.IntVar(0, 1, f'y_m{m}_j{j1}_j{j2}')
                num_binary_vars += 1

                # Si y=1 (j1 avant j2) => start2 >= start1 + dur1
                solver.Add(starts[(j2, t2)] >= starts[(j1, t1)] + dur1 - M * (1 - y))

                # Si y=0 (j2 avant j1) => start1 >= start2 + dur2
                solver.Add(starts[(j1, t1)] >= starts[(j2, t2)] + dur2 - M * y)

    # --- 3. Objectif et Résolution ---
    solver.Minimize(makespan)
    solver.SetTimeLimit(int(time_limit * 1000))  # en ms

    t0 = time.time()
    status_code = solver.Solve()
    elapsed = time.time() - t0

    # Mapping du statut
    status_map = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
    }
    status_str = status_map.get(status_code, "UNKNOWN")

    # Extraction Résultats
    schedule = []
    final_makespan = None

    # Pour le solver MILP, on utilise 'Nodes' comme proxy pour 'Branches'
    # et 'NumConstraints' comme proxy approximatif pour la complexité
    nodes_count = solver.nodes()

    if status_code == pywraplp.Solver.OPTIMAL or status_code == pywraplp.Solver.FEASIBLE:
        final_makespan = solver.Objective().Value()

        for j in range(num_jobs):
            for t, (mach, dur) in enumerate(jobs_data[j]):
                start_val = starts[(j, t)].solution_value()
                schedule.append({
                    "Job": j,
                    "Task": t,
                    "Machine": mach,
                    "Start": start_val,
                    "Duration": dur,
                    "Finish": start_val + dur
                })

    return {
        "status": status_str,
        "time": elapsed,
        "branches": nodes_count,  # SCIP nodes
        "conflicts": num_binary_vars,  # On utilise le nb de var binaires comme métrique de complexité ici
        "makespan": final_makespan,
        "schedule": schedule
    }


# ==========================================
# 2. FONCTIONS DE PLOT (Identiques au modèle)
# ==========================================

def plot_gantt(schedule, makespan, instance_name):
    """Génère un diagramme de Gantt pour une solution donnée"""
    if not schedule:
        return

    df = pd.DataFrame(schedule)

    # --- CORRECTION ICI : On s'assure que Job est bien un entier ---
    df['Job'] = df['Job'].astype(int)
    df['Start'] = df['Start'].astype(float)
    df['Duration'] = df['Duration'].astype(float)

    machines = sorted(df['Machine'].unique())

    # Palette de couleurs
    unique_jobs = df['Job'].unique()
    nb_colors = len(unique_jobs)
    colors = plt.cm.tab20(np.linspace(0, 1, nb_colors))

    fig, ax = plt.subplots(figsize=(12, 6))

    for _, row in df.iterrows():
        m = row['Machine']
        start = row['Start']
        duration = row['Duration']
        job_id = int(row['Job'])  # --- SECURITE SUPPLEMENTAIRE : cast en int ---

        # Protection contre la division par zéro si nb_colors = 0 (peu probable ici mais bon)
        color_idx = job_id % len(colors)

        rect = patches.Rectangle((start, m - 0.4), duration, 0.8,
                                 edgecolor='black',
                                 facecolor=colors[color_idx],
                                 alpha=0.8)
        ax.add_patch(rect)

        if duration > 0:
            # Texte (Job ID)
            ax.text(start + duration / 2, m, f"J{job_id}",
                    ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    ax.set_yticks(machines)
    ax.set_yticklabels([f"Machine {m}" for m in machines])
    ax.set_xlabel("Temps")
    ax.set_title(f"Gantt MILP - {instance_name} (Makespan: {makespan})")

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

    plt.title("Performance MILP: Temps vs Makespan")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stats_time_makespan.png"))
    plt.close()

    # 2. Complexity (Nodes vs Binary Vars)
    plt.figure(figsize=(10, 6))
    # Note: conflicts ici stocke le nombre de var binaires, branches stocke les noeuds SCIP
    plt.scatter(results_df['conflicts'], results_df['branches'],
                s=100, c=results_df['time'], cmap='viridis', edgecolors='k')

    for i, txt in enumerate(results_df['instance']):
        plt.annotate(txt, (results_df['conflicts'][i], results_df['branches'][i]),
                     xytext=(5, 5), textcoords='offset points')

    plt.colorbar(label='Temps (s)')
    plt.xlabel('Nombre de Variables Binaires (Complexité Modèle)')
    plt.ylabel('Nombre de Noeuds Explorés (SCIP Nodes)')
    plt.title('Complexité de résolution MILP')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stats_complexity.png"))
    plt.close()


# ==========================================
# 3. MAIN LOOP
# ==========================================

def main():
    print(f"--- Lancement Benchmark MILP (SCIP) ---")
    print(f"Sortie des plots : {OUTPUT_DIR}")
    print(f"Time Limit : {TIME_LIMIT}s\n")

    summary_results = []

    for name in INSTANCES:
        file_path = os.path.join(INSTANCE_DIR, f"{name}.txt")

        # Vérification existence fichier
        if not os.path.exists(file_path):
            print(f"⚠️  Fichier {name}.txt introuvable dans {INSTANCE_DIR}. Passage.")
            continue

        print(f"Traitement de {name}...", end=" ", flush=True)

        jobs_data, num_machines = parse_instance(file_path)
        if not jobs_data:
            continue

        # Résolution
        res = solve_milp_jsp(jobs_data, num_machines, time_limit=TIME_LIMIT)

        mks_disp = res['makespan'] if res['makespan'] else "N/A"
        print(f"[{res['status']}] Makespan: {mks_disp} | Temps: {res['time']:.2f}s | Noeuds: {res['branches']}")

        # Sauvegarde stats
        summary_results.append({
            "instance": name,
            "status": res['status'],
            "time": res['time'],
            "makespan": res['makespan'] if res['makespan'] else 0,
            "branches": res['branches'],  # SCIP Nodes
            "conflicts": res['conflicts']  # Binary Vars Count
        })

        # Plot Gantt individuel
        if res['schedule']:
            plot_gantt(res['schedule'], res['makespan'], name)

    # Plot global si on a des données
    if summary_results:
        df = pd.DataFrame(summary_results)
        print("\n--- Résumé ---")
        print(df[["instance", "status", "makespan", "time", "branches"]])

        # Sauvegarde CSV des résultats
        df.to_csv(os.path.join(OUTPUT_DIR, "results_summary.csv"), index=False)

        plot_stats(df)
        print(f"\n✅ Terminé. Tous les plots sont dans {OUTPUT_DIR}")
    else:
        print("\nAucun résultat à afficher.")


if __name__ == "__main__":
    main()