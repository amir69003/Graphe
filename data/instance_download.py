import os
import urllib.request

# Liste d'instances variées (Nom, Jobs x Machines, Difficulté)
# Source: JSPLIB
# Générer par LLM
INSTANCES_TO_DOWNLOAD = [
    "la01", "la02", "la03", "la04", "la05", # 10x5
    "la06", "la07", "la08", "la09", "la10", # 15x5
    "la11",    # 20x5

    "la12", "la13", "la14", "la15", # 20x5 (Suite)
    "la16", "la17", "la18", "la19", "la20", # 10x10 (Début de la zone de friction)
    "la21", "la22", "la23", "la24", "la25", # 15x10 (Plus dense)
    "la31",    # 30x10 (Gros volume mais rectangulaire)

    "ft10",    # 10x10 (LÉGENDAIRE : petite mais très dure)
    "abz5", "abz6", "abz7", "abz8", "abz9", # 10x10 (Très denses)
    "orb01", "orb02", "orb03", "orb04", "orb05", # 10x10 (Conçus pour être durs)
    "yn1", "yn2",  # 20x20 (Yamada Nakano - Carrés et durs)

    "la36", "la38", "la40", # 15x15 (Carrés parfaits)
    "ta01",    # 15x15
    "ta11",    # 20x15
    "ta21",    # 20x20
    "ta31",    # 30x15
    "ta41",    # 30x20
    "ta51",    # 50x15 
    "ta71"     # 100x20 
]



def download_instances():
    base_url = "https://raw.githubusercontent.com/tamy0612/JSPLIB/master/instances/"
    save_dir = "instances_jsp"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Dossier '{save_dir}' créé.")

    print(f"Téléchargement de {len(INSTANCES_TO_DOWNLOAD)} instances...\n")

    for name in INSTANCES_TO_DOWNLOAD:
        url = f"{base_url}{name}"
        file_path = os.path.join(save_dir, f"{name}.txt")
        
        try:
            print(f" - Téléchargement de {name}...", end="")
            with urllib.request.urlopen(url) as response:
                content = response.read().decode('utf-8')
                
            with open(file_path, "w") as f:
                f.write(content)
            print(" OK")
            
        except Exception as e:
            print(f" ERREUR : {e}")

    print("\nTout est prêt dans le dossier 'instances_jsp' !")

if __name__ == "__main__":
    download_instances()