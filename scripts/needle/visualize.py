import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import glob
import matplotlib.font_manager as fm
import os

FOLDER_PATH = "./results_needle/results/LlaMA3_heterocache_0.5_v1/"
# FOLDER_PATH = "./results_needle/results/LlaMA3_fullkv_0.5_v1/"
MODEL_NAME = "LlaMA3"
PRETRAINED_LEN = 128000

def main():
    # ==================== Font Settings ====================
    # 1. Set Seaborn style first to avoid overwriting font settings
    sns.set_style("white")

    # 2. Force font to Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'  # Match math text font
    plt.rcParams['pdf.fonttype'] = 42         # Ensure font embedding
    plt.rcParams['ps.fonttype'] = 42

    # 3. Set global base font size
    plt.rcParams.update({'font.size': 42})
    # ========================================================

    # Path to the directory containing JSON results
    folder_path = FOLDER_PATH
    if("/" in folder_path):
        model_name = folder_path.split("/")[-2]
    else: 
        model_name = MODEL_NAME
    print("model_name = %s" % model_name)

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}*.json")
    
    # List to hold the data
    data = []

    # Iterating through each file and extract the required columns
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            try:
                document_depth = json_data.get("depth_percent", None)
                context_length = json_data.get("context_length", None)
            except:
                import pdb
                pdb.set_trace()
            
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            
            # Simple intersection score
            score = len(set(model_response.split()).intersection(set(expected_answer))) / len(set(expected_answer))
            
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)

    locations = list(df["Context Length"].unique())
    locations.sort()
    
    pretrained_len = len(locations) 
    for li, l in enumerate(locations):
        if(l > PRETRAINED_LEN): 
            pretrained_len = li
            break
    
    print(df.head())
    print("Overall score %.3f" % df["Score"].mean())

    # Creating Pivot Table
    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index()
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score")
    
    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(40, 10))
    
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,
        linecolor='grey',
        linestyle='--'
    )

    original_labels = pivot_table.columns
    
    # Format labels to "K" units
    formatted_labels = [f'{int(label // 1000)}K' for label in original_labels]
    
    # Set formatted x-axis labels
    heatmap.set_xticklabels(formatted_labels)

    # More aesthetics
    model_name_ = MODEL_NAME
    
    plt.title(f'"Needle In A HayStack" w/ FullAttention acc {df["Score"].mean():.3f}', fontsize=80)
    # plt.title(f'"Needle In A HayStack" w/ HeteroCache acc {df["Score"].mean():.3f}', fontsize=80)
    plt.xlabel('Token Limit', fontsize=90)
    plt.ylabel('Depth Percent', fontsize=80)
    plt.xticks(rotation=45, fontsize=50)
    plt.yticks(rotation=0, fontsize=50)
    plt.tight_layout()

    # Add a vertical line at the desired column index
    plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)

    save_path = "./results_needle/img/%s.png" % model_name
    print("saving at %s" % save_path)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)

if __name__ == "__main__":
    main()