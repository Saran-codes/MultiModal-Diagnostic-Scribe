import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_splits(registry_path, output_dir):
    if not os.path.exists(registry_path):
        print(f"Error: {registry_path} not found.")
        return
        
    df = pd.read_csv(registry_path)
    initial_count = len(df)

    # only include rows with generated reports
    df = df[df['report_generated'] == True].reset_index(drop=True)
    filtered_count = len(df)
    
    print(f"Loaded {initial_count} records. Kept {filtered_count} records with generated reports.")

    # 80/10/10 stratified split
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=42, 
        stratify=df['mapped_label']
    )

    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=42, 
        stratify=temp_df['mapped_label']
    )

    train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)

    print("-" * 30)
    print(f"TRAIN set: {len(train_df)} samples")
    print(f"VAL set:   {len(val_df)} samples")
    print(f"TEST set:  {len(test_df)} samples")
    print("-" * 30)
    
    print("\nClass Distribution in Training Set (%):")
    print((train_df['mapped_label'].value_counts(normalize=True) * 100).round(2))
    print("\nSplits saved successfully to:", output_dir)

if __name__ == "__main__":
    REGISTRY_PATH = "./data/master_registry.csv"
    OUTPUT_DIR = "./data"
    
    create_splits(REGISTRY_PATH, OUTPUT_DIR)