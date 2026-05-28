import pandas as pd
import os

def read_csv_with_encoding(path):
    encodings = ['utf-8-sig', 'gb18030', 'utf-8', 'gbk']
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError, UnicodeError):
            continue
    return pd.read_csv(path, encoding='utf-8', errors='replace')

def merge_references():
    file_mapping = [
        {
            "source": r"D:\XJTU\ImportantFile\LLMs\datasets\Al_Alloys\USTB\USTB_Al_alloys_processed_split_withID.csv",
            "target": r"d:\XJTU\ImportantFile\LLMs\BERT_ML\datasets\Al_Alloys\aluminum.csv",
            "ref_cols": ["References"]
        },
        {
            "source": r"D:\XJTU\ImportantFile\LLMs\datasets\HEA_data\RoomTemperature_HEA_train_with_ID.csv",
            "target": r"d:\XJTU\ImportantFile\LLMs\BERT_ML\datasets\HEA_data\hea.csv",
            "ref_cols": ["References", "DOIs", "reference_id", "doi", "Number"]
        },
        {
            "source": r"D:\XJTU\ImportantFile\LLMs\datasets\Steel\USTB_steel_processed_withID.csv",
            "target": r"d:\XJTU\ImportantFile\LLMs\BERT_ML\datasets\Steel\steel.csv",
            "ref_cols": ["DOIs", "References"]
        },
        {
            "source": r"D:\XJTU\ImportantFile\LLMs\datasets\Ti_alloys\Titanium_Alloy_Dataset_Processed_cleaned.csv",
            "target": r"d:\XJTU\ImportantFile\LLMs\BERT_ML\datasets\Ti_alloys\titanium.csv",
            "ref_cols": ["reference_id", "References", "DOIs"]
        }
    ]

    for item in file_mapping:
        src_path = item["source"]
        tgt_path = item["target"]
        
        if not os.path.exists(src_path):
            print(f"Source file not found: {src_path}")
            continue
        if not os.path.exists(tgt_path):
            print(f"Target file not found: {tgt_path}")
            continue

        print(f"Processing: {os.path.basename(tgt_path)}...")
        df_src = read_csv_with_encoding(src_path)
        df_tgt = read_csv_with_encoding(tgt_path)

        src_id_col = next((c for c in ['ID', 'id'] if c in df_src.columns), None)
        tgt_id_col = next((c for c in ['ID', 'id'] if c in df_tgt.columns), None)

        if not src_id_col or not tgt_id_col:
            print(f"ID column missing for {os.path.basename(tgt_path)}")
            continue

        found_ref_col = next((c for c in item["ref_cols"] if c in df_src.columns), None)
        if not found_ref_col:
            found_ref_col = next((c for c in df_src.columns if any(x in c.lower() for x in ["reference", "doi"])), None)

        if not found_ref_col:
            print(f"No reference column found for {os.path.basename(tgt_path)}")
            continue

        print(f"Found reference column: {found_ref_col}")
        df_src_subset = df_src[[src_id_col, found_ref_col]].copy()
        if src_id_col != tgt_id_col:
            df_src_subset = df_src_subset.rename(columns={src_id_col: tgt_id_col})

        if found_ref_col in df_tgt.columns:
            df_tgt = df_tgt.drop(columns=[found_ref_col])

        df_merged = pd.merge(df_tgt, df_src_subset, on=tgt_id_col, how='left')
        
        cols = list(df_merged.columns)
        cols.remove(found_ref_col)
        cols.insert(cols.index(tgt_id_col) + 1, found_ref_col)
        df_merged = df_merged[cols]

        df_merged.to_csv(tgt_path, index=False, encoding='utf-8-sig')
        print(f"Successfully updated {tgt_path}")

if __name__ == "__main__":
    merge_references()
