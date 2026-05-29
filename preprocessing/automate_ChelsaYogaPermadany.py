import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# FUNGSI-FUNGSI PREPROCESSING
# ============================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset dari file CSV atau Excel (.xlsx).

    Args:
        filepath: Path ke file raw dataset
    Returns:
        DataFrame hasil load
    """
    print(f"[1/4] Loading data dari: {filepath}")
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == ".xlsx" or ext == ".xls":
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    print(f"      Shape awal: {df.shape}")
    print(f"      Kolom     : {list(df.columns)}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menangani missing values pada dataset.
    Jika ada, isi dengan median (numerik) atau modus (kategorikal).

    Args:
        df: DataFrame input
    Returns:
        DataFrame tanpa missing values
    """
    print("[2/4] Handling missing values...")
    missing_before = df.isnull().sum().sum()

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"      Kolom '{col}' (kategorikal) diisi dengan modus.")
            else:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"      Kolom '{col}' (numerik) diisi dengan median.")

    missing_after = df.isnull().sum().sum()
    print(f"      Missing values: {missing_before} → {missing_after}")
    return df


def encode_label(df: pd.DataFrame, target_col: str = 'Class') -> tuple:
    """
    Melakukan Label Encoding pada kolom target.

    Args:
        df        : DataFrame input
        target_col: Nama kolom target (default: 'Class')
    Returns:
        Tuple (DataFrame dengan kolom target ter-encode, objek LabelEncoder)
    """
    print(f"[3/4] Encoding kolom target '{target_col}'...")
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])
    print(f"      Mapping kelas: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return df, le


def scale_features(df: pd.DataFrame, target_col: str = 'Class') -> tuple:
    """
    Melakukan StandardScaler pada fitur (selain kolom target).

    Args:
        df        : DataFrame input
        target_col: Nama kolom target yang dikecualikan dari scaling
    Returns:
        Tuple (DataFrame dengan fitur ter-scale, objek StandardScaler)
    """
    print(f"[4/4] Scaling fitur dengan StandardScaler...")
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    df_scaled = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
    print(f"      Scaling selesai. Shape akhir: {df_scaled.shape}")
    return df_scaled, scaler


def save_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Menyimpan DataFrame hasil preprocessing ke file CSV.

    Args:
        df         : DataFrame hasil preprocessing
        output_path: Path tujuan penyimpanan CSV
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Data preprocessing berhasil disimpan ke: {output_path}")
    print(f"   Shape final: {df.shape}")


# ============================================================
# FUNGSI UTAMA — PIPELINE PREPROCESSING
# ============================================================

def preprocess(input_path: str, output_path: str, target_col: str = 'Class') -> pd.DataFrame:
    """
    Menjalankan seluruh pipeline preprocessing secara otomatis.

    Pipeline:
        1. Load data (support .csv dan .xlsx)
        2. Handle missing values
        3. Label encoding pada target
        4. Standard scaling pada fitur
        5. Simpan hasil ke CSV

    Args:
        input_path : Path file raw dataset (.csv atau .xlsx)
        output_path: Path file CSV output hasil preprocessing
        target_col : Nama kolom target (default: 'Class')
    Returns:
        DataFrame hasil preprocessing yang siap dilatih
    """
    print("=" * 55)
    print("   PIPELINE PREPROCESSING — RAISIN DATASET")
    print("=" * 55)

    df = load_data(input_path)
    df = handle_missing_values(df)
    df, le = encode_label(df, target_col)
    df_final, scaler = scale_features(df, target_col)
    save_data(df_final, output_path)

    print("=" * 55)
    print("   PREPROCESSING SELESAI")
    print("=" * 55)
    return df_final


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Path relatif dari root repository
    INPUT_PATH  = "raisin_raw/Raisin_Dataset.xlsx"   # ← support .xlsx
    OUTPUT_PATH = "preprocessing/raisin_preprocessing/raisin_preprocessed.csv"
    TARGET_COL  = "Class"

    df_ready = preprocess(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        target_col=TARGET_COL
    )

    print("\nSample data hasil preprocessing:")
    print(df_ready.head())
