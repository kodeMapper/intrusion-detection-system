import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class DLPreprocessor:
    """
    Shared Preprocessor for the Deep Learning Pipeline (LSTM and Autoencoder).
    Ensures that both offline training validation and live real-time inference
    apply exactly the same normalizations, derivations, and encodings.
    """
    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        
        # Load components
        self.scaler = joblib.load(self.artifacts_dir / "scaler_unsw_v1_20260612.pkl")
        self.encoder = joblib.load(self.artifacts_dir / "onehot_encoder_unsw_v1_20260612.pkl")
        
        with open(self.artifacts_dir / "preprocessing_config_unsw_v1_20260612.json", "r") as f:
            self.config = json.load(f)
            
        with open(self.artifacts_dir / "feature_list_unsw_v1_20260612.json", "r") as f:
            self.feature_columns = json.load(f)

        self.drop_cols = self.config.get("dropped_columns", [])
        self.cat_cols = self.config.get("categorical_features", [])

    def derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineering defined during dataset preparation."""
        # Clean whitespace from attack_cat if present
        if 'attack_cat' in df.columns:
            df['attack_cat'] = df['attack_cat'].astype(str).str.strip()

        # Derive ports
        if 'sport' in df.columns:
            sport_num = pd.to_numeric(df['sport'], errors='coerce').fillna(65535)
            df['is_wellknown_sport'] = (sport_num < 1024).astype(int)
        else:
            df['is_wellknown_sport'] = 0

        if 'dsport' in df.columns:
            dsport_num = pd.to_numeric(df['dsport'], errors='coerce').fillna(65535)
            df['is_wellknown_dsport'] = (dsport_num < 1024).astype(int)
        else:
            df['is_wellknown_dsport'] = 0

        # Ratios
        sbytes = df.get('sbytes', pd.Series(np.zeros(len(df)), index=df.index))
        dbytes = df.get('dbytes', pd.Series(np.zeros(len(df)), index=df.index))
        spkts = df.get('spkts', pd.Series(np.zeros(len(df)), index=df.index))
        dpkts = df.get('dpkts', pd.Series(np.zeros(len(df)), index=df.index))
        
        df['bytes_ratio'] = sbytes / (dbytes + 1)
        df['pkts_ratio'] = spkts / (dpkts + 1)
        df['pkt_size_avg'] = (sbytes + dbytes) / (spkts + dpkts + 1)

        return df

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Takes raw dataframe flows and outputs the normalized numpy array
        ready for PyTorch models.
        """
        df = df.copy()

        # 1. Derive features
        df = self.derive_features(df)

        # 2. Drop non-live features
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore")
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
        if 'attack_cat' in df.columns:
            df = df.drop(columns=['attack_cat'])

        # Align columns that might be missing in live stream before encoding/scaling
        for cat in self.cat_cols:
            if cat not in df.columns:
                df[cat] = 'other'

        # 3. Encoding categorical
        encoded_data = self.encoder.transform(df[self.cat_cols])
        encoded_feature_names = self.encoder.get_feature_names_out(self.cat_cols)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=df.index)

        # Drop original cat columns and merge
        df = df.drop(columns=self.cat_cols)
        
        # 4. Filter to just numeric prior to scaling
        scale_cols = list(getattr(self.scaler, 'feature_names_in_', []))
        if not scale_cols:
            scale_cols = [c for c in df.columns if c not in encoded_feature_names]
        
        # Ensure all numeric columns are actually present and numeric
        for col in scale_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # 5. Scaling numeric
        df[scale_cols] = self.scaler.transform(df[scale_cols])

        # Merge encoded
        df = pd.concat([df[scale_cols], encoded_df], axis=1)

        # 6. Reorder and zero-fill missing columns based on lock list without fragmentation
        features_dict = {}
        for col in self.feature_columns:
            if col in df.columns:
                features_dict[col] = df[col]
            else:
                features_dict[col] = 0.0

        final_features = pd.DataFrame(features_dict, index=df.index)
        return final_features.to_numpy(dtype=np.float32)
