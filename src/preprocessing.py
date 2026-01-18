"""
Data preprocessing for house price prediction
v2: Added new features - size_per_room, age_bucket, season, bed_bath_product
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class Preprocessor:
    def __init__(self):
        self.encoders = {}
        self.scaler = None
        self.feature_cols = None
        self.num_cols = ['Size', 'Bedrooms', 'Bathrooms', 'Year Built']
        self.cat_cols = ['Location', 'Condition', 'Type']

    def _add_features(self, df):
        df = df.copy()

        # property age at time of sale
        df['property_age'] = df['Date Sold'].dt.year - df['Year Built']
        df['property_age'] = df['property_age'].clip(lower=0)

        # size buckets
        df['size_cat'] = pd.cut(
            df['Size'],
            bins=[0, 1500, 2500, 3500, float('inf')],
            labels=['small', 'medium', 'large', 'xlarge']
        )

        # room features
        df['total_rooms'] = df['Bedrooms'] + df['Bathrooms']
        df['bath_ratio'] = df['Bathrooms'] / df['Bedrooms'].replace(0, 1)

        # temporal
        df['year_sold'] = df['Date Sold'].dt.year
        df['month_sold'] = df['Date Sold'].dt.month
        df['quarter'] = df['Date Sold'].dt.quarter

        # flags
        df['is_new'] = (df['property_age'] <= 5).astype(int)
        df['decade'] = (df['Year Built'] // 10) * 10

        # === NEW FEATURES (v2) ===

        # size per bedroom - indicates spaciousness
        df['size_per_bedroom'] = df['Size'] / df['Bedrooms'].replace(0, 1)

        # size per total room
        df['size_per_room'] = df['Size'] / df['total_rooms'].replace(0, 1)

        # bedroom * bathroom interaction
        df['bed_bath_product'] = df['Bedrooms'] * df['Bathrooms']

        # age buckets: new (0-10), mid (11-40), old (40+)
        df['age_bucket'] = pd.cut(
            df['property_age'],
            bins=[-1, 10, 40, 200],
            labels=['new', 'mid', 'old']
        )

        # season based on sale month
        df['season'] = df['month_sold'].map({
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'fall', 10: 'fall', 11: 'fall'
        })

        # days since 2020 (captures time trend)
        ref_date = pd.Timestamp('2020-01-01')
        df['days_since_2020'] = (df['Date Sold'] - ref_date).dt.days

        # is luxury (large size + new condition)
        df['is_luxury'] = ((df['Size'] > 3000) & (df['Condition'] == 'New')).astype(int)

        return df

    def _fill_missing(self, df):
        df = df.copy()

        # numerics -> median
        for col in self.num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # categoricals -> mode
        for col in self.cat_cols:
            if col in df.columns and len(df[col].mode()) > 0:
                df[col] = df[col].fillna(df[col].mode()[0])

        # engineered cols
        eng_num_cols = [
            'property_age', 'total_rooms', 'bath_ratio', 'decade',
            'size_per_bedroom', 'size_per_room', 'bed_bath_product', 'days_since_2020'
        ]
        for col in eng_num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        if 'size_cat' in df.columns:
            df['size_cat'] = df['size_cat'].fillna('medium')
        if 'age_bucket' in df.columns:
            df['age_bucket'] = df['age_bucket'].fillna('mid')
        if 'season' in df.columns:
            df['season'] = df['season'].fillna('summer')

        return df

    def _encode(self, df, fit=True):
        df = df.copy()
        encode_cols = ['Location', 'Condition', 'Type', 'size_cat', 'age_bucket', 'season']

        for col in encode_cols:
            if col not in df.columns:
                continue

            if fit:
                self.encoders[col] = df[col].unique().tolist()

            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])

        return df

    def fit_transform(self, df, target='Price'):
        df = df.dropna(subset=[target])

        df = self._add_features(df)
        df = self._fill_missing(df)
        df = self._encode(df, fit=True)

        drop_cols = ['Property ID', 'Date Sold', 'Price']
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        X = X.select_dtypes(include=[np.number])

        self.feature_cols = X.columns.tolist()
        y = df[target].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y, self.feature_cols

    def transform(self, df):
        df = self._add_features(df)
        df = self._fill_missing(df)
        df = self._encode(df, fit=False)

        X = df.reindex(columns=self.feature_cols, fill_value=0)
        X = X.select_dtypes(include=[np.number])

        return self.scaler.transform(X)

    def save(self, path):
        joblib.dump({
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'num_cols': self.num_cols,
            'cat_cols': self.cat_cols
        }, path)

    def load(self, path):
        data = joblib.load(path)
        self.encoders = data['encoders']
        self.scaler = data['scaler']
        self.feature_cols = data['feature_cols']
        self.num_cols = data['num_cols']
        self.cat_cols = data['cat_cols']


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "Case Study 1 Data (1).xlsx"

    print("Loading data...")
    df = pd.read_excel(data_path)

    prep = Preprocessor()
    X, y, features = prep.fit_transform(df)

    print(f"Shape: {X.shape}")
    print(f"Features ({len(features)}): {features}")
