import pandas as pd
import numpy as np

class BasicEDA():
    def __init__(self,df:pd.DataFrame):
        self.df = df
        self.n = len(df)

    # Missing_vlaue
    def check_missing_values(self):
        missing = self.df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column':missing.index,
            'Count':missing.values,
            'pct':((missing.values)/self.n) * 100
            })
        missing_df = missing_df[missing_df['pct'] > 0].sort_values('Count',ascending=False)
        if missing_df.shape[0] == 0:
            return f'No Missing value'
        return missing_df

    # Duplicate row
    def check_duplicate(self):
        row_hashes = pd.util.hash_pandas_object(self.df,index=False)
        unique_hashes, inverse_indices,counts = np.unique(
            row_hashes.values,
            return_inverse=True,
            return_counts=True
        )
        mask = counts > 1
        if mask.sum() > 0:
            pair = []
            duplicate_hashes = unique_hashes[mask]
            for val in duplicate_hashes:
                indices = np.where(row_hashes.values == val)[0]
                pair.append(indices.tolist())
            return pair
        return f'No duplicate'

    # n_unique()
    def check_unique_count(self):
        counts = pd.DataFrame({
            'Feature': self.df.columns,
            'Unique':[self.df[col].nunique() for col in self.df.columns],
            'Sample_size': self.n,
            'Cardinality_Ratio': [round(self.df[col].nunique()/self.n * 100,5) for col in self.df.columns]
        })
        return counts.sort_values('Cardinality_Ratio',ascending=False)

    def skewness_kurtosis(self, features = None):
        if not features:
            return "Select numerical features"
        desc = self.df[features].describe().T
        desc['skewness'] = self.df[features].skew()
        desc['kurtosis'] = self.df[features].kurtosis()
        desc['range'] = desc['max'] - desc['min']
        desc['iqr'] = desc['75%'] - desc['25%']
        result = desc.reset_index().rename(columns={'index': 'feature'})
        column_order = ['feature', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 
                    'max', 'range', 'iqr', 'skewness', 'kurtosis']
        
        return result[column_order].sort_values(['skewness', 'kurtosis'],ascending=True)

    def Categorical_Dis(self,features = None):
        if not features:
            return "Select categorical features"
        results = []
        for col in features:
            value_counts = self.df[col].value_counts()
            ratios = value_counts.values/self.n
            gini = 1 - np.sum(ratios ** 2)
            entropy = -np.sum(ratios*np.log2(ratios + 1e-10))
            results.append({
                'feature':col,
                'gini':gini,
                'entropy':entropy
            })
        return pd.DataFrame(results).sort_values('entropy',ascending=True)