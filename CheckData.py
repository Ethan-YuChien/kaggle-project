from scipy.stats import wasserstein_distance, ks_2samp,chi2_contingency
import numpy as np
import pandas as pd
class ShiftData():
    def __init__(self,train,test):
        self.train = train
        self.test = test
    
    def calculate_kl_divergence(self,p, q):
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        return np.sum(p * np.log(p / q))

    def calculate_js_divergence(self,p, q):
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        m = 0.5 * (p + q)
        return 0.5 * self.calculate_kl_divergence(p, m) + 0.5 * self.calculate_kl_divergence(q, m)

    def check_num_shift(self,num_fea):
        results = []
        
        for feature in num_fea:
            train_vals = self.train[feature].dropna()
            test_vals = self.test[feature].dropna()
            
            try:
                w_dist = wasserstein_distance(train_vals, test_vals)
            except:
                w_dist = None
            
            try:
                ks_stat, ks_p = ks_2samp(train_vals, test_vals)
            except:
                ks_stat, ks_p = None, None
            
            results.append({
                'feature': feature,
                'wasserstein_dist': w_dist,
                'ks_pvalue': ks_p
            })
        
        return pd.DataFrame(results).sort_values('ks_pvalue',ascending=True)
        
    def check_cate_shift(self,cate_fea):

        results = []
        
        for feature in cate_fea:
            
            train_dist = self.train[feature].value_counts(normalize=True).sort_index()
            test_dist = self.test[feature].value_counts(normalize=True).sort_index()
            
            all_categories = set(train_dist.index) | set(test_dist.index)
            train_vec = np.array([train_dist.get(cat, 0) for cat in all_categories])
            test_vec = np.array([test_dist.get(cat, 0) for cat in all_categories])
            
            # 1. 卡方检验
            try:
                train_counts = self.train[feature].value_counts()
                test_counts = self.test[feature].value_counts()
                contingency_table = pd.DataFrame({'train': train_counts, 'test': test_counts}).fillna(0).T
                chi2, chi_p, dof, _ = chi2_contingency(contingency_table)
            except:
                chi2, chi_p, dof = None, None, None
            
            kl_div = self.calculate_kl_divergence(train_vec, test_vec)
            
            js_div = self.calculate_js_divergence(train_vec, test_vec)
            
            results.append({
                'feature': feature,
                'chi2_pvalue': chi_p,
                'kl_divergence': kl_div,
                'js_divergence': js_div,
                'n_unique_train': len(train_dist),
                'n_unique_test': len(test_dist)
            })
        
        return pd.DataFrame(results).sort_values('js_divergence',ascending=True)
        