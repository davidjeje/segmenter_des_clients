from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score

def get_cluster_labels(model, X):
    return model.fit_predict(X)

def compute_ari_between_periods(df, periods, model_class, model_kwargs):
    aris = []
    for i in range(len(periods)-1):
        X1 = df[df['period'] == periods[i]].drop(columns=['period'])
        X2 = df[df['period'] == periods[i+1]].drop(columns=['period'])

        model1 = model_class(**model_kwargs)
        model2 = model_class(**model_kwargs)

        labels1 = get_cluster_labels(model1, X1)
        labels2 = get_cluster_labels(model2, X2)

        # Nécessaire : faire correspondre les observations si même individus (sinon l'ARI n'est pas pertinent)
        # Si X1 et X2 ne contiennent pas les mêmes individus, l'ARI n'est pas interprétable ici.

        ari = adjusted_rand_score(labels1, labels2)
        aris.append({'period_1': periods[i], 'period_2': periods[i+1], 'ari': ari})
    return pd.DataFrame(aris)

def compute_ks_test_between_periods(df, periods):
    results = []
    numeric_cols = df.select_dtypes(include='number').columns.drop('period')

    for i in range(len(periods)-1):
        df1 = df[df['period'] == periods[i]]
        df2 = df[df['period'] == periods[i+1]]
        
        for col in numeric_cols:
            stat, p_value = ks_2samp(df1[col], df2[col])
            results.append({
                'period_1': periods[i],
                'period_2': periods[i+1],
                'feature': col,
                'ks_stat': stat,
                'p_value': p_value
            })
    return pd.DataFrame(results)

def plot_ari(df_ari):
    plt.figure(figsize=(10, 5))
    plt.plot(df_ari['period_2'].astype(str), df_ari['ari'], marker='o')
    plt.xticks(rotation=45)
    plt.ylabel('Adjusted Rand Index')
    plt.title('Stabilité du clustering dans le temps')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_distribution(df, feature):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='period', y=feature, data=df)
    plt.title(f'Évolution temporelle de la variable : {feature}')
    plt.xticks(rotation=45)
    plt.show()
