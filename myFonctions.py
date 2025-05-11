from sklearn.cluster import KMeans, DBSCAN
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from datetime import timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler


def validite_M0(df, date_col, features, n_clusters=4, pas_jours=5, periode_init=0.2, seuil_ARI=0.7):
    # Tri et mise à l'échelle
    df = df.sort_values(date_col).copy()
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Dates limites
    date_min, date_max = df[date_col].min(), df[date_col].max()
    nb_jours_total = (date_max - date_min).days
    date_seuil_init = date_min + timedelta(days=int(nb_jours_total * periode_init))

    # Entraînement de M0 sur T0
    df_T0 = df[df[date_col] <= date_seuil_init].copy()
    M0 = entrainer_modele(df_T0, features, n_clusters)
    df['M0_preds'] = M0.predict(df[features])

    # Initialisation boucle
    results = []
    date_courante = date_seuil_init
    periode_index = 1

    while date_courante + timedelta(days=pas_jours) <= date_max:
        date_debut = date_courante
        date_courante = date_debut + timedelta(days=pas_jours)

        # Données de la nouvelle tranche Tn uniquement
        mask_Tn = (df[date_col] > date_debut) & (df[date_col] <= date_courante)
        df_Tn = df[mask_Tn].copy()

        if df_Tn.shape[0] < n_clusters:
            break  # Pas assez de données pour entraîner un KMeans

        # Entraînement de Mn sur tout jusqu'à date_courante
        df_Tcum = df[df[date_col] <= date_courante].copy()
        Mn = entrainer_modele(df_Tcum, features, n_clusters)

        # Prédiction sur la tranche Tn uniquement
        pred_M0 = df.loc[mask_Tn, 'M0_preds']
        pred_Mn = Mn.predict(df_Tn[features])

        # Calcul ARI
        ari = adjusted_rand_score(pred_M0, pred_Mn)
        results.append({'periode': f'T{periode_index}', 'date': date_courante, 'ARI': ari})

        # Vérification du seuil
        if ari < seuil_ARI:
            print(f"⚠️  M0 devient obsolète **après** la période {periode_index - 1} ({date_debut.date()}) car ARI = {ari:.3f} est passé sous le seuil de {seuil_ARI}")
            break

        periode_index += 1

    # Résultats
    df_result = pd.DataFrame(results)
    
    # Graphique
    plt.figure(figsize=(10, 5))
    plt.plot(df_result['date'], df_result['ARI'], marker='o')
    plt.axhline(seuil_ARI, color='red', linestyle='--', label=f'Seuil ARI = {seuil_ARI}')
    plt.title("Évolution de l'ARI entre M0 et Mn (sur Tn uniquement)")
    plt.xlabel("Date")
    plt.ylabel("ARI")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return df_result

def entrainer_modele(data, features, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(data[features])
    return model

def comparer_models(df, date_col, features, n_clusters, periode_init=0.2, pas_jours_list=[2, 5, 10, 15]):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Standardisation
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    date_min, date_max = df[date_col].min(), df[date_col].max()
    nb_jours_total = (date_max - date_min).days
    date_seuil_init = date_min + timedelta(days=int(nb_jours_total * periode_init))

    # Résultats finaux
    dict_resultats = {}

    for pas_jours in pas_jours_list:
        resultats = []
        date_courante = date_seuil_init

        # Données pour T0 et M0
        df_T_cumule = df[df[date_col] <= date_courante].copy()
        modele_prec = entrainer_modele(df_T_cumule, features, n_clusters)

        i = 1
        while date_courante < date_max:
            # Étendre la période
            date_debut_Tn = date_courante
            date_courante = date_courante + timedelta(days=pas_jours)

            df_Tn = df[(df[date_col] > date_debut_Tn) & (df[date_col] <= date_courante)].copy()
            if df_Tn.empty:
                break

            # Étendre la période cumulée
            df_T_cumule = df[df[date_col] <= date_courante].copy()

            # Entraîner le nouveau modèle Mn
            modele_n = entrainer_modele(df_T_cumule, features, n_clusters)

            # Comparer Mn vs Mn-1 sur Tn
            pred_prec = modele_prec.predict(df_Tn[features])
            pred_n = modele_n.predict(df_Tn[features])
            ari = adjusted_rand_score(pred_prec, pred_n)

            resultats.append({
                'periode_jours': pas_jours,
                'iteration': i,
                'date_Tn': date_courante,
                'ARI': ari
            })

            # Mettre à jour le modèle précédent
            modele_prec = modele_n
            i += 1

        # Sauvegarder les résultats pour cette période
        dict_resultats[pas_jours] = pd.DataFrame(resultats)

    return dict_resultats

def tracer_stabilite_ari(dict_res):
    # Créer un graphique avec 2x2 sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # On transforme axes en un tableau 1D pour faciliter l'indexation
    axes = axes.flatten()

    for i, pas_jours in enumerate([2, 5, 10, 15]):
        df_ari = dict_res[pas_jours]

        # Calcul de la moyenne ARI
        ari_moyenne = df_ari['ARI'].mean()

        # Tracer l'ARI sur chaque sous-graphe
        axes[i].plot(df_ari['date_Tn'], df_ari['ARI'], marker='o', label=f'ARI (période {pas_jours} j)')
        axes[i].axhline(0.9, color='green', linestyle='--', label='Seuil haut (0.9)')
        axes[i].axhline(0.5, color='red', linestyle='--', label='Seuil bas (0.5)')
        axes[i].axhline(ari_moyenne, color='blue', linestyle='-.', label=f'Moyenne ARI = {ari_moyenne:.2f}')
        
        # Titres et légendes
        axes[i].set_title(f'Période de {pas_jours} jours')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('ARI')
        axes[i].legend(loc='best')

    # Ajuster l'espacement entre les graphes
    plt.tight_layout()
    plt.show()

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
