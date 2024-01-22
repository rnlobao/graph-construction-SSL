import numpy as np
import pandas as pd
from scipy.stats import zscore, iqr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import LocalOutlierFactor
import sslbookdata
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csc_matrix
import os


datasets = {
    'USPS': sslbookdata.load_usps(0),
    'DIGIT': sslbookdata.load_digit1(0),
    'GC': sslbookdata.load_g241c(0),
    'GN': sslbookdata.load_g241n(0),
    'COIL': sslbookdata.load_coil2(0),
    'TEXT': sslbookdata.load_text(0)
}

def detect_outliers(data, dataset_name):
    scaler = StandardScaler(with_mean=False)
    data_scaled = scaler.fit_transform(data)

    if dataset_name == 'TEXT':
        n_components = 241
        svd = TruncatedSVD(n_components=n_components)
        data_scaled = svd.fit_transform(data_scaled)

    z_scores = np.abs(zscore(data_scaled))
    limiar_z_score = 3

    iqr_values = iqr(data_scaled, axis=0)
    lower_bound = np.median(data_scaled, axis=0) - 1.5 * iqr_values
    upper_bound = np.median(data_scaled, axis=0) + 1.5 * iqr_values
    outliers_iqr = np.any((data_scaled < lower_bound) | (data_scaled > upper_bound), axis=1)

    lof_outliers = LocalOutlierFactor(n_neighbors=20, contamination='auto').fit_predict(data_scaled) == -1

    result_df = pd.DataFrame(data_scaled, columns=[f'Feature_{i+1}' for i in range(data_scaled.shape[1])])
    result_df['Outlier_ZScore'] = np.any(z_scores > limiar_z_score, axis=1)
    result_df['Outlier_IQR'] = outliers_iqr
    result_df['Outlier_LOF'] = lof_outliers
    result_df['Dataset'] = dataset_name

    return result_df

all_outliers_df = pd.DataFrame()
outliers_count = {}

for dataset_name, dataset in datasets.items():
    outliers_df = detect_outliers(dataset['data'], dataset_name)
    all_outliers_df = pd.concat([all_outliers_df, outliers_df])

    outliers_count[dataset_name] = {
        'ZScore': outliers_df['Outlier_ZScore'].sum(),
        'IQR': outliers_df['Outlier_IQR'].sum(),
        'LOF': outliers_df['Outlier_LOF'].sum()
    }

print("Contagem de Outliers por Dataset:")
for dataset_name, count in outliers_count.items():
    print(f"{dataset_name}: Z-Score - {count['ZScore']} outliers, IQR - {count['IQR']} outliers, LOF - {count['LOF']} outliers")


def plot_density_analysis(data, dataset_name, save_folder):
    scaler = StandardScaler(with_mean=False)
    data_scaled = scaler.fit_transform(data)

    if dataset_name == 'TEXT':
        n_components = 241
        svd = TruncatedSVD(n_components=n_components)
        data_scaled = svd.fit_transform(data_scaled)

    df = pd.DataFrame(data_scaled, columns=[f'Feature_{i+1}' for i in range(241)])

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Feature_1', y='Feature_2', alpha=0.5)
    plt.title(f'Densidade dos Dados - {dataset_name}')
    plt.savefig(os.path.join(save_folder, f'density_scatter_{dataset_name}.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.histplot(df, bins=50, kde=True)
    plt.title(f'Histograma dos Dados - {dataset_name}')
    plt.savefig(os.path.join(save_folder, f'density_histogram_{dataset_name}.png'))
    plt.close()

    if isinstance(data, csc_matrix):
        data = data.toarray()
    variance = np.var(data)
    print(f"Variância dos Dados ({dataset_name}): {variance}")

save_folder = 'Imagens_Analise_Densidade'
os.makedirs(save_folder, exist_ok=True)

for dataset_name, dataset in datasets.items():
    plot_density_analysis(dataset['data'], dataset_name, save_folder)