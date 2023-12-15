from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

def makePCA(matrix, pca_num):
    # matrix가 2차원보다 큰 경우에만 2차원으로 변환
    if len(np.array(matrix).shape) > 2:
        matrix = [item for sublist in matrix for item in sublist]
    
    n_components = pca_num  # 원하는 차원 수
    svd = TruncatedSVD(n_components=n_components)
    corpus_embeddings = svd.fit_transform(matrix)
    return corpus_embeddings

def makeclassinto(embeddings, classdf) :
    df_sembed = pd.DataFrame(embeddings)
    df_sembed["class"] = classdf
    return df_sembed

# For scaling
def min_max_scaler(x):
    if 'class' in x.columns:
        x_without_class = x.drop(columns=['class'])
        min_val = x_without_class.min()
        max_val = x_without_class.max()
        scaled_x = (x_without_class - min_val) / (max_val - min_val)
        scaled_x['class'] = x['class']  # Add the 'class' column back
        return scaled_x
    else:
        min_val = x.min()
        max_val = x.max()
        scaled_x = (x - min_val) / (max_val - min_val)
        return scaled_x

# For normalization
def zero_centered_scaling(x):
    if 'class' in x.columns:
        x_without_class = x.drop(columns=['class'])
        mean_val = x_without_class.mean()
        scaled_x = x_without_class - mean_val
        scaled_x['class'] = x['class']  # Add the 'class' column back
        return scaled_x
    else:
        mean_val = x.mean()
        scaled_x = x - mean_val
        return scaled_x

def show_dimension_graph(values):
    # df_sembed의 각 차원 값 추출
    dimension_values = values.drop(columns=['class']).values.T

    # 총 차원 수
    total_dimensions = len(dimension_values)

    # subplot의 행과 열 계산
    num_rows = (total_dimensions + 2) // 3  # 올림 연산을 사용하여 행 수 계산
    num_cols = min(3, total_dimensions)  # 최대 3개의 열을 가짐

    # 차원별로 히스토그램 시각화
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    for i, values in enumerate(dimension_values):
        row, col = divmod(i, num_cols)
        axs[row, col].hist(values, bins=30, color='skyblue', edgecolor='black')
        axs[row, col].set_title(f'Distribution of Dimension {i}')
        axs[row, col].set_xlabel(f'Dimension {i} Value')
        axs[row, col].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()