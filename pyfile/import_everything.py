import subprocess
import sys
import importlib
from IPython.display import display, Javascript

def installall():
    required_version_keras = "2.12.0"  # Main 코드를 동작시키기 위해서 keras의 버전이 필요합니다.
    required_libraries = ["emoji", "sentence_transformers", "transformers", 'pyod']  # Install이 필요한 라이브러리는 이 리스트 내부에 추가해주세요!

    # Install or upgrade additional libraries
    for library in required_libraries:
        try:
            importlib.import_module(library)
            print(f"{library} is already installed.")
        except ImportError:
            print(f"Installing {library}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])
            importlib.import_module(library)
            print(f"{library} has been installed.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {library}: {e}")

    # Install or upgrade Keras
    try:
        import keras
        installed_version_keras = keras.__version__

        if installed_version_keras != required_version_keras:
            print(f"Installing Keras version {required_version_keras}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"keras=={required_version_keras}"])
            print(f"Keras version {required_version_keras} has been installed.")
        else:
            print(f"Keras version {required_version_keras} is already installed.")
    except ImportError:
        print("Keras is not installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing Keras version {required_version_keras}: {e}")
        
    # Restart the Colab runtime
    display(Javascript('IPython.notebook.kernel.restart()'))

        
# 필요한 라이브러리들을 로드하는 함수
def load_libraries():
    import numpy as np
    import pandas as pd
    from bs4 import BeautifulSoup
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA, TruncatedSVD
    import statsmodels.api as sm
    from emoji import core
    import re
    import json
    from sklearn.feature_extraction.text import TfidfVectorizer
    import torch
    from transformers import ElectraModel, BertModel, BertTokenizer, AutoTokenizer, AutoModel
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import statistics
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.utils import class_weight
    from keras.models import Sequential, Model, load_model
    from keras.layers import LSTM, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Lambda
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, make_scorer, roc_curve, roc_auc_score
    from keras.callbacks import ModelCheckpoint, Callback
    from keras import backend as K
    from keras import regularizers
    from scipy.stats import mode
    from tqdm import tqdm
    from contextlib import redirect_stdout
    import os
    from keras.wrappers.scikit_learn import KerasClassifier
    import warnings

    # 모든 경고를 무시하려면
    warnings.filterwarnings("ignore")
    
    import_libraries=['np', 'pandas', 'bs4', 'sentence_transformers', 'sklearn.decomposition', 'sm', 
                      'emoji', 're', 'json', 'seaborn', 'matplotlib.pyplot', 'warnings', 'TfidfVectorizer',
                      'torch', 'ElectraModel', 'BertModel', 'BertTokenizer', 'AutoTokenizer', 'AutoModel',
                      'go', 'px', 'make_subplots', 'statistics', 'class_weight', 'Sequential', 'Model', 'load_model',
                      'LSTM', 'RepeatVector', 'TimeDistributed', 'Conv1D', 'MaxPooling1D', 'Flatten', 'Dense', 'Dropout', 'Input', 'Lambda', 'train_test_split', 'GridSearchCV',
                      'classification_report', 'confusion_matrix', 'accuracy_score', 'f1_score', 'make_scorer', 'roc_curve', 'roc_auc_score',
                      'ModelCheckpoint', 'Callback', 'backend', 'regularizers', 'mode', 'tqdm', 'redirect_stdout', 'os', 'KerasClassifier']
    
    # Install or upgrade additional libraries
    for library in import_libraries:
        print(f"Doing Import {library}...")

    return (np, pd, BeautifulSoup, SentenceTransformer, PCA, TruncatedSVD, sm, core, re, json, sns, plt, warnings, 
            TfidfVectorizer, torch, ElectraModel, BertModel, BertTokenizer, AutoTokenizer, AutoModel,
            go, px, make_subplots, statistics, class_weight, Sequential, Model, load_model, LSTM, RepeatVector, TimeDistributed, Conv1D, 
            MaxPooling1D, Flatten, Dense, Dropout, Input, Lambda,
            train_test_split, GridSearchCV, classification_report, confusion_matrix, accuracy_score, f1_score, make_scorer, roc_curve, roc_auc_score,
            ModelCheckpoint, Callback, K, regularizers, mode, tqdm, redirect_stdout, os, KerasClassifier)

