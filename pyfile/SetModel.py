from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, make_scorer, roc_curve, roc_auc_score, precision_score, recall_score
from keras.callbacks import Callback
import numpy as np
import pandas as pd
from pyod.models.xgbod import XGBOD
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.covariance import EllipticEnvelope
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, RepeatVector, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Lambda
from keras import regularizers
from tqdm import tqdm
from contextlib import redirect_stdout
import os
from scipy.stats import mode
import pickle
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from keras.wrappers.scikit_learn import KerasClassifier
    
class CustomDataset:
    def __init__(self, df, vectorname):
        self.dataset = []
        self.X = None
        self.y = None
        self.df = df
        self.vectorname = vectorname

    def makeDataset(self):
        # 데이터 불균형 해소 : smotetomek 기법 활용하여 Over/Under Sampling 동시 진행
        new_df_sembed = self.df.copy()
        self.y = new_df_sembed['class']
        self.X = new_df_sembed.drop(columns=['class'])
        smoteto = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
        self.X, self.y = smoteto.fit_resample(self.X, self.y)
        return self.X, self.y, new_df_sembed
    
    def splitdata(self):
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=96, shuffle=True, stratify=self.y)
        self.dataset=[X_train, X_test, y_train, y_test]
        return self.dataset
    
    def showtypes(self):
        # 데이터 분할
        X_train, X_test, y_train, y_test = self.dataset[0], self.dataset[1], self.dataset[2], self.dataset[3]
        print('X_train shape : ', X_train.shape)
        print('X_test shape : ', X_test.shape)
        print('y_train shape : ', y_train.shape)
        print('y_test shape : ', y_test.shape)
        
    def setWeights(self, weight_default=True, model=None, xy=None):
        class_weights = None
        
        # 가중치 그리드 설정
        weight_grid1 = np.arange(0, 6, 1)
        weight_grid2 = np.arange(0, 31, 1)
        
        if weight_default:
            class_weights = class_weight.compute_class_weight('balanced',
                                                    classes=np.unique(self.dataset[2]),
                                                    y=self.dataset[2])
        else:
            # True Positive(TP)와 True Negative(TN) 비율을 최적화하기 위한 가중치 찾기(너무 과적합되어 사용불가하고 위의 class_weights를 사용하도록 한다.)
            best_weights = None
            best_tp_tn_ratio = 0.0
            best_false_negatives = float('inf')
            best_false_positives = float('inf')

            total_iterations = len(weight_grid1) * len(weight_grid2)
            pbar = tqdm(total=total_iterations, desc="Grid Search Progress", position=0, leave=True)

            for w1 in weight_grid1:
                for w2 in weight_grid2:
                    current_weights = {0: w1, 1: w2}
                    # Redirect stdout to suppress output from model.predict
                    with redirect_stdout(open(os.devnull, "w")):
                        y_pred = model.predict(np.array(xy[0]))
                    # 평균 계산
                    average_value = np.mean(y_pred)

                    # Check if the most frequent value is greater than 0.5
                    mode_value, _ = mode(y_pred.flatten())
                    min_value = min(y_pred.flatten())
                    max_value = max(y_pred.flatten())
                    if min_value > 0:
                        if max_value >= 0.5:
                            average_value = 0.5

                    y_pred_binary = (y_pred > average_value).astype(int)

                    # Calculate TP, TN, FP, FN
                    cm = confusion_matrix(xy[1], y_pred_binary)
                    tn, fp, fn, tp = cm.ravel()

                    # Calculate TP/TN ratio (you can customize this ratio based on your preference)
                    tp_tn_ratio = (tp + tn) / (tp + tn + fp + fn)

                    if tp_tn_ratio >= best_tp_tn_ratio and fn <= best_false_negatives and fp <= best_false_positives:
                        best_tp_tn_ratio = tp_tn_ratio
                        best_false_negatives = fn
                        best_false_positives = fp
                        best_weights = current_weights

                    # tqdm 업데이트
                    pbar.update(1)

            # tqdm 닫기
            pbar.close()

            class_weights = best_weights
        
        return class_weights
    
    def show_weighted_vector_distribution(self):
        # x_train과 y_train을 데이터프레임으로 변환
        x_train_df = pd.DataFrame(self.dataset[0])
        y_train_df = pd.DataFrame({'Class': self.dataset[2]})

        # x_train과 y_train을 합친 데이터프레임
        combined_df = pd.concat([x_train_df, y_train_df], axis=1)

        # 클래스 별 가중치 적용 후의 데이터 계산
        weights_dict = dict(enumerate(self.setWeights()))
        weighted_vectors = self.dataset[0] * (np.array([weights_dict[y] for y in np.array(self.dataset[2])]).reshape(-1, 1))

        # Convert weighted_vectors to a DataFrame with the same column names
        weighted_df = pd.DataFrame(weighted_vectors, columns=x_train_df.columns)
        
        # x_train과 y_train을 합친 데이터프레임
        combined_w_df = pd.concat([weighted_df, y_train_df], axis=1)

        # 시각화
        fig, axs = plt.subplots(self.dataset[0].shape[1], 2, figsize=(15, 20))

        for i in range(self.dataset[0].shape[1]):
            #가중치 적용 전
            sns.histplot(combined_df.loc[:, i][combined_df['Class'] == 0], bins=50, kde=False, color='skyblue', label='Class 0', ax=axs[i, 0])
            sns.histplot(combined_df.loc[:, i][combined_df['Class'] == 1], bins=50, kde=False, color='orange', label='Class 1', ax=axs[i, 0])
            axs[i, 0].set_title(f'Original Dimension {i} Distribution')
            axs[i, 0].set_xlabel(f'Dimension {i} Value')
            axs[i, 0].set_ylabel('Count')
            axs[i, 0].legend()

            # 가중치 적용 후
            sns.histplot(combined_w_df.loc[:, i][combined_w_df['Class'] == 0], bins=50, kde=False, color='skyblue', label='Class 0', ax=axs[i, 1])
            sns.histplot(combined_w_df.loc[:, i][combined_w_df['Class'] == 1], bins=50, kde=False, color='orange', label='Class 1', ax=axs[i, 1])
            axs[i, 1].set_title(f'Weighted Dimension {i} Distribution')
            axs[i, 1].set_xlabel(f'Dimension {i} Value')
            axs[i, 1].set_ylabel('Count')
            axs[i, 1].legend()

        plt.tight_layout()
        plt.show()


class Utility:
    def __init__(self, types, datasetsplit_list):
        self.types = types
        self.ds_list = datasetsplit_list

    def getXY(self):
        if self.types == 'kce':
            return self.ds_list[0]
        elif self.types == 'kce':
            return self.ds_list[1]
        else:
            return self.ds_list[2]

class SaveBestModelForMinorityClass(Callback):
    def __init__(self, model, name, types, x_val, y_val):
        super(SaveBestModelForMinorityClass, self).__init__()
        self.model = model
        self.name = name
        self.types = types
        self.x_val = x_val
        self.y_val = y_val
        self.best_false_negatives = float('inf')
        self.best_false_positives = float('inf')
        self.best_model = None

    def on_epoch_end(self, epoch, logs=None):
        jss_route='/content/drive/MyDrive/papers'
        y_pred = self.model.predict(self.x_val)
        # 평균 계산
        average_value = np.mean(y_pred)
        min_value = min(y_pred.flatten())
        max_value = max(y_pred.flatten())
        if min_value > 0:
            if max_value >= 0.5:
                average_value = 0.5
        y_pred_binary = (y_pred > average_value).astype(int)
        cm = confusion_matrix(self.y_val, y_pred_binary)
        false_negatives = cm[1][0]  # False Negatives
        false_positives = cm[0][1]  # False Positives

        if false_negatives <= self.best_false_negatives and false_positives <= self.best_false_positives:
            self.best_false_negatives = false_negatives
            self.best_false_positives = false_positives
            self.best_model = self.model
            self.best_model.save(f'{jss_route}/jss_models/best_model_{self.name}_{self.types}.h5', overwrite=True)

class ModelMaker:
    def __init__(self, modelname, types, datasetsplit_list, classweight_list, dataset_list, original_dim=5, intermediate_dim=64, latent_dim=2):
        self.modelname = modelname
        self.types = types
        self.ds_list = datasetsplit_list
        self.cw_list = classweight_list
        self.data_list = dataset_list
        self.ds_in_list = []
        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None
        self.original_dim = original_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.best_model=None

    def getXY(self):
        self.ds_in_list=Utility(self.types, self.ds_list).getXY()
        self.X_train, self.X_test, self.y_train, self.y_test = self.ds_in_list[0], self.ds_in_list[1], self.ds_in_list[2], self.ds_in_list[3]
    
    def getDataset(self):
        if self.types == 'kce':
            return self.data_list[0]
        elif self.types == 'kce':
            return self.data_list[1]
        else:
            return self.data_list[2]

    def getWeights(self, weights_in, model):
        if weights_in:
            if self.types == 'kce':
                return self.cw_list[0]
            elif self.types == 'kce':
                return self.cw_list[1]
            else:
                return self.cw_list[2]
        else:
            xy=[self.X_train, self.y_train]
            #model = KerasClassifier(build_fn=model, epochs=30, batch_size=64)
            weights = CustomDataset(self.getDataset(), self.types).setWeights(False, model, xy)
            return weights

    # Shallow CNN 모델 생성
    def create_shallow_cnn_model(self):
        shallow_cnn = Sequential()
        shallow_cnn.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(self.X_train.shape[1], 1)))
        shallow_cnn.add(MaxPooling1D(pool_size=1))
        shallow_cnn.add(Flatten())
        shallow_cnn.add(Dense(10, activation='relu'))
        shallow_cnn.add(Dense(1, activation='sigmoid'))
        return shallow_cnn

    # Moderate CNN 모델 생성
    def create_moderate_cnn_model(self):
        moderate_cnn = Sequential()
        moderate_cnn.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(self.X_train.shape[1], 1)))
        moderate_cnn.add(MaxPooling1D(pool_size=1))
        moderate_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        moderate_cnn.add(MaxPooling1D(pool_size=1))
        moderate_cnn.add(Flatten())
        moderate_cnn.add(Dense(64, activation='relu'))
        moderate_cnn.add(Dense(1, activation='sigmoid'))
        return moderate_cnn

    # Deep CNN 모델 생성
    def create_deep_cnn_model(self):
        # Deep CNN 모델 생성 (4개의 컨볼루션 레이어)
        deep_cnn = Sequential()
        deep_cnn.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(self.X_train.shape[1], 1)))
        deep_cnn.add(MaxPooling1D(pool_size=1))

        deep_cnn.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
        deep_cnn.add(MaxPooling1D(pool_size=1))

        deep_cnn.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
        deep_cnn.add(MaxPooling1D(pool_size=1))


        deep_cnn.add(Flatten())

        deep_cnn.add(Dense(64, activation='relu'))
        deep_cnn.add(Dense(1, activation='sigmoid'))
        return deep_cnn
    
    # EllipticEnvelope
    def create_elliptic_model(self):
        model = EllipticEnvelope(contamination=0.5, random_state = 96)
        return model
    
    # XGBOD
    def create_xgbod_model(self):
        model = XGBOD(random_state = 96)
        return model
        
    
    # DevNet 모델 생성
    def create_devnet_model(self):
        # 아래는 구현전인 모델레이어입니다.
        x_input = Input(shape=(self.X_train.shape[1],))
        intermediate = Dense(128, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
        intermediate = Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01), name = 'hl2')(intermediate)
        intermediate = Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01), name = 'hl3')(intermediate)
        intermediate = Dense(1, activation='linear', name = 'score')(intermediate)
        return Model(x_input, intermediate)
    
    def getmodel(self):
        if self.modelname == 'shallow_cnn':
            model=self.create_shallow_cnn_model()
            return model
        elif self.modelname == 'moderate_cnn':
            model=self.create_moderate_cnn_model()
            return model
        elif self.modelname == 'deep_cnn':
            model=self.create_deep_cnn_model()
            return model
        elif self.modelname == 'ee':
            model=self.create_elliptic_model()
            return model
        elif self.modelname == 'xgbod':
            model=self.create_xgbod_model()
            return model
        else:
            model=self.create_devnet_model()
            return model

    def get_pred_label(self, model_pred):
        # create_elliptic_model 모델 출력 (1:정상, -1:불량(사기)) 이므로 (0:정상, 1:불량(사기)로 Label변환)
        model_pred = np.where(model_pred == 1, 0, model_pred)
        model_pred = np.where(model_pred == -1, 1, model_pred)
        return model_pred


    def fitlogic(self):
        jss_route='/content/drive/MyDrive/papers'
        self.getXY()
        model=self.getmodel()
        if self.modelname != 'ee' and self.modelname != 'xgbod':
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            checkpoint_sc=SaveBestModelForMinorityClass(model,self.modelname, self.types, self.X_test, self.y_test)
            weights=dict(enumerate(self.getWeights(True, None)))
            #weights=dict(enumerate(self.getWeights(False, model)))
            print('Grid Search Best Weights : ', weights)
            model.fit(self.X_train, self.y_train, epochs=50, batch_size=64, class_weight=weights, callbacks=[checkpoint_sc])
        else:
            if self.modelname == 'xgbod':
                model.fit(self.X_train, self.y_train)
            else:
                model.fit(self.X_train)
            # 모델 저장
            with open(f'{jss_route}/jss_models/best_model_{self.modelname}_{self.types}.pkl', 'wb') as file:
                pickle.dump(model, file)
            
    def predictlogic(self):
        jss_route='/content/drive/MyDrive/papers'
        # Load the best model
        if self.modelname != 'ee' and self.modelname != 'xgbod':
            self.best_model = load_model(f'{jss_route}/jss_models/best_model_{self.modelname}_{self.types}.h5')
            # Evaluate the best model
            y_pred = self.best_model.predict(self.X_test)
        else:
            # 모델 로드
            with open(f'{jss_route}/jss_models/best_model_{self.modelname}_{self.types}.pkl', 'rb') as file:
                self.best_model = pickle.load(file)
            if self.modelname != 'xgbod':
                y_pred = self.best_model.predict(self.X_test)
                y_pred = self.get_pred_label(y_pred)
            else:
                y_pred = self.best_model.decision_function(self.X_test)
        
        
        # 평균 계산
        average_value = np.mean(y_pred)

        # Check if the most frequent value is greater than 0.5
        mode_value, _ = mode(y_pred.flatten())
        min_value = min(y_pred.flatten())
        max_value = max(y_pred.flatten())
        if min_value > 0:
            if max_value >= 0.5:
                average_value = 0.5

        y_pred_binary = (y_pred > average_value).astype(int)

        # 혼동 행렬 출력
        cm = confusion_matrix(self.y_test, y_pred_binary)  # 테스트 세트에 대한 혼동 행렬 생성
        # 혼동 행렬을 히트맵으로 시각화
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # Additional evaluation metrics
        accuracy = accuracy_score(self.y_test, y_pred_binary)
        precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
        recall = recall_score(self.y_test, y_pred_binary)
        f1 = f1_score(self.y_test, y_pred_binary)
        tpr = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # True Negative Rate
        fpr = cm[0, 1] / (cm[0, 0] + cm[0, 1])  # False Positive Rate

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall(TNR):", recall)
        print("F1 Score:", f1)
        print("True Positive Rate (TPR):", tpr)
        print("False Positive Rate (FPR):", fpr)

        # Compute AUROC
        auroc = roc_auc_score(self.y_test, y_pred)

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='AUROC = {:.3f}'.format(auroc))
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

        self.getResult(y_pred, y_pred_binary)
    
    def getResult(self, y_pred, y_pred_binary):
        result_cnn=self.makeplothue(self.getDataset(), y_pred, y_pred_binary)
        fig = px.histogram(result_cnn, x="y_scores")
        fig.update_traces(marker_line_color='black', marker_line_width=1)
        fig.update_layout(
            title="Count of y_scores",
            xaxis_title="y_scores",
            yaxis_title="Count"
        )
        fig.update_layout(bargap=0.1)  # 각 막대 사이의 간격 조정
        fig.show()

    

    def makeplothue(self, df, y_pred, y_pred_binary):
        pd.options.display.float_format = '{:.5f}'.format
        save_col = df.columns
        new_cols = save_col.drop('class')
        X_test_2d = self.X_test.values.reshape(self.X_test.shape[0], -1)  # DataFrame을 Numpy 배열로 변환하고 2D로 변경
        dfname = pd.DataFrame(X_test_2d, columns=new_cols)  # Pandas DataFrame으로 변환
        # 'y_scores'와 'y_hue' 컬럼이 이미 존재한다면 제거합니다.
        dfname = dfname.drop(['y_scores', 'y_hue'], axis=1, errors='ignore')

        dfname['y_scores'] = y_pred.astype(float)
        dfname['y_hue'] = y_pred_binary

        return dfname