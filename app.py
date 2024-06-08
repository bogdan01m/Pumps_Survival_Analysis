import streamlit as st
import torch
from torch.utils.data import Dataset, DataLoader
import copy
import plotly_express as px
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
import argparse
import time
import os
from scipy.fft import fft, ifft
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


####################################################################
#Fast Fourier Tranform
####################################################################  

def apply_fft(data, columns):
    # Применяем FFT только к указанным столбцам
    fft_features = {}
    for column in columns:
        if column in data.columns:
            # Преобразуем данные в числовой тип, игнорируя ошибки
            numeric_data = pd.to_numeric(data[column], errors='coerce').fillna(0).to_numpy()
            if numeric_data.ndim == 1:  # Проверяем, что это одномерный массив
                fft_result = fft(numeric_data)
                # Извлекаем реальные и мнимые части
                fft_real = np.real(fft_result)
                fft_imag = np.imag(fft_result)
                # Добавляем результаты FFT в словарь
                fft_features[f'{column}_fft_real'] = fft_real
                fft_features[f'{column}_fft_imag'] = fft_imag
    return pd.DataFrame(fft_features)

####################################################################
#здесь логика загрузки данных 
####################################################################    
def prepare_data(data):

    

    fft_columns = ['MeasureMRM12', 'MeasureMRM142', 'MeasureMRM143', 'MeasureMRM187', 
               'MeasureMRM188', 'MeasureMRM219', 'MeasureMRM204']

    df_fft = apply_fft(data, fft_columns)
    df_fft=df_fft[['MeasureMRM12_fft_real',
       'MeasureMRM12_fft_imag', 'MeasureMRM142_fft_real',
       'MeasureMRM142_fft_imag', 'MeasureMRM143_fft_real',
       'MeasureMRM143_fft_imag', 'MeasureMRM187_fft_real',
       'MeasureMRM187_fft_imag', 'MeasureMRM188_fft_real',
       'MeasureMRM188_fft_imag', 'MeasureMRM219_fft_real',
       'MeasureMRM219_fft_imag', 'MeasureMRM204_fft_real',
       'MeasureMRM204_fft_imag']]
    # Создаем экземпляр MinMaxScaler
    columns_fft=df_fft.columns
    scaler = MinMaxScaler()
    
    df_fft = pd.DataFrame(scaler.fit_transform(df_fft),columns=columns_fft)
    
    # Нормализуем данные в датафре  йме X
    
    return df_fft, columns_fft


class Create_dataset(Dataset):
    def __init__(self, data, seq_len=1, batch_size=1):
        self.data = data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.n_samples, self.n_features = self.data.shape
        self.dataset = self.create_dataset(self.data, batch_size)

    def create_dataset(self, data, batch_size):
        sequences = []
        for i in range(len(data) - self.seq_len + 1):
            sequence = data.iloc[i:i+self.seq_len].values.astype(np.float32)
            sequences.append(torch.from_numpy(sequence))
        return DataLoader(sequences, batch_size=batch_size, shuffle=False)

    def __getitem__(self, index):
        return self.data.iloc[index]

    def __len__(self):
        return len(self.data) - self.seq_len + 1

####################################################################
#Здесь модель
####################################################################


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, args):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.hidden_dim = 2 * args.embedding_dim
        self.lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=args.n_layers,
            batch_first=True,
            dtype=torch.float32
        ).to(args.device)
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=args.embedding_dim,
            num_layers=args.n_layers,
            batch_first=True,
            dtype=torch.float32
        ).to(args.device)

    def forward(self, x):
        x = x.to(args.device).to(torch.float32)
        batch_size = x.size(0)

        # print(f'ENCODER input dim: {x.shape}')
        x = x.reshape((batch_size, self.seq_len, self.n_features))
        # print(f'ENCODER reshaped dim: {x.shape}')
        x, (_, _) = self.lstm1(x)
        # print(f'ENCODER output lstm1 dim: {x.shape}')
        x, (hidden_n, _) = self.lstm2(x)
        # print(f'ENCODER output lstm2 dim: {x.shape}')
        # print(f'ENCODER hidden_n lstm2 dim: {hidden_n.shape}')
        # print(f'ENCODER hidden_n wants to be reshaped to : {(batch_size, args.embedding_dim)}')
        return hidden_n.reshape((batch_size, args.embedding_dim))

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, args):
        super(Decoder, self).__init__()
        self.seq_len, self.input_dim = seq_len, args.embedding_dim
        self.hidden_dim, self.n_features = 2 * args.embedding_dim, n_features
        self.lstm1 = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.embedding_dim,
            num_layers=1,
            batch_first=True
        ).to(args.device)
        self.lstm2 = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        ).to(args.device)
        self.output_layer = nn.Linear(self.hidden_dim, n_features).to(args.device)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(args.device)
        # print(f'DECODER input dim: {x.shape}')
        x = x.repeat_interleave(self.seq_len, dim=0)
        # print(f'DECODER repeat dim: {x.shape}')
        x = x.reshape((batch_size, self.seq_len, self.input_dim))
        # print(f'DECODER reshaped dim: {x.shape}')
        x, (hidden_n, cell_n) = self.lstm1(x)
        # print(f'DECODER output rnn1 dim:/ {x.shape}')
        x, (hidden_n, cell_n) = self.lstm2(x)
        x = x.reshape((batch_size, self.seq_len, self.hidden_dim))
        return self.output_layer(x)

class LSTM_AUTO_ENCODER(nn.Module):
    def __init__(self, seq_len, n_features, args):
        super(LSTM_AUTO_ENCODER, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.encoder = Encoder(seq_len, n_features, args)
        self.decoder = Decoder(seq_len, n_features, args)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args('')

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== data loading ==== #
args.batch_size = 1

# ==== model capacity ==== #
args.n_layers = 1
args.embedding_dim = 128


# ==== regularization ==== #
# args.dropout = 0  # Установка значения dropout
args.use_bn = False  # batch normalization

# ==== optimizer & training  # ====
args.lr = 0.001
args.epoch = 180
####################################################################
#Предсказание модели
####################################################################



def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.MSELoss(reduction='sum').to(args.device)

    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(device=args.device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses
####################################################################
#здесь веб-приложение
####################################################################


def show_data(dataframe):
    # st.header('Данные')
    st.write(dataframe)

def stats(dataframe):
    st.header('Статистика')
    st.write(dataframe.describe())    
    
    

def threshold_plot(score):
    # Создаем трассы для графика
    trace_loss = go.Scatter(
        x=score.index,
        y=score['Loss'],
        mode='lines',
        name='Loss',
        line=dict(color='blue')
    )
    
    trace_threshold = go.Scatter(
        x=score.index,
        y=[score['Threshold'].iloc[0]] * len(score.index),
        mode='lines',
        name='Threshold',
        line=dict(color='red', dash='dash')
    )
    
    trace_anomalies = go.Scatter(
        x=score[score['Anomaly']].index,
        y=score[score['Anomaly']]['Loss'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='darkorange', size=10)
    )
    
    # Собираем все трассы в один график
    fig = go.Figure(data=[trace_loss, trace_threshold, trace_anomalies])
    
    # Обновляем макет графика
    fig.update_layout(
        title='График потерь и порогового значения',
        xaxis_title='Индекс',
        yaxis_title='Значение',
        yaxis_type='log',
        legend_title='Легенда'
    )
    
    # Отображаем график в Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
def preds_plot_interactive(data_loader, model, columns, seq_len, n_features, device):
    combo_data_np = torch.cat([batch for batch in data_loader], dim=0).detach().cpu().numpy()
    combo_data_np = combo_data_np.reshape(-1, seq_len, n_features)

    preds = []
    with torch.no_grad():
        model = model.eval()
        for seq_true in data_loader:
            seq_true = seq_true.to(device)
            seq_true = seq_true.reshape((-1, seq_len, n_features))

            seq_pred = model(seq_true)
            preds.append(seq_pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    # Создаем макет с подграфиками
    fig = make_subplots(rows=n_features, cols=1, subplot_titles=columns)

    # Добавляем графики для каждой фичи
    for i in range(n_features):
        fig.add_trace(go.Scatter(
            x=np.arange(combo_data_np.shape[0]),
            y=combo_data_np[:, :, i].flatten(),
            mode='lines',
            name='Original',
            line=dict(color='darkgoldenrod'),
            showlegend=i == 0  # Показываем легенду только для первого графика
        ), row=i+1, col=1)

        fig.add_trace(go.Scatter(
            x=np.arange(preds.shape[0]),
            y=preds[:, :, i].flatten(),
            mode='lines',
            name='Prediction',
            line=dict(color='blueviolet'),
            showlegend=i == 0  # Показываем легенду только для первого графика
        ), row=i+1, col=1)

    # Обновляем общий макет
    fig.update_layout(height=300 * n_features, showlegend=True, title_text='Оригинальные данные и востановленные моделью')

    # Отображаем график в Streamlit
    st.plotly_chart(fig, use_container_width=True)





   
st.title("Pumps Survival Analysis")


st.sidebar.title('Навигация')

uploaded_file=st.sidebar.file_uploader('Загрузите ваш файл')

options= st.sidebar.radio('Опции:',
options=['Главная',
'Статистика',
'Получить предсказание',
])
if uploaded_file:
    #считываем данные 
    delimiter_option = st.selectbox(
        'Выберите разделитель для вашего csv-файла:',
        options=[',', ';', '\t', '|', ' '],  # Список возможных разделителей
        index=0  # Индекс выбранного по умолчанию разделителя
    )

    # Чтение данных с выбранным разделителем
    data = pd.read_csv(uploaded_file, low_memory=False, delimiter=delimiter_option)
    st.markdown('**Ваши данные**')
    show_data(data) 
        # Получить seq_len и n_features из первого элемента dataset


       

    if options =='Статистика':
        st.markdown('**Ваша статистика**')
        stats(data)
         
 
   
    # elif options =='Гистограмма функции потерь':
    #     st.markdown('**Изменение функции потерь со временем**')
    #     graph = sns.displot(x=losses, bins=50, kde=True) 
    #     hist(graph)  
    # elif options == 'Скоринг':   
  
  
    elif options =='Получить предсказание':
        st.markdown('ВНИМАНИЕ: данные, передаваемые в модель должны быть без пропусков')
        st.markdown('**График разделения аномальных и нормальных значений**')
        data_fft, columns=prepare_data(data) 
        # show_data(data_fft)  
        data_to_tensor = Create_dataset(data_fft)
        data_loader=data_to_tensor.dataset
        seq_len = data_to_tensor.seq_len
        n_features = data_to_tensor.n_features
        model = torch.load('lstmae_fft_only.pth', map_location=args.device)
        
        
        predictions, losses=predict(model, data_loader)
        st.markdown('**Выберите пороговое значение**')
        Threshold=0.003
        # visual_threshold = Threshold * 1e+1
        Threshold = st.slider('Пороговое значение', min_value=0.0, max_value=1.0, value=Threshold, step=1e-6, format='%.6f')
        Threshold = st.number_input('Введите пороговое значение вручную', min_value=0.0, max_value=1.0, value=Threshold, format='%.6f')
        # Threshold /= 1e+1
        score = pd.DataFrame()
        score['Loss'] = losses
        score['Threshold'] = Threshold
        score['Anomaly'] = score['Loss'] > score['Threshold']
        
        # Объединяем 'score' с 'data', помещая 'score' слева от последней колонки в 'data'
        result = pd.concat([score, data], axis=1)
        threshold_plot(score)
        # st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown('**Результат детектирования аномалий**')
        
        show_data(result) 
        st.markdown('**Графики параметров (оригинал и восстановленные)**')  
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(preds_plot_interactive(data_loader, model, columns, seq_len, n_features, device))
        
