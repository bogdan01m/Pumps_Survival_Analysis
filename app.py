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
import os
from scipy.fft import fft
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


device=torch.device('cpu')


####################################################################
#здесь логика загрузки данных 
####################################################################    
def prepare_data(data):
    df=data[['ReservoirTemperature_c', 'MeasureMRM204', 'MeasureMRM205',
       'ProducingGOR_m3_t', 'LiquidViscosity', 'WeightedParticlesFactor_mg_l',
       'MeasureMRM187', 'MeasureMRM188', 'MeasureMRM12', 'MeasureMRM30',
       'MeasureMRM143', 'MeasureMRM144']]
    data_fft = fft(df)
    # Создаем экземпляр MinMaxScaler
    scaler = MinMaxScaler()
    columns=df.columns
    # Нормализуем данные в датафре  йме X
    data_fft = pd.DataFrame(scaler.fit_transform(data_fft.real), columns=columns)
    return data_fft, columns
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

args.device = torch.device('cpu')

# ===== data loading ==== #
args.batch_size = 1

# ==== model capacity ==== #
args.n_layers = 1
args.embedding_dim = 256


# ==== regularization ==== #
# args.dropout = 0  # Установка значения dropout
args.use_bn = False  # batch normalization

# ==== optimizer & training  # ====
args.lr = 0.001
args.epoch = 100
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
    st.header('Данные')
    st.write(dataframe)

def stats(dataframe):
    st.header('Статистика')
    st.write(dataframe.describe())    
    
    
def interactiveplot(data):
    x_axis_val = st.selectbox('Select X-Axis Value', options=data.columns)
    y_axis_val = st.selectbox('Select Y-Axis Value', options=data.columns)
    col=st.color_picker('Выберите цвет')
    
    plot= px.scatter(data, x=x_axis_val, y=y_axis_val)       
    plot.update_traces(marker=dict(color=col))
    st.plotly_chart(plot)

    
def threshold_plot(score):
    sns.set(style='dark')
    score.plot(logy=True, figsize=(16,9), color=['blue','red'],linewidth=2.5)
    # plt.savefig('score_plot.png')
    plt.grid()
    
def preds_plot(data_loader, model, columns):
    combo_data_np = torch.cat([batch for batch in data_loader], dim=0).detach().cpu().numpy()

    # Перестроить данные в подходящий формат для построения графиков
    combo_data_np = combo_data_np.reshape(-1, seq_len, n_features)

    # Получить предсказания модели
    preds = []
    with torch.no_grad():
        model = model.eval()
        for seq_true in data_loader:
            seq_true = seq_true.to(args.device)
            seq_true = seq_true.reshape((-1, seq_len, n_features))

            seq_pred = model(seq_true)
            preds.append(seq_pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    # Создать подграфики для каждой фичи
    fig, axs = plt.subplots(n_features, 1, figsize=(10, n_features * 3))

    # Построить график для каждой фичи
    for i in range(n_features):
        line1, = axs[i].plot(combo_data_np[:, 0, i], color='blueviolet', label='original data')
        line2, = axs[i].plot(preds[:, 0, i], color='orange', label='predictions')
        for j in range(1, seq_len):
            axs[i].plot(combo_data_np[:, j, i], color='blueviolet')
            axs[i].plot(preds[:, j, i], color='orange')
        
        # Использовать названия колонн вместо номеров фич
        axs[i].set_title(columns[i])
        axs[i].legend(handles=[line1, line2])

    plt.tight_layout()
    plt.show()






   
st.title("DPump")
st.markdown('## Искуственный интеллект в предсказании выбытия УЭЦН ')
st.markdown('Модель решает задачу детектирования аномалий во временных рядах')
st.markdown('Что значит аномалия?')
st.markdown('**Аномалия- последние 15 дней работы УЭЦН**')
st.markdown('Такой подход позволяет явно и эффективно реагировать на возможные отклонения и принимать решение о дальнейшей судьбе УЭЦНа')
st.markdown('Как выглядят аномальные данные?')
# Загрузка изображения
image = 'anomalies_and_clean_data.png'

# Отображение изображения
st.image(image, caption='График аномалий и чистых данных', use_column_width=True)
st.markdown('Наша нейросеть позволяет детектировать аномалии исходя из данных телеметрии')
st.markdown('Ввиду особенностей модели, возможны ложные срабатывания при изменении ваших показателей')

st.markdown('**Метрики качества модели:**')
metrics=pd.read_csv('metrics_test_fft_mse_seq=1.csv')
st.write(metrics) 


st.sidebar.title('Навигация')

uploaded_file=st.sidebar.file_uploader('Загрузите ваш файл')

options= st.sidebar.radio('Опции:',
options=['Главная',
'Статистика',
'Интерактивный график данных',
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
    elif options =='Интерактивный график данных':
        st.markdown('**Построение графиков на основе данных**')
        interactiveplot(data)            
 
   
    # elif options =='Гистограмма функции потерь':
    #     st.markdown('**Изменение функции потерь со временем**')
    #     graph = sns.displot(x=losses, bins=50, kde=True) 
    #     hist(graph)  
    # elif options == 'Скоринг':   
  
    elif options =='Получить предсказание':
        st.markdown('ВНИМАНИЕ: данные, передаваемые в модель должны быть без пропусков')
        st.markdown('**график детектирования аномалий**')
        data_fft, columns=prepare_data(data)  
        data_to_tensor = Create_dataset(data_fft)
        data_loader=data_to_tensor.dataset
        seq_len = data_to_tensor.seq_len
        n_features = data_to_tensor.n_features
        model = torch.load('lstmae_mse_fft_seq=1.pth')
        model = model.to(torch.device('cpu'))
        
        predictions, losses=predict(model, data_loader)
        st.markdown('**Выберите пороговое значение**')
        Threshold = st.slider('Пороговое значение', min_value=0.0, max_value=1.0, value=0.0005, step=0.0002, format='%.4f')
        Threshold = st.number_input('Введите пороговое значение вручную', min_value=0.0, max_value=1.0, value=Threshold, format='%.4f')
        score = pd.DataFrame()
        score['Loss'] = losses
        score['Threshold'] = Threshold
        score['Anomaly'] = score['Loss'] > score['Threshold']
        threshold_plot(score)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        # st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown('**результат детектирования аномалий**')
        
        show_data(score) 
        st.markdown('**Восстановленные данные**')  
        
        st.pyplot(preds_plot(data_loader, model, columns)) 
