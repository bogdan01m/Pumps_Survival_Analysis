import streamlit as st
import pandas as pd
import numpy as np
import plotly_express as px
import torch
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
import argparse
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')



map_location=torch.device('cpu')


####################################################################
#здесь логика загрузки данных 
####################################################################    

def create_dataset(dataframe):
    df=dataframe[['ReservoirTemperature_c', 'MeasureMRM204', 'MeasureMRM205',
       'ProducingGOR_m3_t', 'LiquidViscosity', 'MeasureMRM219',
       'WeightedParticlesFactor_mg_l', 'MeasureMRM187', 'MeasureMRM188',
       'MeasureMRM12', 'MeasureMRM30', 'MeasureMRM143', 'MeasureMRM144', ]]
    
    # # Создаем экземпляр MinMaxScaler
    scaler = MinMaxScaler()

    # Извлекаем названия колонок
    columns = df.columns

    # Нормализуем данные в датафрейме X
    series_normalized = pd.DataFrame(scaler.fit_transform(df), columns=columns)
   
    sequences = series_normalized.astype(np.float32).to_numpy().tolist()
    dataset=[torch.tensor(s).unsqueeze(1) for s in sequences]
    n_seq, seq_len, n_features=torch.stack(dataset).shape
    return dataset, seq_len, n_features, series_normalized

####################################################################
#Здесь модель
####################################################################


class Encoder(nn.Module):
    def __init__(self, seq_len, n_features):
        super(Encoder, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = 2 * args.embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=args.n_layers,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=args.embedding_dim,
            num_layers=args.n_layers,
            batch_first=True
        )

    def forward(self, x):
        x, (_, _) = self.rnn1(x)
        x, (hidden_n, _) = self.rnn2(x)
        return hidden_n  # Возвращаем hidden_n без изменения формы

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.hidden_dim = 2 * args.embedding_dim
        self.n_features = n_features

        self.rnn1 = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.embedding_dim,
            num_layers=args.n_layers,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=args.n_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        # x shape (batch_size, embedding_dim)
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # Добавляем измерение seq_len и повторяем x seq_len раз
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape(-1, self.hidden_dim)  # Изменяем форму x на (batch_size * seq_len, hidden_dim)
        x = self.output_layer(x)
        return x.view(-1, self.seq_len, self.n_features)  # Возвращаем x в форме (batch_size, seq_len, n_features)

class LSTM_AUTO_ENCODER(nn.Module):
    def __init__(self, seq_len, n_features):
        super(LSTM_AUTO_ENCODER, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = args.embedding_dim  # Добавляем атрибут embedding_dim

        self.encoder = Encoder(seq_len, n_features).to(args.device)
        self.decoder = Decoder(seq_len, n_features).to(args.device)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.embedding_dim)  # Изменяем форму x на (batch_size, embedding_dim)
        x = self.decoder(x)
        return x

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

parser = argparse.ArgumentParser()
args = parser.parse_args('')

args.device = 'cuda' if torch.cuda.is_available else 'cpu'

# ===== data loading ==== #
args.batch_size = 4

# ==== model capacity ==== #
args.n_layers = 1
args.embedding_dim = 128


# ==== regularization ==== #
args.dropout = 0
args.use_bn = False

# ==== optimizer & training  # ====
args.lr = 0.001
args.epoch = 180
####################################################################
#Предсказание модели
####################################################################



def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(args.device)

    with torch.no_grad():
        model = model.eval()
        for seq_true in dataset:
            seq_true = seq_true.to(args.device)
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
    score.plot(logy=True, figsize=(16,9), ylim=[1e-2, 1e2], color=['blue','red'],linewidth=2.5)
    # plt.savefig('score_plot.png')
    plt.grid()
    
    
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
st.markdown('Физические процессы и датчики- основа в отобре параметров')
st.markdown('Ввиду особенностей модели, возможны ложные сробатывания при изменении ваших показателей')

st.markdown('**Метрики качества модели:**')
metrics=pd.read_csv('metrics.csv')
st.write(metrics) 
st.markdown('ВНИМАНИЕ: данные, передаваемые в модель должны быть без пропусков')

st.sidebar.title('Навигация')

uploaded_file=st.sidebar.file_uploader('Загрузите ваш файл')

options= st.sidebar.radio('Страницы ',
options=['Главная',
'Данные', 
'Статистика',
'Интерактивный график данных',
'Данные для модели',
'Получить предсказание',
])
if uploaded_file:
    #считываем данные 

    data=pd.read_csv(uploaded_file, low_memory=False)

        # Получить seq_len и n_features из первого элемента dataset


    



    # строим фигурку
        
    if options == 'Данные':   
        st.markdown('**Ваши данные**')
        show_data(data) 
    elif options =='Статистика':
        st.markdown('**Ваша статистика**')
        stats(data)
    elif options =='Интерактивный график данных':
        st.markdown('**Построение графиков на основе данных**')
        interactiveplot(data)            
    elif options == 'Данные для модели': 
        dataset, seq_len, n_features, df=create_dataset(data)  
        st.markdown('**Данные для модели**')
        show_data(df)   
   
    # elif options =='Гистограмма функции потерь':
    #     st.markdown('**Изменение функции потерь со временем**')
    #     graph = sns.displot(x=losses, bins=50, kde=True) 
    #     hist(graph)  
    # elif options == 'Скоринг':   
  
    elif options =='Получить предсказание':

        st.markdown('**график детектирования аномалий**')
        dataset, seq_len, n_features, df=create_dataset(data)

        model = torch.load('DPump.pth')
        model = model.to(args.device)
        predictions, losses=predict(model, dataset)
        Threshold = st.slider('Пороговое значение', min_value=0.0, max_value=1.0, value=0.03, step=0.01)
        score = pd.DataFrame(index = data.index)
        score['Loss'] = losses
        score['Threshold'] = Threshold
        score['Anomaly'] = score['Loss'] > score['Threshold']
        threshold_plot(score)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        # st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown('**результат детектирования аномалий**')
        
        show_data(score)   
