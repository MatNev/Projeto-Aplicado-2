
# File com os códigos usados no Projeto.



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import unidecode
import googlemaps
from openpyxl import Workbook
import folium
import geopandas as gpd
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune.registry import register_env
from gym import Env
from gym.spaces import Discrete, Box
import networkx as nx
import osmnx as ox


# Funcções Gerais

def valida_lat_long(latitude, longitude):

    lat_regex = r"^(-?\d{1,2})(\.\d+)?$"

    long_regex = r"^(-?\d{1,3})(\.\d+)?$"

    if not re.match(lat_regex, str(latitude)):
        return False

    if not re.match(long_regex, str(longitude)):
        return False

    if float(latitude) < -90 or float(latitude) > 90 or float(latitude) == 0:
        return False

    if float(longitude) < -180 or float(longitude) > 180 or float(longitude) == 0:
        return False
    return True



def valida_formato_hora(hora):
    """
    Valida se o valor de hora está no formato HH:MM:SS.

    Args:
        hora: Valor da hora como string ou None.

    Returns:
        True se o valor estiver no formato correto, False caso contrário.
    """
    # Verifica se hora é nulo ou não é uma string
    if hora is None or not isinstance(hora, str):
        return False

    # Expressão regular para validar o formato de hora HH:MM:SS
    hora_regex = r"^([01]?[0-9]|2[0-3]):([0-5][0-9])(:[0-5][0-9])?$"

    return re.match(hora_regex, hora) is not None

#STEP RODADO LOCALMENTE POR PROBLEMA DE MEMÓRIA
import os
import pandas as pd

# Diretório com os arquivos XLSX
diretorio = r"C:\Users\luiz0\OneDrive\Área de Trabalho\Segurança - Projeeto 2\RawData\Criminais"

# DataFrame para armazenar todos os dados
df_total = pd.DataFrame()

# Loop para ler cada arquivo XLSX
for arquivo in os.listdir(diretorio):
    if arquivo.endswith(".xlsx"):
        # Ler o arquivo XLSX como um dicionário de DataFrames (cada planilha é um DataFrame)
        dict_df = pd.read_excel(os.path.join(diretorio, arquivo), sheet_name=None)

        # Iterar sobre cada planilha no dicionário e empilhar os DataFrames
        for _, df in dict_df.items():
            df_total = pd.concat([df_total, df], ignore_index=True)

# Salvar o DataFrame total como CSV
df_total.to_csv(
    r"C:\Users\luiz0\OneDrive\Área de Trabalho\Segurança - Projeeto 2\RawData\Criminais\Dados_Criminais_Consolidados.csv",
    index=False)

print("Todos os dados foram consolidados e salvos em um único arquivo CSV com sucesso!")


# Análise de qualidade dos dados e correções

#Para essa análise usaremos a cidade de são paulo como case.
df = df[df['CIDADE'] == 'S.PAULO']
# Visualizando a relação entre duas variáveis para ver preenchimento da informação de longitude e latitude
resultado_validacao = df.apply(lambda x: valida_lat_long(x['LATITUDE'], x['LONGITUDE']), axis=1)

# Contando os valores True e False
contagem = resultado_validacao.value_counts()

print(contagem)

# Visualizando a relação entre duas variáveis para ver preenchimento da informação de hora
nao_nulas = df[df['HORA_OCORRENCIA_BO'].notnull()]

contagem_nao_nulas = nao_nulas.shape[0]
contagem_nulas = df['HORA_OCORRENCIA_BO'].isnull().sum()

print(contagem_nao_nulas, contagem_nulas)

def normalizar_bairros(bairro):
    if isinstance(bairro, str):
        return re.sub(r'\W+', '', bairro).upper()
    else:
        return "BAIRRO_NAO_ESPECIFICADO"

def preencher_horas_ausentes(df):
    df['BAIRRO_NORMALIZADO'] = df['BAIRRO'].apply(normalizar_bairros)
    df['HORA_OCORRENCIA_DT'] = pd.to_datetime(df['HORA_OCORRENCIA_BO'], format='%H:%M:%S', errors='coerce')
    df['HORA'] = df['HORA_OCORRENCIA_DT'].dt.hour

    # Calcular a mediana geral de horas na base para usar em valores finais nulos
    mediana_geral = df['HORA'].median()

    intervalos = {
        'DE MADRUGADA': (0, 6),
        'PELA MANHÃ': (6, 12),
        'A TARDE': (12, 18),
        'A NOITE': (18, 24)
    }

    medianas_por_grupo = df.groupby(['DESCR_PERIODO', 'RUBRICA', 'BAIRRO_NORMALIZADO'])['HORA'].median().reset_index()

    df['HORA_CALCULADA'] = np.nan

    for i, row in df[df['HORA_OCORRENCIA_BO'].isnull() & df['DESCR_PERIODO'].notnull()].iterrows():
        periodo = row['DESCR_PERIODO']
        rubrica = row['RUBRICA']
        bairro = row['BAIRRO_NORMALIZADO']
        mediana = medianas_por_grupo[
            (medianas_por_grupo['DESCR_PERIODO'] == periodo) &
            (medianas_por_grupo['RUBRICA'] == rubrica) &
            (medianas_por_grupo['BAIRRO_NORMALIZADO'] == bairro)
        ]['HORA'].median()

        if np.isnan(mediana):
            mediana = medianas_por_grupo[
                (medianas_por_grupo['DESCR_PERIODO'] == periodo) &
                (medianas_por_grupo['BAIRRO_NORMALIZADO'] == bairro)
            ]['HORA'].median()

            if np.isnan(mediana):
                mediana = df[(df['DESCR_PERIODO'] == periodo)]['HORA'].median()

        if not np.isnan(mediana):
            df.at[i, 'HORA_CALCULADA'] = pd.Timestamp(year=1900, month=1, day=1, hour=int(mediana)).strftime('%H:%M:%S')

    df['HORA_OCORRENCIA_BO'] = df['HORA_OCORRENCIA_BO'].combine_first(df['HORA_CALCULADA'])

    # Substituir os valores NaN restantes pela mediana geral
    if not np.isnan(mediana_geral):
        hora_mediana_geral = pd.Timestamp(year=1900, month=1, day=1, hour=int(mediana_geral)).strftime('%H:%M:%S')
        df['HORA_OCORRENCIA_BO'].fillna(hora_mediana_geral, inplace=True)

    df.drop(columns=['HORA_CALCULADA'], inplace=True)

    return df

df_preenchido = preencher_horas_ausentes(df_session)
df_session = df_preenchido

#Tentando preencher a latidude e longitude via logradouro

api_key = 'AIzaSyDDi4e4kINLq-fVn7HCRbTqoVhWdIIY24M'
gmaps = googlemaps.Client(key=api_key)

def buscar_coordenadas(logradouro):
    """
    Busca as coordenadas de latitude e longitude para um logradouro usando o Google Maps API.

    Args:
        logradouro (str): O logradouro para pesquisa.

    Returns:
        tuple: Uma tupla contendo (latitude, longitude).
    """
    # Realizar a pesquisa com o Google Maps API
    result = gmaps.geocode(logradouro + ', São Paulo, SP, Brasil')

    if result:
        # Extrair latitude e longitude
        lat = result[0]['geometry']['location']['lat']
        lng = result[0]['geometry']['location']['lng']
        return (lat, lng)
    else:
        # Retornar None se não houver resultados
        return (None, None)

# Aplicando a função para atualizar o DataFrame
for index, row in df.iterrows():
    if pd.isnull(row['LATITUDE']) or row['LATITUDE'] == 0 or pd.isnull(row['LONGITUDE']) or row['LONGITUDE'] == 0:
        lat, lng = buscar_coordenadas(row['LOGRADOURO'])
        if lat and lng:
            df.at[index, 'LATITUDE'] = lat
            df.at[index, 'LONGITUDE'] = lng
          
# Visualizando a relação entre duas variáveis para ver preenchimento da informação de longitude e latitude
resultado_validacao = df.apply(lambda x: valida_lat_long(x['LATITUDE'], x['LONGITUDE']), axis=1)

# Contando os valores True e False
contagem = resultado_validacao.value_counts()

#Quando os valores são nulos, a coluna de descrição de tempo entra. substituindo os valores nulos pela mediana dos períodos para fins de normalizaçãoe e analise
df['HORA_OCORRENCIA_DT'] = pd.to_datetime(df['HORA_OCORRENCIA_BO'], format='%H:%M:%S', errors='coerce')

# Extrair a hora do datetime para nova coluna
df['HORA'] = df['HORA_OCORRENCIA_DT'].dt.hour
df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')

# Definindo os intervalos
intervalos = {
    'De madrugada': (0, 6),
    'Pela manhã': (6, 12),
    'A tarde': (12, 18),
    'A noite': (18, 24)
}

# Calcular a mediana para cada intervalo
medianas = {}
for periodo, (inicio, fim) in intervalos.items():
    # Selecionar horas dentro do intervalo e calcular mediana
    medianas[periodo] = df[(df['HORA'] >= inicio) & (df['HORA'] < fim)]['HORA'].median()

# Para simplificar, assumiremos mediana como inteiro para substituição
medianas = {k: pd.Timedelta(hours=int(v)) if not np.isnan(v) else None for k, v in medianas.items()}

# Substituição de valores nulos com base em DESC_PERIODO
for periodo, mediana in medianas.items():
    if mediana is not None:
        # Substituir valores nulos na coluna de horas para o período especificado
        df.loc[(df['HORA_OCORRENCIA_BO'].isnull()) & (df['DESC_PERIODO'] == periodo), 'HORA_OCORRENCIA_DT'] = df['DATA_OCORRENCIA_BO'] + mediana

# Converter HORA_OCORRENCIA_DT de volta para string formatada HH:MM:SS se necessário
df['HORA_OCORRENCIA_BO'] = df['HORA_OCORRENCIA_DT'].dt.strftime('%H:%M:%S')

#Nova contagem
nao_nulas = df[df['HORA_OCORRENCIA_BO'].notnull()]

contagem_nao_nulas = nao_nulas.shape[0]
contagem_nulas = df['HORA_OCORRENCIA_BO'].isnull().sum()

print(contagem_nao_nulas, contagem_nulas)

#Salvando o DF final

df.to_excel('dataframe_tratado_consolidado.xlsx', index=False)

df_session = pd.read_csv('/content/Dados_Criminais_Consolidados_Cidade_Saopaulo.csv')

df_session.info()
df_session.describe()

print(df_session['HORA_OCORRENCIA_BO'])

# Fazendo uma análise exploratória dos dados

# Configurando o estilo dos gráficos
sns.set(style="whitegrid")
data = df_session

#Crimes por ano
plt.figure(figsize=(10, 6))
sns.countplot(x='ANO_BO', data=data)
plt.title('Distribuição de Crimes por Ano')
plt.xlabel('Ano do Boletim de Ocorrência')
plt.ylabel('Contagem de Crimes')
plt.show()


#crimes por mes
#data['MES_GERAL'] = data['DATA_OCORRENCIA_BO'].dt.strftime('%m')
data = data.sort_values(by='MES_ESTATISTICA', ascending=True)
plt.figure(figsize=(10, 6))
sns.countplot(x='MES_ESTATISTICA', data=data)
plt.title('Distribuição de Crimes por Mês')
plt.xlabel('Mês')
plt.ylabel('Contagem de Crimes')
plt.show()

#  dia da semana
data['DATA_OCORRENCIA_BO'] = pd.to_datetime(data['DATA_OCORRENCIA_BO'], errors='coerce')

data['DIA_SEMANA'] = data['DATA_OCORRENCIA_BO'].dt.day_name()
plt.figure(figsize=(10, 6))
sns.countplot(x='DIA_SEMANA', data=data, order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.title('Distribuição de Crimes por Dia da Semana')
plt.xlabel('Dia da Semana')
plt.ylabel('Contagem de Crimes')
plt.xticks(rotation=45)
plt.show()

#hora do dia

# Interpolar valores NaN usando método linear
df_session['HORA_OCORRENCIA_BO'] = df_session['HORA_OCORRENCIA_BO'].interpolate('linear')
df_session['HORA_OCORRENCIA_BO'] = pd.to_datetime(df_session['HORA_OCORRENCIA_BO'], format='%H:%M:%S',errors='coerce')
plt.figure(figsize=(12, 6))
sns.histplot(data=data, x='HORA_OCORRENCIA_BO', bins=24, kde=True)
plt.title('Distribuição de Crimes por Hora do Dia')
plt.xlabel('Hora do Dia')
plt.ylabel('Contagem de Crimes')
plt.xlim(0, 23)
plt.xticks(range(0, 24))
plt.show()

# Analisando a frequência dos diferentes tipos de crimes (rubrica)
plt.figure(figsize=(12, 8))
crime_types_counts = data['RUBRICA'].value_counts().head(10)
sns.barplot(y=crime_types_counts.index, x=crime_types_counts.values, palette='viridis')
plt.title('Top 10 Tipos de Crimes Mais Frequentes')
plt.xlabel('Contagem de Crimes')
plt.ylabel('Tipo de Crime')
plt.show()

# Relação entre tipos de locais e a incidência de crimes
plt.figure(figsize=(12, 8))
local_crime_counts = data['DESCR_TIPOLOCAL'].value_counts().head(10)
sns.barplot(y=local_crime_counts.index, x=local_crime_counts.values, palette='coolwarm')
plt.title('Top 10 Locais com Maior Incidência de Crimes')
plt.xlabel('Contagem de Crimes')
plt.ylabel('Tipo de Local')
plt.show()


# Análise de texto simplificada da coluna DESCR_CONDUTA
conduta_counts = data['BAIRRO'].value_counts().head(30)

plt.figure(figsize=(12, 8))
sns.barplot(y=conduta_counts.index, x=conduta_counts.values, palette='muted')
plt.title('Top 10 Bairros mais perigosos')
plt.xlabel('Contagem')
plt.ylabel('Bairro')
plt.show()


import folium
from folium.plugins import HeatMap

# Supondo que 'data' seja seu DataFrame e ele contém as colunas 'LATITUDE' e 'LONGITUDE'
# Certifique-se de que as colunas LATITUDE e LONGITUDE estão no formato correto
data = data.dropna(subset=['LATITUDE', 'LONGITUDE'])

# Lista de coordenadas a partir das colunas LATITUDE e LONGITUDE
heat_data = [[row['LATITUDE'], row['LONGITUDE']] for index, row in data.iterrows()]

# Coordenadas do centro de São Paulo
centro_sp = [-23.550520, -46.633308]

# Criando um mapa centrado no centro de São Paulo
mapa = folium.Map(location=centro_sp, zoom_start=12)

# Adicionando a camada de mapa de calor ao mapa
HeatMap(heat_data).add_to(mapa)

# Exibindo o mapa
mapa



# MARL



import pandas as pd

data = pd.read_csv("/content/drive/MyDrive/data/Dados_Criminais_Consolidados_Cidade_Saopaulo.csv")
print(data.info())

data['DATA_OCORRENCIA_BO'] = pd.to_datetime(data['DATA_OCORRENCIA_BO'],errors='coerce')
data['HORA_OCORRENCIA_BO'] = pd.to_datetime(data['HORA_OCORRENCIA_BO'], format='%H:%M:%S',errors='coerce').dt.time

lugar = "São Paulo, São Paulo, Brazil"
grafo = ox.graph_from_place(lugar, network_type='drive')

import osmnx as ox
import networkx as nx
latitude_origem=-23.635693727090537
longitude_origem=-46.64219137955335
latitude_destino=-23.555725762552985
longitude_destino= -46.67674588403965

origem = ox.distance.nearest_nodes(grafo, X=longitude_origem, Y=latitude_origem)
destino = ox.distance.nearest_nodes(grafo, X=longitude_destino, Y=latitude_destino)
rota = nx.shortest_path(grafo, origem, destino, weight='length')

fig, ax = ox.plot_graph_route(grafo, rota, node_size=0)

import pandas as pd

# Carregar os dados novamente (ajuste o caminho conforme necessário)
dados_crime = pd.read_csv("/content/drive/MyDrive/data/Dados_Criminais_Consolidados_Cidade_Saopaulo.csv", low_memory=False)

# Convertendo 'DATA_OCORRENCIA_BO' para datetime com tratamento de erros
dados_crime['DATA_OCORRENCIA_BO'] = pd.to_datetime(dados_crime['DATA_OCORRENCIA_BO'], errors='coerce')

# Preencher valores nulos na coluna 'HORA_OCORRENCIA_BO'
dados_crime['HORA_OCORRENCIA_BO'].fillna("12:00:00", inplace=True)

# Certificando-se de que 'HORA_OCORRENCIA_BO' é uma string
dados_crime['HORA_OCORRENCIA_BO'] = dados_crime['HORA_OCORRENCIA_BO'].astype(str)

# Combinando as colunas 'DATA_OCORRENCIA_BO' e 'HORA_OCORRENCIA_BO' em uma coluna 'timestamp'
# Como 'DATA_OCORRENCIA_BO' é datetime, podemos usá-lo diretamente
dados_crime['timestamp'] = pd.to_datetime(dados_crime['DATA_OCORRENCIA_BO'].dt.strftime('%Y-%m-%d') + ' ' + dados_crime['HORA_OCORRENCIA_BO'])

# Verificando o resultado
dados_crime[['DATA_OCORRENCIA_BO', 'HORA_OCORRENCIA_BO', 'timestamp']].head()
dados_crime_formatado = dados_crime[['LATITUDE', 'LONGITUDE', 'timestamp']]

# Removendo entradas com latitude e longitude iguais a 0 ou nulas do DataFrame dados_crime_formatado
dados_crime_formatado = dados_crime_formatado[(dados_crime_formatado['LATITUDE'] != 0) & (dados_crime_formatado['LATITUDE'].notna())]
dados_crime_formatado = dados_crime_formatado[(dados_crime_formatado['LONGITUDE'] != 0) & (dados_crime_formatado['LONGITUDE'].notna())]

import osmnx as ox
import gym
from gym import spaces
import numpy as np
import random
import pandas as pd
from geopy.distance import geodesic


class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        q_predict = self.q_table[state, action]
        q_target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (q_target - q_predict)
        self.epsilon *= self.epsilon_decay

class ViaturasAmbiente(gym.Env):
    def __init__(self, dados_crime, lugar="São Paulo, São Paulo, Brazil"):
        super(ViaturasAmbiente, self).__init__()
        self.grafo_cidade = ox.graph_from_place(lugar, network_type='drive')
        self.dados_crime = dados_crime
        self.n_states = len(list(self.grafo_cidade.nodes))
        self.action_space = spaces.Discrete(self.n_states)
        self.observation_space = spaces.Discrete(self.n_states)
        self.estado = 0  # Estado inicial

    def step(self, action):
        # Verifica se a ação corresponde a um nó existente no grafo
        if action in self.grafo_cidade.nodes:
            self.estado = action  # Move para o novo estado (nó)
            recompensa = self.calcular_recompensa(self.estado)
        else:
            # Para ações inválidas, aplica uma penalidade e não altera o estado
            recompensa = -10  # Exemplo de penalidade para ação inválida

        done = False  # A lógica de término da simulação vai aqui
        return self.estado, recompensa, done, {}


    def calcular_recompensa(self, estado):
        # Encontra as coordenadas do estado atual (viatura)
        viatura_coords = self.grafo_cidade.nodes[estado]['y'], self.grafo_cidade.nodes[estado]['x']

        # Inicializa a recompensa
        recompensa = 0

        # Verifica cada crime no DataFrame
        for _, crime in self.dados_crime.iterrows():
            crime_coords = (crime['latitude'], crime['longitude'])
            distancia = geodesic(viatura_coords, crime_coords).meters  # Calcula a distância em metros

            if distancia <= 300:  # Se a viatura está dentro de um raio de 300m de um crime
                recompensa += 1  # Aumenta a recompensa

        return recompensa


    def reset(self):
        self.estado = 0  # Resetar para o estado inicial
        return self.estado

    def render(self, mode='console'):
        if mode == 'console':
            print(f"Estado atual: {self.estado}")
        else:
            raise NotImplementedError("Modo não suportado")

def simular_dias_corridos(env, agent, dados_crime, episodios=10):
    for episodio in range(episodios):
        estado = env.reset()
        done = False

        while not done:

            acao = agent.choose_action(estado)
            novo_estado, recompensa, done, _ = env.step(acao)
            agent.learn(estado, acao, recompensa, novo_estado)
            estado = novo_estado
            if done:
                break


dados_crime = dados_crime_formatado
env = ViaturasAmbiente(dados_crime=dados_crime)
agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_states)

print(simular_dias_corridos(env, agent, dados_crime))


# Testes de acurácia


def avaliar_agente(env, agent, dados_crime, episodios=10):
    total_crimes = 0
    crimes_detectados = 0

    for episodio in range(episodios):
        estado = env.reset()
        done = False

        while not done:
            acao = agent.choose_action(estado)
            novo_estado, _, done, _ = env.step(acao)

            # Verifica se a ação leva a um crime detectado
            crime_detectado = env.verificar_crime_detectado(novo_estado)
            if crime_detectado:
                crimes_detectados += 1

            total_crimes += 1
            estado = novo_estado

    taxa_detectacao = crimes_detectados / total_crimes if total_crimes > 0 else 0
    return taxa_detectacao

def verificar_crime_detectado(self, estado):
    viatura_coords = self.grafo_cidade.nodes[estado]['y'], self.grafo_cidade.nodes[estado]['x']
    for _, crime in self.dados_crime.iterrows():
        crime_coords = (crime['latitude'], crime['longitude'])
        distancia = geodesic(viatura_coords, crime_coords).meters
        if distancia <= 300:  # Se a viatura está dentro de um raio de 300m de um crime
            return True
    return False

taxa_detectacao = avaliar_agente(env, agent, dados_crime)
print(f"Taxa de detecção de crimes: {taxa_detectacao}")
