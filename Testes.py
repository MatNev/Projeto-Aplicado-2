

# File para guardar códigos que precisam de ajustes ou simplesmente códigos usados para testes que não condizem com o resultado final.

# ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________

# MARL

import pandas as pd
import osmnx as ox
import gym
from gym import spaces
import numpy as np
from geopy.distance import geodesic

# Carregar os dados 
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


# ___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
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

# Carregar os dados novamente
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
        self.estado = 0  # Reseta para o estado inicial
        return self.estado


env = ViaturasAmbiente(dados_crime_formatado)

# Cria um agente Q-learning com o mesmo número de estados e ações que o ambiente
agente = QLearningAgent(env.n_states, env.n_states)

# Treina o agente por um número de episódios
num_episodios = 1000
for _ in range(num_episodios):
    estado = env.reset()
    done = False
    while not done:
        acao = agente.choose_action(estado)
        proximo_estado, recompensa, done, info = env.step(acao)
        agente.learn(estado, acao, recompensa, proximo_estado)
        estado = proximo_estado

# Imprime a tabela Q após o treinamento
print(agente.q_table)

# Cria uma simulação final após o treinamento
estado = env.reset()
done = False
while not done:
    acao = agente.choose_action(estado)
    proximo_estado, recompensa, done, info = env.step(acao)
    print(f"A viatura está na localização {proximo_estado} com recompensa {recompensa}")

