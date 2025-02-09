import pandas as pd
import numpy as np
from uncertainties import ufloat, ufloat_fromstr
from uncertainties.umath import log as ulog
from uncertainties.unumpy import nominal_values, std_devs
import matplotlib.pyplot as plt

def to_ufloat(df, incertezas, axis=0):
    if type(incertezas) == float or type(incertezas) == int:
        incertezas = np.array([incertezas])
    if incertezas.shape == (1,):
        incertezas = np.repeat(incertezas, df.shape[axis])

    rows = df.shape[0]
    columns = df.shape[1]

    df_x = []
    df_temp = []

    if axis == 0:
      for coluna in range(columns):
        df_x.append([ufloat(valor, incertezas[row]) for row, valor in enumerate(df.iloc[:, coluna])])

      # Atualizando o DataFrame com os novos valores de ufloat
      for coluna in range(columns):
        df[df.columns[coluna]] = df_x[coluna]

    else:
      for row in range(rows):
        df_x.append([ufloat(valor, incertezas[row]) for valor in df.iloc[row, :]])

      # Atualizando o DataFrame com os novos valores de ufloat
      for index in range(columns):
        coluna = []
        for row in df_x:
          coluna.append(row[index])
        df_temp.append(coluna)

      for coluna in range(columns):
        df[df.columns[coluna]] = df_temp[coluna]

    return df  # Return the new DataFrame with ufloat values

def calculate_umean(df_x, axis = 0):
  medias = []
  if axis == 0:
    for i in range(df_x.shape[0]):
      medias.append(np.mean(df_x[i, :]))
  else:
    for i in range(df_x.shape[1]):
      medias.append(np.mean(df_x[:, i]))
  return np.array(medias)

def calculate_ulog(x_means, y):
  log_x = [ulog(val) for val in x_means]
  log_y = [ulog(val) for val in y] 

  return np.array(log_x), np.array(log_y)

def plot_ufloat(ux, uy, xlabel = 'eixo x', ylabel = 'eixo y', title = 'gráfico', markersize = 5, fmt = 'o', 
                figsize = (8, 5), capsize = 8):
  # Criando o gráfico
  plt.figure(figsize=figsize)
  plt.errorbar(nominal_values(ux), nominal_values(uy), xerr = std_devs(ux), yerr=std_devs(uy), 
               markersize = markersize, fmt=fmt, capsize=capsize)
  # Personalização do gráfico
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.grid(True)
  plt.show()

def reg_linear(ulog_x, ulog_y, decimal_cases = 5, return_df = False, plot = False, xlabel = 'eixo x',
               ylabel = 'eixo y', title = 'Gráfico da regressão linear', markersize = 5, fmt = 'o', figsize = (8, 5), capsize = 8):
  stdDev_lny = std_devs(ulog_y)
  stdDev_lnx = std_devs(ulog_x)

  nominal_lnx = nominal_values(ulog_x)
  nominal_lny = nominal_values(ulog_y)

  w = np.power(stdDev_lny,-2)
  somatorio_w = np.sum(w)

  wyx = w * nominal_lnx * nominal_lny
  somatorio_wyx = np.sum(wyx)

  wy = w * nominal_lny
  somatorio_wy = np.sum(wy)

  wx = w * nominal_lnx
  somatorio_wx = np.sum(wx)

  wx2 = w * np.power(nominal_lnx, 2)
  somatorio_wx2 = np.sum(wx2)

  delta = somatorio_w * somatorio_wx2 - np.power(somatorio_wx, 2)

  a = round((somatorio_w * somatorio_wyx - somatorio_wy * somatorio_wx)/delta, decimal_cases)
  b = round((somatorio_wy * somatorio_wx2 - somatorio_wyx * somatorio_wx)/delta, decimal_cases)

  erro_a = round((somatorio_w/delta)**(1/2), decimal_cases)
  erro_b = round((somatorio_wx2/delta)**(1/2), decimal_cases)
  print(f"Inclinação (a): {a}")
  print(f"Intercepto (b): {b}")
  print(f"Incerteza na inclinação (Δa): {erro_a}")
  print(f"Incerteza no intercepto (Δb): {erro_b}")

  if return_df:
    n = len(ulog_y)
    dados = {
                'N': [x for x in range(1, n +1)],
                'Wi': w,
                'Somatório W': somatorio_w,
                'Somatório wyx': somatorio_wyx,
                'Delta': delta,
                'Somatório wx': somatorio_wx,
                'Somatório wy': somatorio_wy,
                'Somatorio wx²':somatorio_wx2,
                'Somatório (wx)²': somatorio_wx**2,
                'Inclinação (a)': a,
                'Intercepto (b)': b,
                'Incerteza na inclinação (Δa)': erro_a,
                'Incerteza no intercepto (Δb)': erro_b
            }          
    df = pd.DataFrame(dados)
    df.to_csv('Regressão Linear.csv', index = False)
    df.to_excel('Regressão Linear.xlsx', index = False)
  if plot:
    # Definindo os dados para a reta
    x = nominal_lnx  # 100 pontos de 0 a 10
    y = a * x + b  # Reta: y = 2x + 1 (exemplo)
    # Plotando a reta
    plt.figure(figsize=figsize)
    plt.plot(x, y, label=f'y = {a}x + {b}', color='blue')  # Linha azul
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()  # Mostra o rótulo da reta
    plt.grid(True)
    plt.errorbar(nominal_lnx, nominal_lny, xerr = stdDev_lnx, yerr=stdDev_lny, 
               markersize = markersize, fmt=fmt, capsize=capsize)
    plt.show()
