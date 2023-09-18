import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
import random
from OCT_tree import RegularOptimalTreeClassifier 
from OCT_tree import MOptimalTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree 
import graphviz

def data_processing():
    
    file_path = "C:\PythonProject\InvalidTreeTest\dataset"
    house_data_path =file_path + "\house-votes-84.data"
    house_data = pd.read_csv(house_data_path,header=None, delimiter=',')
    for i in range(1,17):
        house_data = house_data[house_data[i] != '?']
    house_data = house_data.apply(lambda x: pd.factorize(x)[0])
    cols = list(range(len(house_data.columns)))
    cols.remove(0)
    cols.append(0)
    house_data = house_data.loc[:, cols]
    
    file_path = "C:\PythonProject\InvalidTreeTest\dataset"
    mushroom_data_path = file_path + "\\agaricus-lepiota.data"
    mushroom_data = pd.read_csv(mushroom_data_path,header=None, delimiter=',')
    for i in range(1,len(mushroom_data.columns)):
        mushroom_data = mushroom_data[mushroom_data[i] != '?']
    mushroom_data = mushroom_data.apply(lambda x: pd.factorize(x)[0])
    cols = list(range(len(mushroom_data.columns)))
    cols.remove(0)
    cols.append(0)
    mushroom_data = mushroom_data.loc[:, cols]
    
    file_path = "C:\PythonProject\InvalidTreeTest\dataset"
    chess_data_path = file_path + "\\kr-vs-kp.data"
    chess_data = pd.read_csv(chess_data_path,header=None, delimiter=',')
    for i in range(1,len(chess_data.columns)):
        chess_data = chess_data[chess_data[i] != '?']
    chess_data = chess_data.apply(lambda x: pd.factorize(x)[0])
    
    
    soybean_data_path = file_path + "\\soybean-large.data"
    soybean_data = pd.read_csv(soybean_data_path,header=None, delimiter=',')
    for i in range(1,len(soybean_data.columns)):
        soybean_data = soybean_data[soybean_data[i] != '?']
    cols = list(range(len(soybean_data.columns)))
    cols.remove(0)
    cols.append(0)
    soybean_data = soybean_data.loc[:, cols]
    
    file_path = "C:\PythonProject\InvalidTreeTest\dataset"
    nurse_data_path = file_path + "\\nursery.data"
    nurse_data = pd.read_csv(nurse_data_path,header=None, delimiter=',')
    for i in range(1,len(nurse_data.columns)):
        nurse_data = nurse_data[nurse_data[i] != '?']
    nurse_data = nurse_data.apply(lambda x: pd.factorize(x)[0])
    cols = list(range(len(nurse_data.columns)))
    cols.remove(0)
    cols.append(0)
    nurse_data = nurse_data.loc[:, cols]
    
    rasin_data_path = file_path + "\\rasin.csv"
    rasin_data = pd.read_csv(rasin_data_path)
    
    drybean_data_path = file_path + "\drybean.csv"
    drybean_data = pd.read_csv(drybean_data_path)
    
    cancer_data_path = file_path + "\\breast-cancer-wisconsin.data"
    breastcancerwisconsin = pd.read_csv(cancer_data_path)
    breastcancerwisconsin.drop(columns=['1000025'],inplace=True)
    index = breastcancerwisconsin[ (breastcancerwisconsin['5'] == '?') | (breastcancerwisconsin['1'] == '?') \
                                |(breastcancerwisconsin['1.1'] == '?')|(breastcancerwisconsin['1.2'] == '?')\
                                |(breastcancerwisconsin['2'] == '?')|(breastcancerwisconsin['1.3'] == '?')\
                                |(breastcancerwisconsin['3'] == '?')|(breastcancerwisconsin['1.4'] == '?')\
                                |(breastcancerwisconsin['1.5'] == '?')|(breastcancerwisconsin['2.1'] == '?')].index
    breastcancerwisconsin.drop(index , inplace=True)
    
    car_data_path = file_path + "\car.data"
    car = pd.read_csv(car_data_path)
    car['vhigh'].replace(['high', 'low','med','vhigh'],
                        [0, 1,2,3], inplace=True)
    car['vhigh.1'].replace(['high', 'low','med','vhigh'],
                        [0, 1,2,3], inplace=True)
    car['2'].replace(['2', '3','4','5more'],
                        [0, 1,2,3], inplace=True)
    car['2.1'].replace(['2','4','more'],
                        [0, 1,2], inplace=True)
    car['small'].replace(['small','med','big'],
                        [0, 1,2], inplace=True)
    car['low'].replace(['low','med','high'],
                        [0, 1,2], inplace=True)
    car['unacc'].replace(['unacc','acc','good','vgood'],
                        [0, 1,2,3], inplace=True)
    
    german_path = file_path + "\\german.data"
    german = pd.read_csv(german_path,header=None,sep=' ')
    columns = list(german.columns)
    for element in columns:
        german[element].replace(list(german[element].unique()),
                        list(range(len(list(german[element].unique())))), inplace=True)
    
    balance_path = file_path + "\\balance-scale.data"
    balance = pd.read_csv(balance_path,header=None,sep=',')
    balance[0].replace(list(balance[0].unique()),
                        list(range(len(list(balance[0].unique())))), inplace=True)
    cols = [1,2,3,4,0]
    balance = balance.loc[:, cols]

    biodeg = file_path + "\\biodeg.csv"
    biodeg = pd.read_csv(biodeg,header=None,sep=';')

    glass_path = file_path + "\\glass.data"
    glass = pd.read_csv(glass_path,header=None,sep=',')
    glass= glass.drop(columns=0)

    ionosphere_path = file_path + "\\ionosphere.data"
    ionosphere = pd.read_csv(ionosphere_path,header=None,sep=',')
    ionosphere= ionosphere.drop(columns=1)

    tic_path = file_path + "\\tic-tac-toe.data"
    tic = pd.read_csv(tic_path,header=None,sep=',')
    columns = list(tic.columns)
    for element in columns:
        tic[element].replace(list(tic[element].unique()),
                        list(range(len(list(tic[element].unique())))), inplace=True)

    bank_path = file_path + "\\data_banknote_authentication.txt"
    bank = pd.read_csv(bank_path,header=None,sep=',')
    
    iris_data_path = file_path + "\\iris.csv"
    iris_data = pd.read_csv(iris_data_path)
    iris_data.drop(columns=["Id"], inplace=True)
    data_path = file_path + "\\abalone.data"
    abalone_data = pd.read_csv(data_path)
    abalone_data['M'] = abalone_data['M'].replace(['M'],'0')
    abalone_data['M'] = abalone_data['M'].replace(['F'],'1')
    abalone_data['M'] = abalone_data['M'].replace(['I'],'2')
    data_path =file_path + "\\winequality-white1.txt"
    white_wine_data = pd.read_csv(data_path,sep=";")
    data_path =file_path + "\\winequality-red1.txt"
    red_wine_data = pd.read_csv(data_path,sep=";")
  

    
    
    data_list = [house_data,mushroom_data,chess_data,drybean_data,nurse_data,soybean_data,breastcancerwisconsin,car,german,balance,biodeg,\
                 glass,red_wine_data,tic,white_wine_data]
    xy_list = []
    for data in data_list:
        xy_list.append([data.iloc[:,0:len(data.columns)-1].to_numpy(),data[data.columns[len(data.columns)-1]].to_numpy()])
        
    return xy_list

def fit_OCT(data_list, data_set, max_depth, small_epsilon, epsilon_version):
    
    
    n = 200
    xy_list = data_list
    samples = [[],[]]
    for t in range(int(round(n,0))):
        random_number=random.randint(0,len(xy_list[data_set][0])-1)
        samples[0].append(xy_list[data_set][0][random_number])
        samples[1].append(xy_list[data_set][1][random_number])
    
    X = samples[0]
    y = samples[1]
    
    # OCT parameters
    max_depth = max_depth
    min_samples_leaf = 1
    alpha = 0.01
    time_limit = 5  # minute
    mip_gap_tol = 0.01  # optimal gap percentage
    mip_focus = 'optimal'
    mip_polish_time = None
    warm_start = False
    log_file = None

#model, run_time, solution_condition = solve_oct_MILP(X_transformed, y_transformed, L_hat, epsilons,
#                   alpha=0.01, max_depth=2, min_samples_leaf=1, small_number = 0.0000001,
#                   solver="gurobi", small_epsilon=False,epsilon_version=False,verbose=False, log_file=None)

# Construct OCT classifier
    aborted_status = 0
    oct_model = RegularOptimalTreeClassifier(max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  alpha=alpha,
                                  criterion="gini",
                                  solver="gurobi",
                                  time_limit=time_limit,
                                  small_epsilon=small_epsilon,
                                  epsilon_version=epsilon_version,
                                  verbose=True,
                                  warm_start=warm_start,
                                  log_file=log_file,
                                  solver_options={'mip_cuts': 'auto',
                                                  'mip_gap_tol': mip_gap_tol,
                                                  'mip_focus': mip_focus,
                                                  'mip_polish_time': mip_polish_time
                                                  }
                                  )
#    try:
    fitted_model = oct_model.fit(X, y)
#    except:
#        aborted_status = 1
    
    return fitted_model, aborted_status


def fit_mOCT(data_list, data_set, max_depth, small_epsilon, epsilon_version):
    
    
    n = 200
    xy_list = data_list
    samples = [[],[]]
    for t in range(int(round(n,0))):
        random_number=random.randint(0,len(xy_list[data_set][0])-1)
        samples[0].append(xy_list[data_set][0][random_number])
        samples[1].append(xy_list[data_set][1][random_number])
    
    X = samples[0]
    y = samples[1]
    
    # OCT parameters
    max_depth = max_depth
    min_samples_leaf = 1
    alpha = 0.01
    time_limit = 5  # minute
    mip_gap_tol = 0.01  # optimal gap percentage
    mip_focus = 'optimal'
    mip_polish_time = None
    warm_start = False
    log_file = None

#model, run_time, solution_condition = solve_oct_MILP(X_transformed, y_transformed, L_hat, epsilons,
#                   alpha=0.01, max_depth=2, min_samples_leaf=1, small_number = 0.0000001,
#                   solver="gurobi", small_epsilon=False,epsilon_version=False,verbose=False, log_file=None)

# Construct OCT classifier
    aborted_status = 0
    oct_model = MOptimalTreeClassifier(max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  alpha=alpha,
                                  criterion="gini",
                                  solver="gurobi",
                                  time_limit=time_limit,
                                  small_epsilon=small_epsilon,
                                  epsilon_version=epsilon_version,
                                  verbose=True,
                                  warm_start=warm_start,
                                  log_file=log_file,
                                  solver_options={'mip_cuts': 'auto',
                                                  'mip_gap_tol': mip_gap_tol,
                                                  'mip_focus': mip_focus,
                                                  'mip_polish_time': mip_polish_time
                                                  }
                                  )
#    try:
    fitted_model = oct_model.fit(X, y)
#    except:
#        aborted_status = 1
    
    return fitted_model, aborted_status





