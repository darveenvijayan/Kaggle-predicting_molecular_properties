
import numpy as np
import pandas as pd 
from IPython.display import display
pd.options.display.max_columns = None
from functools import reduce
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
#import plotly.plotly as py
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import ase
from ase import Atoms
import ase.visualize
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#%%
dipole_moments = pd.read_csv(r'C:\Users\darveen.vijayan\Desktop\Data Science\R\Kaggle Training\Predicting Molecular Properties\dipole_moments.csv')
magnetic_shielding_tensors = pd.read_csv(r'C:\Users\darveen.vijayan\Desktop\Data Science\R\Kaggle Training\Predicting Molecular Properties\magnetic_shielding_tensors.csv')
mulliken_charges = pd.read_csv(r'C:\Users\darveen.vijayan\Desktop\Data Science\R\Kaggle Training\Predicting Molecular Properties\mulliken_charges.csv')
potential_energy = pd.read_csv(r'C:\Users\darveen.vijayan\Desktop\Data Science\R\Kaggle Training\Predicting Molecular Properties\potential_energy.csv')
scalar_coupling_contributions = pd.read_csv(r'C:\Users\darveen.vijayan\Desktop\Data Science\R\Kaggle Training\Predicting Molecular Properties\scalar_coupling_contributions.csv')
structures = pd.read_csv(r'C:\Users\darveen.vijayan\Desktop\Data Science\R\Kaggle Training\Predicting Molecular Properties\structures.csv')
train = pd.read_csv(r'C:\Users\darveen.vijayan\Desktop\Data Science\R\Kaggle Training\Predicting Molecular Properties\train.csv')
test = pd.read_csv(r'C:\Users\darveen.vijayan\Desktop\Data Science\R\Kaggle Training\Predicting Molecular Properties\test.csv')

# EDA

#Choose the molecule
mol1 = structures[structures.molecule_name=='dsgdb9nsd_000045']
mol1
# get the coordinates in array
atoms_coord = mol1.iloc[:, 3:].values
#get the atom names
atom_name = mol1.iloc[:, 2].values
#Build the system
system = Atoms(positions=atoms_coord, symbols=atom_name)
#plot the system
ase.visualize.view(system, viewer="x3d")

#Preprocessing

#helper function to crop the datasets to take the first n number molecules in train
def data_crop(data,n_molecule):
    molecule_name =train.molecule_name.unique()[0:n_molecule]
    data1 = data['molecule_name'].isin(molecule_name)
    train_crop_fin = data[data1]
    return train_crop_fin

train_crop = data_crop(train,train.molecule_name.nunique())
structures_crop = data_crop(structures,train.molecule_name.nunique())
dipole_moments_crop = data_crop(dipole_moments,train.molecule_name.nunique())
magnetic_shielding_tensors_crop = data_crop(magnetic_shielding_tensors,train.molecule_name.nunique())
mulliken_charges_crop = data_crop(mulliken_charges,train.molecule_name.nunique())
potential_energy_crop = data_crop(potential_energy,train.molecule_name.nunique())
scalar_coupling_contributions_crop = data_crop(scalar_coupling_contributions,train.molecule_name.nunique())

# Merge the train and scalar_coupling_contributions
dataset_1 = pd.merge(train_crop,scalar_coupling_contributions_crop,on=['molecule_name','atom_index_0','atom_index_1','type'],how='left')
# Merge the structures, magnetic_shielding_tensors, mulliken_charges
dataset_2_temp = pd.merge(structures_crop,magnetic_shielding_tensors_crop,on=['molecule_name','atom_index'],how='left')
dataset_2 = pd.merge(dataset_2_temp,mulliken_charges_crop,on=['molecule_name','atom_index'],how='left')
# Merge the dipole_moments and potential_energy
dataset_3 = pd.merge(dipole_moments_crop,potential_energy_crop,on=['molecule_name'],how='left')

# Feature Engineering

group = dataset_1.groupby('molecule_name').count()
group1 = pd.DataFrame(group)
group1['no_of_atoms'] = group.iloc[:,1]
dataset1_1 = pd.merge(dataset_1,group1['no_of_atoms'],how='left', on='molecule_name')

#for atom_index_0
structures_temp = structures.copy()
structures_temp.columns = ['molecule_name','atom_index_0','atom_0','x_0','y_0','z_0']
dataset1_2 = pd.merge(dataset1_1,structures_temp,how='left', on=['molecule_name','atom_index_0'])

# for atom_index_1
structures_temp_1 = structures.copy()
structures_temp_1.columns = ['molecule_name','atom_index_1','atom_1','x_1','y_1','z_1']
dataset1_3 = pd.merge(dataset1_2,structures_temp_1,how='left', on=['molecule_name','atom_index_1'])

# add more interactions for atom_0
dataset_2.columns = ['molecule_name','atom_index_0','atom_0','x','y','z','XX_0','YX_0','ZX_0','XY_0','YY_0','ZY_0','XZ_0','YZ_0','ZZ_0','mulliken_charge_0']
dataset_2_1 = dataset_2[['molecule_name','atom_index_0','atom_0','XX_0','YX_0','ZX_0','XY_0','YY_0','ZY_0','XZ_0','YZ_0','ZZ_0','mulliken_charge_0']]
dataset1_3_1 = pd.merge(dataset1_3,dataset_2_1,how='left', on=['molecule_name','atom_index_0','atom_0'])

# add more interactions for atom_1
dataset_2.columns = ['molecule_name','atom_index_1','atom_1','x','y','z','XX_1','YX_1','ZX_1','XY_1','YY_1','ZY_1','XZ_1','YZ_1','ZZ_1','mulliken_charge_1']
dataset_2_2 = dataset_2[['molecule_name','atom_index_1','atom_1','XX_1','YX_1','ZX_1','XY_1','YY_1','ZY_1','XZ_1','YZ_1','ZZ_1','mulliken_charge_1']]
dataset1_3_2 = pd.merge(dataset1_3_1,dataset_2_2,how='left', on=['molecule_name','atom_index_1','atom_1'])

#Add more interactions from dataset_3
dataset_3_1 = dataset_3.copy()
dataset_3_1.columns = ['molecule_name', 'dipole_X', 'dipole_Y', 'dipole_Z', 'potential_energy']
dataset1_3_3 = pd.merge(dataset1_3_2,dataset_3_1,how='left', on=['molecule_name'])

#Add some features to train

#for atom_index_0
structures_temp = structures.copy()
structures_temp.columns = ['molecule_name','atom_index_0',
                           'atom_0','x_0','y_0','z_0']
test_1 = pd.merge(test,structures_temp,how='left',
                   on=['molecule_name','atom_index_0'])


# for atom_index_1
structures_temp_1 = structures.copy()
structures_temp_1.columns = ['molecule_name','atom_index_1',
                             'atom_1','x_1','y_1','z_1']
test_2 = pd.merge(test_1,structures_temp_1,how='left',
                      on=['molecule_name','atom_index_1'])


group = test_2.groupby('molecule_name').count()
group1 = pd.DataFrame(group)
group1['no_of_atoms'] = group.iloc[:,1]
test_3 = pd.merge(test_2,group1['no_of_atoms'],how='left',
                      on='molecule_name')

#Cleanup test & train
train_final = dataset1_3_3[['molecule_name','type','atom_0','x_0','y_0','z_0',
                            'XX_0','YX_0', 'ZX_0', 'XY_0', 'YY_0', 'ZY_0',
                            'XZ_0','YZ_0', 'ZZ_0','mulliken_charge_0',
                            'atom_1', 'x_1', 'y_1', 'z_1',  'XX_1','YX_1', 
                            'ZX_1', 'XY_1', 'YY_1', 'ZY_1','XZ_1', 'YZ_1', 
                            'ZZ_1','mulliken_charge_1','fc', 'sd', 'pso', 
                            'dso','dipole_X','dipole_Y','dipole_Z', 
                            'potential_energy','no_of_atoms',
                            'scalar_coupling_constant'
                            ]]


test_final = test_3[['molecule_name','type', 'atom_0','x_0', 'y_0', 'z_0', 
                     'atom_1', 'x_1', 'y_1', 'z_1', 'no_of_atoms']]

#function to calculate distance
def calculateDistance(x1,y1,z1,x2,y2,z2):  
    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return dist  


#Add euclidean distance between the two atoms for train and test 
train_final['euclidean_distance'] = calculateDistance(
        train_final['x_0'].values,train_final['y_0'].values,train_final['z_0'].values,
        train_final['x_1'].values,train_final['y_1'].values,train_final['z_1'].values
        )

test_final['euclidean_distance'] = calculateDistance(
        test_final['x_0'].values,test_final['y_0'].values,test_final['z_0'].values,
        test_final['x_1'].values,test_final['y_1'].values,test_final['z_1'].values
        )
        