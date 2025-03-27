# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:51:03 2024

@author: SAI GOWTAM VALLURI

******* Machine Learning-based Electron Precipitation Model (ML-PPM) ********

This Python code contains the Machine Learning-based Electron Precipitation Model (ML-EPM).
ML-EPM is trained using the DMSP SSJ data from (1987-1988, 2000-2012). The model can predict the 
differential energy flux of electrons, total energy flux and mean energy of electrons.

ML-EPM inputs:

['SC_AACGM_LAT', 'sin_ut', 'cos_ut', 'sin_doy', 'cos_doy', 'sin_SC_AACGM_LTIME', 'cos_SC_AACGM_LTIME', 'F107',
'Bz', 'By', 'Bx', 'vsw', 'psw', 'AU', 'AL', 'SymH', 'PC',
'Bz_6hr', 'By_6hr', 'Bx_6hr', 'vsw_6hr', 'psw_6hr', 'AU_6hr', 'AL_6hr', 'SymH_6hr', 'PC_6hr',
'Bz_3hr', 'By_3hr', 'Bx_3hr', 'vsw_3hr', 'psw_3hr', 'AU_3hr', 'AL_3hr', 'SymH_3hr', 'PC_3hr',
'Bz_2hr', 'By_2hr', 'Bx_2hr', 'vsw_2hr', 'psw_2hr', 'AU_2hr', 'AL_2hr', 'SymH_2hr', 'PC_2hr',
'Bz_1hr', 'By_1hr', 'Bx_1hr', 'vsw_1hr', 'psw_1hr', 'AU_1hr', 'AL_1hr', 'SymH_1hr', 'PC_1hr',
'Bz_45min', 'By_45min', 'Bx_45min', 'vsw_45min', 'psw_45min', 'AU_45min', 'AL_45min', 'SymH_45min', 'PC_45min',
'Bz_30min', 'By_30min', 'Bx_30min', 'vsw_30min', 'psw_30min', 'AU_30min', 'AL_30min', 'SymH_30min', 'PC_30min',
'Bz_10min', 'By_10min', 'Bx_10min', 'vsw_10min', 'psw_10min', 'AU_10min', 'AL_10min', 'SymH_10min', 'PC_10min',
'StormTime']

ML-EPM outputs:
    Differential Energy Spectra of electrons, total energy flux, mean energy 
    (see examples in main part of the code)
    
Example usage of the current model: 
    ML-EPM runs with Datetime string. This function will automatically
    generates the required inputs from high-resolution NASA-OMNI and creates
    the predictions.
Note that some indices are not available in real-time, so the real-time predictions are not possible right now.
    we are working on the real time operational model, stay tuned.
    

"""

import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from nasaomnireader import omnireader
from time_hist2 import time_history
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
import keras
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import aacgmv2
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade
plt.rcParams.update({'font.size': 16})
plt.rcParams.update({'font.weight': 'bold'})
import warnings
warnings.filterwarnings("ignore")

### Create the solar wind data for whole year ####
def sw_imf_global(time1, min_mlat = 40, max_mlat = 90):
    '''
    Input time->starting time
    Output->required inputs to Run ML_PPMv0
    '''
    start_tm = time1-timedelta(minutes=420)
    end_tm = time1+timedelta(minutes=1)
    
    omniInt = omnireader.omni_interval(start_tm,end_tm,'1min')
    omniInt1 = omnireader.omni_interval(start_tm,end_tm+timedelta(minutes=59),'hourly')
    t = omniInt['Epoch'] #datetime timestamps
    Bz,By,Bx,vsw,psw,au,al,symh,PC = omniInt['BZ_GSM'],omniInt['BY_GSM'],omniInt['BX_GSE'],omniInt['flow_speed'],\
            omniInt['Pressure'],omniInt['AU_INDEX'],omniInt['AL_INDEX'],omniInt['SYM_H'], omniInt['PC_N_INDEX']
    t1 = omniInt1['Epoch']
    F107 = omniInt1['F10_INDEX']
    values1 = {'timestamps':t1,'F107':F107}
    fdf = pd.DataFrame(values1)
    fdf = fdf.set_index('timestamps')
    
    fdf = fdf.resample("5T").mean().ffill()
    fdf.drop(fdf.tail(1).index,inplace=True)
    values = {'timestamps':t,
              'Bz' : Bz,
              'By' : By,
              'Bx' : Bx,
              'vsw' : vsw,
              'psw' : psw,
              'AU' : au,
              'AL' : al,
              'SymH' : symh,
              'PC' : PC}
    df = pd.DataFrame(values)
    
    
    # Check if all values in 'df["PC"]' are NaN
    if df['PC'].isna().all():
        new_df1 = pd.read_csv('data/pcn20240510p.csv')
        new_df2 = pd.read_csv('data/pcn20240511p.csv')
        new_df3 = pd.read_csv('data/pcn20240512p.csv')
        new_df4 = pd.read_csv('data/pcn20240513p.csv')
        
        new_df = pd.concat([new_df1,new_df2,new_df3,new_df4])
        start_time = df['timestamps'].iloc[0]
        end_time = df['timestamps'].iloc[-1]
    
        new_df['date'] = pd.to_datetime(new_df['date'])
    
        # Filter the 'new_df' using the timestamp range
        filtered_df = new_df[(new_df['date'] >= start_time) & (new_df['date'] <= end_time)]
        df['PC'] = filtered_df['pcn'].values
    
    else:
        print()
    
    
    
    
    fdf = fdf.reset_index()
    
    df1 = pd.concat([df,fdf['F107']],axis = 1)
    df1 = df1.set_index('timestamps')
    df1 = df1.interpolate(method='nearest', axis=0).ffill()
    df = time_history(df1)
    
    df = df.dropna(axis=0)
        
    columns_drop = ['F107_6hr','F107_5hr','F107_4hr','F107_3hr','F107_2hr','F107_1hr',
                    'F107_55min','F107_50min','F107_45min','F107_40min','F107_35min','F107_30min',
                    'F107_25min','F107_20min','F107_15min', 'F107_10min','F107_5min']
    df = df.drop(columns_drop, axis = 1)
    df = df.reset_index()
    
    df = df.dropna(axis=0)
    
    day = df['timestamps'].dt.day_of_year

    UT = df['timestamps'].dt.hour + (df['timestamps'].dt.minute)/60

    sin_ut = np.sin((2*np.pi*UT)/24.)
    cos_ut = np.cos((2*np.pi*UT)/24.)

    df['sin_ut'] = sin_ut
    df['cos_ut'] = cos_ut
    sin_doy = np.sin(2*np.pi*day/365.)
    cos_doy = np.cos(2*np.pi*day/365.)

    df['sin_doy'] = sin_doy
    df['cos_doy'] = cos_doy

   
    df['StormTime'] = df['Bz'].apply(lambda x: 1 if x < 0 else 0)
    df = df.set_index('timestamps')
    df = df.iloc[[-1]]
    
    Nl = 50
    Nmlt = 51
    mlat = np.linspace(min_mlat,max_mlat,Nl)
    mlt = np.linspace(0,2*np.pi,Nmlt)
    mlt = mlt[0:50]
    mlt,mlat = np.meshgrid(mlt,mlat)
    mlt = np.rad2deg(mlt)/15
    SC_AACGM_LTIME = mlt.flatten()
    SC_AACGM_LAT = mlat.flatten()
    
    sin_SC_AACGM_LTIME = np.sin((2*np.pi*SC_AACGM_LTIME)/24)
    cos_SC_AACGM_LTIME = np.cos((2*np.pi*SC_AACGM_LTIME)/24)

    df = pd.DataFrame(np.repeat(df.values, 2500, axis=0),columns=df.columns)
    df['SC_AACGM_LAT'] = SC_AACGM_LAT
    df['sin_SC_AACGM_LTIME'] = sin_SC_AACGM_LTIME
    df['cos_SC_AACGM_LTIME'] = cos_SC_AACGM_LTIME
    return df

def sw_imf_local(time1,mlat,mlt):
    '''
    Input time->starting time of the year
    Output->DataFrame is one year long SW/IMF and gemagnetic indices with 6-hours
    time history (1-hour with 5-min resolution, 6 hours with 1-hour resolution)
    '''
    start_tm = time1-timedelta(minutes=420)
    end_tm = time1+timedelta(minutes=1)
    
    omniInt = omnireader.omni_interval(start_tm,end_tm,'1min')
    omniInt1 = omnireader.omni_interval(start_tm,end_tm+timedelta(minutes=59),'hourly')
    t = omniInt['Epoch'] #datetime timestamps
    Bz,By,Bx,vsw,psw,au,al,symh,PC = omniInt['BZ_GSM'],omniInt['BY_GSM'],omniInt['BX_GSE'],omniInt['flow_speed'],\
            omniInt['Pressure'],omniInt['AU_INDEX'],omniInt['AL_INDEX'],omniInt['SYM_H'], omniInt['PC_N_INDEX']
    t1 = omniInt1['Epoch']
    F107 = omniInt1['F10_INDEX']
    values1 = {'timestamps':t1,'F107':F107}
    fdf = pd.DataFrame(values1)
    fdf = fdf.set_index('timestamps')
    
    fdf = fdf.resample("5T").mean().ffill()
    fdf.drop(fdf.tail(1).index,inplace=True)
    values = {'timestamps':t,
              'Bz' : Bz,
              'By' : By,
              'Bx' : Bx,
              'vsw' : vsw,
              'psw' : psw,
              'AU' : au,
              'AL' : al,
              'SymH' : symh,
              'PC' : PC}
    df = pd.DataFrame(values)
    
    # Check if all values in 'df["PC"]' are NaN
    if df['PC'].isna().all():
        new_df1 = pd.read_csv('data/pcn20240510p.csv')
        new_df2 = pd.read_csv('data/pcn20240511p.csv')
        new_df3 = pd.read_csv('data/pcn20240512p.csv')
        new_df4 = pd.read_csv('data/pcn20240513p.csv')
        
        new_df = pd.concat([new_df1,new_df2,new_df3,new_df4])
        start_time = df['timestamps'].iloc[0]
        end_time = df['timestamps'].iloc[-1]
    
        new_df['date'] = pd.to_datetime(new_df['date'])
    
        # Filter the 'new_df' using the timestamp range
        filtered_df = new_df[(new_df['date'] >= start_time) & (new_df['date'] <= end_time)]
        df['PC'] = filtered_df['pcn'].values
    
    else:
        print()
    
    fdf = fdf.reset_index()
    
    df1 = pd.concat([df,fdf['F107']],axis = 1)
    df1 = df1.set_index('timestamps')
    df1 = df1.interpolate(method='nearest', axis=0).ffill()
    df = time_history(df1)
    
    df = df.dropna(axis=0)
        
    columns_drop = ['F107_6hr','F107_5hr','F107_4hr','F107_3hr','F107_2hr','F107_1hr',
                    'F107_55min','F107_50min','F107_45min','F107_40min','F107_35min','F107_30min',
                    'F107_25min','F107_20min','F107_15min', 'F107_10min','F107_5min']
    df = df.drop(columns_drop, axis = 1)
    df = df.reset_index()
    df = df.dropna(axis=0)
    
    day = df['timestamps'].dt.day_of_year

    UT = df['timestamps'].dt.hour + (df['timestamps'].dt.minute)/60

    sin_ut = np.sin((2*np.pi*UT)/24.)
    cos_ut = np.cos((2*np.pi*UT)/24.)

    df['sin_ut'] = sin_ut
    df['cos_ut'] = cos_ut
    sin_doy = np.sin(2*np.pi*day/365.)
    cos_doy = np.cos(2*np.pi*day/365.)

    df['sin_doy'] = sin_doy
    df['cos_doy'] = cos_doy

   
    df['StormTime'] = df['Bz'].apply(lambda x: 1 if x < 0 else 0)
    df = df.set_index('timestamps')
    if not df.empty:
        df = df.iloc[[-1]]
        
        
        SC_AACGM_LTIME = mlt
        SC_AACGM_LAT = mlat
        
        sin_SC_AACGM_LTIME = np.sin((2*np.pi*SC_AACGM_LTIME)/24)
        cos_SC_AACGM_LTIME = np.cos((2*np.pi*SC_AACGM_LTIME)/24)
    
        df['SC_AACGM_LAT'] = SC_AACGM_LAT
        df['sin_SC_AACGM_LTIME'] = sin_SC_AACGM_LTIME
        df['cos_SC_AACGM_LTIME'] = cos_SC_AACGM_LTIME
    else:
        df = pd.DataFrame()
    return df

### Load Model, scaler_X and required inputs ###
def ML_PPM(m):

    # Define input features and output channels
    input_SW_IMF = ['SC_AACGM_LAT', 'sin_ut', 'cos_ut', 'sin_doy', 'cos_doy', 'sin_SC_AACGM_LTIME', 'cos_SC_AACGM_LTIME', 'F107',
                    'Bz', 'By', 'Bx', 'vsw', 'psw', 'AU', 'AL', 'SymH', 'PC',
                    'Bz_6hr', 'By_6hr', 'Bx_6hr', 'vsw_6hr', 'psw_6hr', 'AU_6hr', 'AL_6hr', 'SymH_6hr', 'PC_6hr',
                    'Bz_3hr', 'By_3hr', 'Bx_3hr', 'vsw_3hr', 'psw_3hr', 'AU_3hr', 'AL_3hr', 'SymH_3hr', 'PC_3hr',
                    'Bz_2hr', 'By_2hr', 'Bx_2hr', 'vsw_2hr', 'psw_2hr', 'AU_2hr', 'AL_2hr', 'SymH_2hr', 'PC_2hr',
                    'Bz_1hr', 'By_1hr', 'Bx_1hr', 'vsw_1hr', 'psw_1hr', 'AU_1hr', 'AL_1hr', 'SymH_1hr', 'PC_1hr',
                    'Bz_45min', 'By_45min', 'Bx_45min', 'vsw_45min', 'psw_45min', 'AU_45min', 'AL_45min', 'SymH_45min', 'PC_45min',
                    'Bz_30min', 'By_30min', 'Bx_30min', 'vsw_30min', 'psw_30min', 'AU_30min', 'AL_30min', 'SymH_30min', 'PC_30min',
                    'Bz_10min', 'By_10min', 'Bx_10min', 'vsw_10min', 'psw_10min', 'AU_10min', 'AL_10min', 'SymH_10min', 'PC_10min',
                    'StormTime']

    chanelsE = ['Ech1', 'Ech2', 'Ech3', 'Ech4', 'Ech5',
                'Ech6', 'Ech7', 'Ech8', 'Ech9', 'Ech10',
                'Ech11', 'Ech12', 'Ech13', 'Ech14', 'Ech15',
                'Ech16', 'Ech17', 'Ech18', 'Ech19']
    
    chanelsI = ['Ich1', 'Ich2', 'Ich3', 'Ich4', 'Ich5',
                'Ich6', 'Ich7', 'Ich8', 'Ich9', 'Ich10',
                'Ich11', 'Ich12', 'Ich13', 'Ich14', 'Ich15',
                'Ich16', 'Ich17', 'Ich18', 'Ich19']
    
    keras.saving.get_custom_objects().clear()
    # custom activation and custom_loss
    @keras.saving.register_keras_serializable(package="my_pakage", name="custom_activation")
    def custom_activation(x):
        return tf.where(x > 0, x, tf.zeros_like(x))
    
    
    @keras.saving.register_keras_serializable(package="my_loss", name="custom_loss")
    # Compile the model with custom loss
    def custom_loss(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)  # Ensure y_pred is of type tf.float32
        zero_mask = tf.equal(y_true, 0)
        non_zero_mask = tf.math.logical_not(zero_mask)
    
        mse_loss = tf.reduce_mean(tf.square(tf.boolean_mask(y_true, non_zero_mask) - tf.boolean_mask(y_pred, non_zero_mask)))
    
        # non_zero_loss = tf.reduce_mean(tf.pow(tf.abs(tf.boolean_mask(y_true, non_zero_mask) - tf.boolean_mask(y_pred, non_zero_mask)), 4))
    
        # Custom loss term for zero values (e.g., using absolute difference)
        zero_loss = tf.reduce_mean(tf.abs(tf.boolean_mask(y_true, zero_mask) - tf.boolean_mask(y_pred, zero_mask)))
        # Combine the losses using weighted sum or any desired strategy
        total_loss = mse_loss + zero_loss
        return total_loss
    if m == 'ele':
        # Load the scaler from the file
        with open('data/input_scaler_sw_imf_ele.pkl', 'rb') as handle:
            scaler_X = pickle.load(handle)
        def build_resnet_model(input_shape):
            # Input layer
            inputs = Input(shape=input_shape)
            
            x = Dense(256,activation='relu')(inputs)
            x = BatchNormalization()(x)
            x = Dropout(0.1)(x)
            
            x = Dense(512,activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.1)(x)
            
            x = Dense(64,activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.1)(x)
            
            x = Dense(256,activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.1)(x)
            
            x = Dense(32,activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.1)(x)
            
            outputs = Dense(19, activation=custom_activation)(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            return model

        input_shape = (81,)
        model = build_resnet_model(input_shape)
        # Loading ML-PPM electron precipiation model
        loaded_weights = np.load('data/ele.npz')
        # Convert the weights back to a list
        wt_list = [loaded_weights[key] for key in sorted(loaded_weights.files, key=lambda x: int(x.split('_')[1]))]
        # Set the weights to the model
        model.set_weights(wt_list)
        return model, input_SW_IMF, chanelsE, scaler_X

def process_energy_flux(diff_flux):
    """
    Process a single 19-channel array or an N x 19 array/dataframe of differential energy flux values to
    calculate total energy flux, total number flux, and average energy for each event.
    
    Parameters:
    diff_flux (numpy array or pandas dataframe): Differential energy flux values for one or multiple events.
    
    Returns:
    pandas dataframe: DataFrame containing total energy flux, total number flux, and average energy for each event.
    """
    
    # DMSP electron energy levels
    E_levels = [30000, 20400, 13900, 9450, 6460, 4400, 3000, 2040, 1392, 949, 
                646, 440, 300, 204, 139, 95, 65, 44, 30]
    
    def calculate_energy_flux_and_avg(diff_flux_row):
        """
        Calculate Total Energy Flux and Average Energy based on a single row of differential energy flux values.
        
        Parameters:
        diff_flux_row (list or numpy array): Differential energy flux values for a single event.
        
        Returns:
        tuple: Total energy flux, total number flux, and average energy.
        """
        
        def j_E(E_i):
            index = E_levels.index(E_i)
            return diff_flux_row[index]
        
        def j_N(E_i):
            index = E_levels.index(E_i)
            return diff_flux_row[index] / E_i

        # Calculate Total Energy Flux
        J_ETotal = j_E(E_levels[0]) * (E_levels[1] - E_levels[0])
        for i in range(1, len(E_levels)-1):
            J_ETotal += j_E(E_levels[i]) * (E_levels[i+1] - E_levels[i-1]) / 2
        J_ETotal += j_E(E_levels[-1]) * (E_levels[-1] - E_levels[-2])
        J_ETotal *= -1
        
        # Calculate Total Number Flux
        J_NTotal = j_N(E_levels[0]) * (E_levels[1] - E_levels[0])
        for i in range(1, len(E_levels)-1):
            J_NTotal += j_N(E_levels[i]) * (E_levels[i+1] - E_levels[i-1]) / 2
        J_NTotal += j_N(E_levels[-1]) * (E_levels[-1] - E_levels[-2])
        J_NTotal *= -1

        # Calculate Average Energy
        avg_energy = J_ETotal / J_NTotal if J_NTotal != 0 else None

        return J_ETotal, J_NTotal, avg_energy
    
    if isinstance(diff_flux, np.ndarray):
        if diff_flux.ndim == 1:  # Single 19-channel array
            diff_flux = pd.DataFrame([diff_flux])  # Convert to DataFrame
        elif diff_flux.ndim == 2:  # N x 19 array
            diff_flux = pd.DataFrame(diff_flux)  # Convert to DataFrame
        else:
            raise ValueError("The input numpy array must be 1D or 2D with shape (19,) or (N, 19).")
    elif isinstance(diff_flux, pd.DataFrame):
        if diff_flux.shape[1] != 19:
            raise ValueError("The DataFrame must have exactly 19 columns corresponding to the energy levels.")
    else:
        raise TypeError("The input must be a numpy array or pandas DataFrame.")

    results = diff_flux.apply(calculate_energy_flux_and_avg, axis=1)
    results_df = pd.DataFrame(results.tolist(), columns=['Total Energy Flux', 'Total Number Flux', 'Average Energy'])
    
    return results_df

def gen_global_data(time1, min_mlat = 40, max_mlat = 90, m=None):
    chanelsE = ['Ech1', 'Ech2', 'Ech3', 'Ech4', 'Ech5',
                'Ech6', 'Ech7', 'Ech8', 'Ech9', 'Ech10',
                'Ech11', 'Ech12', 'Ech13', 'Ech14', 'Ech15',
                'Ech16', 'Ech17', 'Ech18', 'Ech19']
    
    chanelsI = ['Ich1', 'Ich2', 'Ich3', 'Ich4', 'Ich5',
                'Ich6', 'Ich7', 'Ich8', 'Ich9', 'Ich10',
                'Ich11', 'Ich12', 'Ich13', 'Ich14', 'Ich15',
                'Ich16', 'Ich17', 'Ich18', 'Ich19']
    Nl = 50
    Nmlt = 51
    mlat = np.linspace(min_mlat, max_mlat, Nl)
    mlt = np.linspace(0, 2*np.pi, Nmlt)
    mlt = mlt[0:50]
    mlt, mlat = np.meshgrid(mlt, mlat)
    mlt = np.rad2deg(mlt) / 15
    mlong = mlt * 15
    mlong = np.deg2rad(mlong)
    # Energy = [25013.5, 17389.2, 12088.8, 8404.1, 5842.4, 4061.6, 2823.6, 1963.0, 1364.6,
    #           948.7, 659.5, 458.5, 318.7, 221.6, 154.0, 107.1, 74.4, 51.8, 36.0]
    if m=='ele':
        model, input_SW_IMF, chanelsE, scaler_X = ML_PPM('ele')
        
        df = sw_imf_global(time1, min_mlat, max_mlat)
        inputs = df[input_SW_IMF]
        inputs_scaled = scaler_X.transform(inputs.values)
        
        predictions = model.predict(inputs_scaled)
        predictions = pd.DataFrame(predictions)
        predictions.columns = chanelsE
        predictions[predictions < 5.6] = 0
        predictions = predictions.applymap(lambda x: 10**x if x != 0 else x)
        
        tmp = process_energy_flux(predictions)
        tmp['Total Energy Flux'] = tmp['Total Energy Flux'] * 1.602 * 10**-12 
        predictions = pd.concat([predictions, tmp], axis=1)
        fig1, axes = plt.subplots(4, 5, figsize=(20, 20), dpi=300, subplot_kw=dict(projection='polar'))

        axes = axes.flatten()
        vmin = 0.0
        vmax = 7e7
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        # axes = np.delete(axes, -1)
        selected_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

        qcs_list = []

        for i, ch in enumerate(selected_channels):
            ch_data = np.array(predictions[f'Ech{ch}'])
            ch_data = np.reshape(ch_data, [100, 100])
            ax = axes[i]
            qcs = ax.pcolormesh(mlong, 90 - mlat, ch_data, cmap='plasma', norm=norm)
            ax.set_theta_zero_location("S")
            ax.set_ylim(0, 50)
            ax.xaxis.grid(linestyle='--', linewidth=0.8)
            ax.yaxis.grid(linestyle='--', linewidth=0.8)
            xtickpos = [0.0, 0.78539816, 1.57079633, 2.35619449, 3.14159265, 3.92699082, 4.71238898, 5.49778714]
            xticks = ['00', '03', '06', '09', '12', '15', '18', '21']
            ax.set_xticks(xtickpos, xticks, fontweight="bold",fontsize = 16)
            ytickpos = [10.0, 20.0, 30.0, 40.0, 50.0]
            yticks = ['80$^\circ$', '70$^\circ$', '60$^\circ$', '50$^\circ$', '40$^\circ$']
            ax.set_yticks(ytickpos, yticks)
            ax.set_title(f'Ch{ch}: {Energy[ch - 1]}(eV)', y=1.1, fontweight="bold", fontsize=20)
            qcs_list.append(qcs)

        axes[19].axis('off')
        plt.subplots_adjust(bottom=0.01, top=0.95, left=0.05, right=0.95, hspace=0.5)
        cbar = fig1.colorbar(qcs_list[0], ax=axes[:19], orientation='horizontal', pad=0.05, shrink=0.5)
        cbar.set_label(r'eV/cm$^2$.s')
        plt.show()
        
        fig2, axes = plt.subplots(figsize=(10, 10), dpi=300, subplot_kw=dict(projection='polar'))
        vmin = 0.0
        vmax = 2.0
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        selected_channels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        ch_data = np.array(predictions['Total Energy Flux'])
        ch_data = np.reshape(ch_data, [100, 100])
        ax = axes
        qcs = ax.pcolormesh(mlong, 90 - mlat, ch_data, cmap='plasma', norm = norm)
        ax.set_theta_zero_location("S")
        ax.set_ylim(0, 40)
        ax.xaxis.grid(linestyle='--', linewidth=0.8)
        ax.yaxis.grid(linestyle='--', linewidth=0.8)
        xtickpos = [0.0, 0.78539816, 1.57079633, 2.35619449, 3.14159265, 3.92699082, 4.71238898, 5.49778714]
        xticks = ['00', '03', '06', '09', '12', '15', '18', '21']
        ax.set_xticks(xtickpos, xticks, fontweight="bold",fontsize = 16)
        ytickpos = [10.0, 20.0, 30.0, 40.0]
        yticks = ['80$^\circ$', '70$^\circ$', '60$^\circ$', '50$^\circ$']
        ax.set_yticks(ytickpos, yticks)
        ax.set_title('Electron Total Energy Flux', y=1.1, fontweight="bold", fontsize=20)
        cbar = fig2.colorbar(qcs, ax=ax, orientation='horizontal', pad=0.05, shrink=0.5)
        cbar.set_label(r'mW/m$^2$')
        plt.show()
        
        return df, inputs, inputs_scaled, model, mlt, mlat, predictions

def gen_spectrum_local(time1, mlat, mlt, m=None):
    if m=='ele':
        model, input_SW_IMF, chanelsE, scaler_X = ML_PPM('ele')
        df = sw_imf_local(time1, mlat, mlt)
        inputs = df[input_SW_IMF]
        inputs_scaled = scaler_X.transform(inputs.values)
        
        predictions = model.predict(inputs_scaled)
        predictions = pd.DataFrame(predictions)
        predictions.columns = chanelsE
        predictions[predictions < 5.6] = 0
        predictions = predictions.applymap(lambda x: 10**x if x != 0 else x)
        
        # Total Energy Flux, Total Number Flux, Mean Energy
        Total_mean = process_energy_flux(predictions)
                
        
        predictions = np.squeeze(predictions)
        predictions = pd.DataFrame(predictions).T
        predictions.index = inputs.index
        Total_mean.index = inputs.index
        df = pd.concat([inputs, predictions, Total_mean], axis=1)
        return df, inputs, inputs_scaled, model

def plot_flux_map(predictions, timestamp, mlt_grid, mlat_grid):
    """
    Plot electron flux predictions using pcolormesh on a global map.
    
    Args:
        predictions: 2D numpy array of flux values (MLAT x MLT grid).
        timestamp: datetime object for AACGMv2 transformations.
        mlt_grid: 2D numpy array representing Magnetic Local Time grid.
        mlat_grid: 2D numpy array representing Magnetic Latitude grid.
    """
    
    
    # Ensure the grids match the predictions array
    if mlt_grid.shape != predictions.shape or mlat_grid.shape != predictions.shape:
        raise ValueError("Shapes of mlt_grid, mlat_grid, and predictions must match.")
    
    # Convert MLT to MLON
    mlt = mlt_grid[0, :]  # Extract 1D MLT from the grid
    mlon_1D = aacgmv2.convert_mlt(mlt, timestamp, m2a=True)
    mlon_grid = np.tile(mlon_1D, (mlat_grid.shape[0], 1))

    # Flatten MLAT and MLON for geographic conversion
    mlat_flat = mlat_grid.flatten()
    mlon_flat = mlon_grid.flatten()

    # Convert MLAT/MLON to geographic coordinates
    glat_flat, glon_flat, _ = aacgmv2.convert_latlon_arr(
        mlat_flat, mlon_flat, 100, timestamp, method_code="A2G"
    )

    # Reshape geographic coordinates to 2D
    glat_grid = glat_flat.reshape(mlat_grid.shape)
    glon_grid = glon_flat.reshape(mlon_grid.shape)



    coastcolor, bordercolor = 'white', 'white'
    coastalpha, borderalpha = 0.4, 0.5

    # Create figure
    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={"projection": ccrs.Orthographic(-110, 70)},
        dpi=300,
    )
    ax.set_global()

    # Add map features
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.coastlines('10m', color=coastcolor, alpha=coastalpha)
    borders = cfeature.NaturalEarthFeature(
        category='cultural', name='admin_0_countries', scale='10m', facecolor='none'
    )
    ax.add_feature(borders, edgecolor=bordercolor, alpha=borderalpha)
    
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.75, linestyle='--')
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 30)) 
    gl.ylocator = plt.FixedLocator(np.arange(-90, 91, 10))   
    gl.top_labels = False 
    gl.right_labels = False 
    gl.bottom_labels = False
    gl.left_labels = False 
    
    latitudes = np.arange(-90, 91, 20)
    for lat in latitudes:
        ax.text(
            -179, lat, f"{lat}°",  
            transform=ccrs.PlateCarree(),
            ha='left', va='center',
            fontsize=11, color='gray', alpha=0.7
        )

    vmin, vmax = 0.0, 2.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(
        glon_grid,
        glat_grid,
        predictions,
        transform=ccrs.PlateCarree(),
        norm=norm,
        cmap='plasma',
        shading='auto',
        alpha=0.8
    )

    # Add colorbar
    cbar = fig.colorbar(mesh, orientation='horizontal', pad=0.05, shrink=0.5)
    cbar.set_label(r'Electron Flux (mW/m$^2$)', fontsize=14, fontweight='bold')
    
    # **Auroral Boundary Detection**
    threshold = 0.2 
    eq_edges, pol_edges = [], []

    for i in range(mlt_grid.shape[1]):  # Loop over MLT
        tef_column = predictions[:, i]
        above_thresh = np.where(tef_column > threshold)[0]

        if len(above_thresh) > 0:
            eq_edges.append((mlat_grid[above_thresh[0], i], mlon_grid[above_thresh[0], i]))
            pol_edges.append((mlat_grid[above_thresh[-1], i], mlon_grid[above_thresh[-1], i]))
        else:
            max_idx = np.argmax(tef_column)
            pol_edges.append((mlat_grid[max_idx, i], mlon_grid[max_idx, i]))

    # Convert boundaries to geographic coordinates
    if eq_edges and pol_edges:
        eq_edges = np.array(eq_edges)
        pol_edges = np.array(pol_edges)

        eq_glat, eq_glon, _ = aacgmv2.convert_latlon_arr(eq_edges[:, 0], eq_edges[:, 1], 100, timestamp, method_code="A2G")
        pol_glat, pol_glon, _ = aacgmv2.convert_latlon_arr(pol_edges[:, 0], pol_edges[:, 1], 100, timestamp, method_code="A2G")

        # **Plot auroral boundaries**
        ax.scatter(eq_glon, eq_glat, color='red', s=10, transform=ccrs.PlateCarree(), label='Equatorward Edge')
        ax.scatter(pol_glon, pol_glat, color='green', s=10, transform=ccrs.PlateCarree(), label='Poleward Edge')
    
    ax.set_title(f'ML-EPM: Electron Total Ener. Flux Predictions\n{timestamp.strftime("%Y-%m-%d %H:%M UT")}', fontsize=16, fontweight='bold', y=1.05)
    ax.add_feature(Nightshade(timestamp, alpha=0.3))
    plt.show()
    return fig


if __name__== "__main__":

    time1 = datetime(2014,2,27,18,0,0)
    # df, inputs, inputs_scaled, model = gen_spectrum_local(time1, 65, 0, m='ele')
    
    df, inputs, inputs_scaled, model, mlt, mlat, predictions_ele = gen_global_data(time1, min_mlat = 40, max_mlat = 90, m='ele')
    tef = predictions_ele['Total Energy Flux'].values
    tef = np.reshape(tef, [50, 50])
    fig = plot_flux_map(tef, time1, mlt, mlat)
