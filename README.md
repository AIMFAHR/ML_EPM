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
