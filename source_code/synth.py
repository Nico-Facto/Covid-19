import pandas as  pd
import numpy as np 

class data_synth():

    def __init__(self,df_cy,df_ph):
        self.df_cy = df_cy
        self.df_ph = df_ph

    def synth_cy(self):
        
        m_nb_obs = len(self.df_cy)
        m_tot_dur = round(sum(self.df_cy['duration']),2)
        m_tot_auc = round(sum(self.df_cy['auc_to_mean']),2)
        return m_nb_obs, m_tot_dur, m_tot_auc

    def synth_ph(self):

        df_ph_asc = self.df_ph[self.df_ph['cat'].isin([1])]
        df_ph_des = self.df_ph[self.df_ph['cat'].isin([-1])]

        m_nb_obs_asc = len(df_ph_asc)
        m_tot_dur_asc = round(sum(df_ph_asc['diff_t']),2)
        m_tot_auc_asc = round(sum(df_ph_asc['auc_to_min']),2)

        m_nb_obs_des = len(df_ph_des)
        m_tot_dur_des = round(sum(df_ph_des['diff_t']),2)
        m_tot_auc_des = round(sum(df_ph_des['auc_to_min']),2)

        return m_nb_obs_asc, m_tot_dur_asc, m_tot_auc_asc, m_nb_obs_des, m_tot_dur_des, m_tot_auc_des