import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
import shap

class interpret():

    def __init__(self,df):
        self.df = df

    def continent_viz(self):
        x = self.df['continent'].values
        y = self.df['new_deaths'].values
        fig = plot_affect(x,y)
        return fig

    def shap_importances(self):
        df = self.df[~self.df.location.isin(['World'])]
        df = df[df['date'].isin(["2020-09-30"])]
        feat_droped = ['id','iso_code','date','location','continent','total_tests','new_tests','new_tests_smoothed',
                          'total_tests_per_thousand','new_tests_per_thousand','new_tests_smoothed_per_thousand',
                          'new_cases','new_deaths','new_cases_smoothed','new_deaths_smoothed',
                          'total_cases_per_million','new_cases_per_million', 'new_cases_smoothed_per_million',
                          'total_deaths_per_million', 'new_deaths_per_million','new_deaths_smoothed_per_million',
                          'tests_per_case','positive_rate','total_cases','tests_units','total_deaths','stringency_index']

        y_full = df['new_deaths']
        y_full = y_full.fillna(0)
        x_full = df.drop(columns=feat_droped)
        x_full = x_full.fillna(0)
        
        preprocessor = MinMaxScaler()
        model = RandomForestRegressor(max_depth=4, min_samples_split=4, random_state=0)
        pip = make_pipeline(preprocessor, model)
        model.fit(x_full, y_full)
        mod = pip.get_params()['randomforestregressor']
        explainer = shap.TreeExplainer(mod)
        shap_values = explainer.shap_values(x_full)
        fig = shap.summary_plot(shap_values, x_full)
        # fig = plot_affect(data_to_ret[0],data_to_ret[1])
        return fig

def plot_affect(x,y):
    fig = px.bar(x=x, y=y)
    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                      marker_line_width=1.5, opacity=0.6)
    fig.update_layout()
    fig.update_layout(title="Intensity of Impact on number of death by day",
                      xaxis=dict(title="Categories observed"),
                      yaxis=dict(title="Score of Intensity"),
                      showlegend=True,
                      plot_bgcolor='rgba(0,0,0,0)') 
    return fig