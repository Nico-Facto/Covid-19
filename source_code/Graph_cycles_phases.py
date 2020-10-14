import pandas as pd
import numpy as np

import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

class cycles_phases():
    
    def __init__(self,full_df,user_id,col_id,col_date,col_target):
        self.user_id = user_id
        self.col_id = str(col_id)
        self.col_date = str(col_date)
        self.col_target = str(col_target)

        self.full_df = full_df[full_df[self.col_id].isin([self.user_id])]
        self.full_df = self.full_df.reset_index()
        self.full_df[self.col_date] = pd.to_datetime(self.full_df[self.col_date])
        self.full_df = self.full_df.drop(columns=["index","id"])
        self.full_df = self.full_df.rename(columns={self.col_id: "id", self.col_date: "time", self.col_target: "NegA"})
        self.full_df = pd.DataFrame(self.full_df, columns=(["id", "time", "NegA"]))
        self.len_obs = len(self.full_df["NegA"])
        self.mean_val = np.mean(self.full_df["NegA"])
        self.std_val = np.std(self.full_df["NegA"])
        self.min_val = self.full_df["NegA"].min()
        self.max_val = self.full_df["NegA"].max()
         
    ## graph functions code ##
    def viz_cycles(self):
        
        inc_df = self.full_df
        
        count = 0
        mean_val = np.mean(inc_df["NegA"])
        inc_df["mean"] =  self.mean_val

        this_df_up_mean = pd.DataFrame(columns=inc_df.columns)
        break_des = False
        m_show_legend_1 = True

        fig = go.Figure()

        for i in inc_df["NegA"]:

            if i >= mean_val :
                #phase upper
                if count != 0:
                    this_df_up_mean.at[inc_df.index[count-1]] = inc_df.iloc[count-1]    
                this_df_up_mean.at[inc_df.index[count]] = inc_df.iloc[count]
            else:
                break_des = True
                this_df_up_mean.at[inc_df.index[count]] = inc_df.iloc[count]
                this_df_up_mean.at[inc_df.index[count],'NegA'] = mean_val
                this_df_up_mean.at[inc_df.index[count],'NegA'] = inc_df.at[count,"NegA"]
                if count != (self.len_obs-1):
                    this_df_up_mean.at[inc_df.index[count+1],'NegA'] = mean_val

            if break_des:
                break_des = False
                fig.add_scattergl(x=this_df_up_mean.index, y=(this_df_up_mean['NegA']), line={'color': 'white'}, 
                                  fill="tozeroy",fillcolor="red", name="Upper mean",mode='lines+markers',
                                  legendgroup='Upper mean',showlegend=m_show_legend_1)
                this_df_up_mean = pd.DataFrame(columns=inc_df.columns)
                if m_show_legend_1: ## only first occurence then we not show other legend
                    m_show_legend_1 = False

            count +=1
            
        if break_des == False:               
            fig.add_scattergl(x=this_df_up_mean.index, y=(this_df_up_mean['NegA']), line={'color': 'white'}, 
                                    fill="tozeroy",fillcolor="red", name="Upper mean",mode='lines+markers',
                                    legendgroup='Upper mean',showlegend=m_show_legend_1)
            this_df_up_mean = pd.DataFrame(columns=inc_df.columns)
            if m_show_legend_1: ## only first occurence then we not show other legend
                m_show_legend_1 = False

            # Line Horizontal
        fig.add_scattergl(x=inc_df.index, y=(inc_df['mean']),mode='lines',line={'color': 'red'},name='Mean Level',
                          fill="tozeroy", fillcolor="white",showlegend=False)

        fig.add_scattergl(x=inc_df.index, y=inc_df.NegA,line={'color': 'blue'}, 
                          name="Under mean",mode='lines+markers',legendgroup='Under mean')

        fig.update_layout(title="Cycles",xaxis=dict(title="Segment since the start of ESM"),
                          yaxis=dict(title="Intensity of negative affect"),showlegend=True) 

        ### trying to fix the graph display, we have large possibility 
        if self.max_val < 10000:
            y_plt_max = 200
        else:
            y_plt_max = 2000  
        fig.update_yaxes(range=[0, (self.max_val+y_plt_max)])
        fig.update_xaxes(range=[0, self.len_obs-1])

        return fig
    
    def viz_phases(self):

        inc_df = self.full_df
        count = 0

        this_df_asc_ph = pd.DataFrame(columns=inc_df.columns)
        this_df_des_ph = pd.DataFrame(columns=inc_df.columns)
        break_asc = False
        break_des = False
        m_show_legend_1 = True
        m_show_legend_2 = True
        last_i = inc_df.at[0,"NegA"]

        ## 0 asc , 1 desc
        last_ph = 0

        fig = go.Figure()

        for i in inc_df["NegA"]:

            if i > last_i:
                last_ph = 0
            elif i < last_i:
                last_ph = 1

            if i >= last_i and last_ph == 0:
                #phase asd
                if count != 0:
                    this_df_asc_ph.at[inc_df.index[count-1]] = inc_df.iloc[count-1]    
                this_df_asc_ph.at[inc_df.index[count]] = inc_df.iloc[count]
                break_des = True

            elif i <= last_i and last_ph == 1:  
                #phase desc
                this_df_des_ph.at[inc_df.index[count-1]] = inc_df.iloc[count-1]
                this_df_des_ph.at[inc_df.index[count]] = inc_df.iloc[count]
                break_asc = True

            if break_des and last_ph == 0:
                break_des = False
                fig.add_scattergl(x=this_df_asc_ph.index, y=this_df_asc_ph.NegA, line={'color': 'blue'}, 
                          fill="tozeroy",fillcolor="red",
                          mode='lines+markers',name="Asc Segment",legendgroup='Asc Segment',showlegend=m_show_legend_1)
                this_df_asc_ph = pd.DataFrame(columns=inc_df.columns)
                if m_show_legend_1: ## only first occurence then we not show other legend
                    m_show_legend_1 = False

            if break_asc and last_ph == 1:
                break_asc = False  
                fig.add_scattergl(x=this_df_des_ph.index, y=this_df_des_ph.NegA,
                         line={'color': 'yellow'}, fill="tozeroy", fillcolor="blue",
                         mode='lines+markers',name="Desc Segment",legendgroup='Desc Segment',showlegend=m_show_legend_2)
                this_df_des_ph = pd.DataFrame(columns=inc_df.columns)
                if m_show_legend_2: ## only first occurence then we not show other legend
                    m_show_legend_2 = False
            last_i = i
            count +=1
        ## if the last seg is == we have to plot them 
        fig.add_scattergl(x=this_df_asc_ph.index, y=this_df_asc_ph.NegA, line={'color': 'blue'}, 
              fill="tozeroy",fillcolor="red",
              mode='lines+markers',name="Asc Segment",legendgroup='Asc Segment',showlegend=False)

        fig.update_layout(title="Phase",
                   xaxis= dict(title="Segment since the start of ESM"),
                   yaxis=dict(title="Intensity of negative affect"))

        if self.max_val < 10000:
            y_plt_max = 200
        else:
            y_plt_max = 2000  
        fig.update_yaxes(range=[0, (self.max_val+y_plt_max)])
        fig.update_xaxes(range=[0, (self.len_obs-1)])
        return fig

    ## matrice functions code ##
    def matrice_seg(self):
        
        ff = self.full_df
        # is_mean = self.mean_val
        is_min = self.min_val

        count = 0
        user_id = self.user_id
        new_df = pd.DataFrame()

        for t in ff["time"]:
            
            if count == (self.len_obs-1):
                # new_df["day"] = ff["day"]
                # new_df["phase_nb"] = self.full_df['phase_nb']
                return new_df

            new_df.at[count,"id"] = user_id

            t2 = ff.at[count,'time_cumul']
            t3 = ff.at[count+1,'time_cumul']

            diff_t = t3-t2
            new_t2 = diff_t + t2

            new_df.at[count,"t1"] = t2
            new_df.at[count,"t2"] = new_t2
            new_df.at[count,"diff_t"] = diff_t

            y1 = ff.at[count,"NegA"]
            y2 = ff.at[count+1,"NegA"]

            new_df.at[count,"y1"] = y1
            new_df.at[count,"y2"] = y2

            new_df.at[count,"diff_y"] = new_df.at[count,"y2"] - new_df.at[count,"y1"]
            # new_df.at[count,"slope"] = new_df.at[count,"diff_y"]/new_df.at[count,"diff_t"]

            # new_df.at[count,"intercept"] = (new_t2*y1 - t2*y2) / diff_t

            # new_df.at[count,"auc_to_mean"] = np.trapz(x = (t2, new_t2),
            #                                            y = (y1 - is_mean, y2 - is_mean))

            new_df.at[count,"auc_to_min"] = np.trapz(x =(t2, new_t2),
                                                      y = (y1 - is_min, y2 - is_min))

            ## if true cut at mean desending -1 else 0 or 1 if cut at mean ascending
            # if int(y2 > is_mean) == 0 and int(y1 > is_mean) == 1:
            #     new_df.at[count,"mean_inter"] = -1
            # else:    
            #     new_df.at[count,"mean_inter"] = int((y2 > is_mean) ^ (y1 > is_mean))

            new_df.at[count,"segment_nb"] = count+1
            count+=1
      
    def matrice_cy(self,seg_data):
        
        in_cycle = True
        start_in_cycle = False
        Last_cycle_not_over = False
        
        inc_df = self.full_df
        this_df = pd.DataFrame(columns=inc_df.columns)
        new_df = pd.DataFrame()

        count = 0
        count_cy = 0
        y1 = round(self.mean_val,2)

        for i in inc_df["NegA"]:
            
            if i > self.mean_val:
                this_df.at[count] = inc_df.iloc[count]
                in_cycle = True
                if count == 0:
                    start_in_cycle = True
            else :
                in_cycle = False
            nb_seg = len(this_df) 
            
            if count == (len(inc_df["NegA"])-1) and in_cycle :
                ## Si il reste un cycle et qu'il n'a pas été fermé on le ferme et on garde l'auc
                in_cycle = False
                Last_cycle_not_over = True

            if not in_cycle and nb_seg > 0:
                this_df = this_df.reset_index()

                if count !=0:
                    first_index = this_df.at[0,"index"]
                else :
                    ## ici on est sur un segment de 2 asc ou desc sur les 2 premiéres observations
                    first_index = 1

                if first_index == 0:
                    first_index = 1
                    
                L1 = line([inc_df.at[first_index-1,"time_cumul"],inc_df.at[first_index-1,"NegA"]], 
                          [inc_df.at[first_index,"time_cumul"],inc_df.at[first_index,"NegA"]])
                
                L2 = line([0,self.mean_val], [1000,self.mean_val])
                t1= intersection(L1, L2)
               
                L1 = line([inc_df.at[count-1,"time_cumul"],inc_df.at[count-1,"NegA"]], 
                          [inc_df.at[count,"time_cumul"],inc_df.at[count,"NegA"]])
                L2 = line([0,self.mean_val], [100,self.mean_val])
                t2= intersection(L1, L2)

                # y2 = this_df["NegA"].max()

                if count_cy == 0 and start_in_cycle: 
                    new_df.at[count_cy,"t1"] = 0.0000
                else:
                    new_df.at[count_cy,"t1"] = t1[0]
                if Last_cycle_not_over :
                    new_df.at[count_cy,"tn"] = inc_df["time_cumul"].max()
                else:
                    new_df.at[count_cy,"tn"] = t2[0]
                
                new_df.at[count_cy,"duration"] = new_df.at[count_cy,"tn"]- new_df.at[count_cy,"t1"]
                # new_df.at[count_cy,"y_min"] = y1
                # new_df.at[count_cy,"y_max"] = y2
                # new_df.at[count_cy,"amplitude"] = y2 - y1

                x_trapz = []
                y_trapz = []
                this_x_trap = this_df['time_cumul'].values
                this_y_trap = this_df['NegA'].values

                if count_cy == 0 and start_in_cycle  :   
                    x_trapz.append(0.00)
                else:
                    x_trapz.append(t1[0])
                    
                for values in this_x_trap:
                    x_trapz.append(values)
                if Last_cycle_not_over :
                    x_trapz.append(inc_df["time_cumul"].max())
                else:
                    x_trapz.append(t2[0])
                
                y_trapz.append(y1)
                for values in this_y_trap:
                    y_trapz.append(values)
                y_trapz.append(y1)
                
                auc_res = abs(np.trapz(x_trapz, y_trapz))
                new_df.at[count_cy,"auc_to_mean"] = round(auc_res,4)
                # new_df.at[count_cy,"cycle_nb"] = count_cy +1
                count_cy +=1
                this_df = pd.DataFrame(columns=inc_df.columns)
                
            count +=1
        return new_df
        
    def matrice_ph(self,seg_data):
            
        inc_df = self.full_df
        this_df = pd.DataFrame(columns=inc_df.columns)
        new_df = pd.DataFrame()
         
        count = 0
        count_ph = 0
        last_i = inc_df.at[0,"phase_statut"] 
        
        for i in inc_df["phase_statut"]:
            
            if count <= 1:
                this_df.at[inc_df.index[count]] = inc_df.iloc[count]

            else :
                if i == last_i :
                    this_df.at[inc_df.index[count-1]] = inc_df.iloc[count-1]  
                    this_df.at[inc_df.index[count]] = inc_df.iloc[count]

                else:
                    count_ph +=1
                    this_df = this_df.reset_index()

                    t1 = this_df["time_cumul"].min()
                    tn = this_df["time_cumul"].max()

                    y1 = this_df.at[0,"NegA"]
                    len_mini_obs = len(this_df)-1
                    yn = this_df.at[len_mini_obs,"NegA"]

                    # new_df.at[count,"id"] = self.user_id            
                    new_df.at[count,"t1"] = t1
                    new_df.at[count,"tn"] = tn
                    new_df.at[count,"diff_t"] = tn - t1
                    # new_df.at[count,"y1"] = y1

                    # new_df.at[count,"yn"] = yn
                    new_df.at[count,"diff_y"] = yn - y1
                    # new_df.at[count,"slope_e"] = new_df.at[count,"diff_y"] / new_df.at[count,"diff_t"]
                     
                    mask = (seg_data['t1'] >= t1) & (seg_data['t2'] <= tn)
                    auc_to_min = seg_data.loc[mask]
                    auc_to_min = sum(auc_to_min["auc_to_min"])
                    new_df.at[count,"auc_to_min"] = auc_to_min

                    ph_cat = last_i

                    new_df.at[count,"cat"] = ph_cat
                    # new_df.at[count,"phase_nb"] = count_ph 

                    new_df = new_df.reset_index()
                    new_df = new_df.drop(columns=['index'])
                    this_df = pd.DataFrame(columns=inc_df.columns)

                    if count !=0:
                        this_df.at[inc_df.index[count-1]] = inc_df.iloc[count-1] 
                    this_df.at[inc_df.index[count]] = inc_df.iloc[count]
               
            count +=1
            last_i = i 
        
        # hors loop, si il reste un df on applique la methode
        len_last_obs = len(this_df) 
        if len_last_obs >0:
            count_ph +=1
            this_df = this_df.reset_index()
            
            t1 = this_df["time_cumul"].min()
            tn = this_df["time_cumul"].max()

            # new_df.at[count,"id"] = self.user_id            
            new_df.at[count,"t1"] = t1
            new_df.at[count,"tn"] = tn
            new_df.at[count,"diff_t"] = tn - t1
            # new_df.at[count,"y1"] = this_df.at[0,"NegA"]

            y1 = this_df.at[0,"NegA"]
            len_mini_obs = len(this_df)-1
            yn = this_df.at[len_mini_obs,"NegA"]
            # new_df.at[count,"yn"] = this_df.at[len_mini_obs,"NegA"]
            new_df.at[count,"diff_y"] = yn - y1
            # new_df.at[count,"slope_e"] = new_df.at[count,"diff_y"] / new_df.at[count,"diff_t"]

            mask = (seg_data['t1'] >= t1) & (seg_data['t2'] <= tn)
            auc_to_min = seg_data.loc[mask]
            auc_to_min = sum(auc_to_min["auc_to_min"])
            new_df.at[count,"auc_to_min"] = auc_to_min

            ph_cat = last_i

            new_df.at[count,"cat"] = ph_cat
            # new_df.at[count,"phase_nb"] = count_ph 

            new_df = new_df.reset_index()
            new_df = new_df.drop(columns=['index'])
            this_df = pd.DataFrame(columns=inc_df.columns)

            this_df.at[inc_df.index[count-1]] = inc_df.iloc[count-1] 
            
        return new_df   

    ## tools for class ##
    def time_stamp(self):
        
        df = self.full_df
        print(df.columns)
        count = 0
        # day_count = 1
        # last_tl = df.at[0,'time']
        # last_tl = last_tl.date().day
       
        for t in df["time"]:
            
            # this_day = t.date().day
            # if this_day > last_tl:
            #     df.at[count,"day"] = day_count + 1
            #     day_count += 1
            #     last_tl = this_day
            # else:
            #     df.at[count,"day"] = day_count 
            
            ### End of function ###
            if count == (self.len_obs-1):
                self.full_df = df 
                data_dict = {"Id : ":self.user_id ,
                            "Observation : ": self.len_obs ,
                            "Start date : " : self.full_df.at[0,'time'] ,
                            "End date : " : self.full_df.at[self.len_obs-1,'time'] ,
                            "Mean NegA : ": round(self.mean_val,2) ,
                            "Std NegA : ": round(self.std_val,2) ,
                            "Min NegA : ": self.min_val ,
                            "Max NegA : ":self.max_val ,} 
                
                return data_dict

            if count != 0:
                t2 = df.at[count,'time']
                t3 = df.at[count+1,'time']
                diff_t = t3-t2
                diff_t = str(divmod(diff_t, 3600)[0])
                diff_t = int(diff_t.split("00:00:")[1])
                new_t2 = diff_t + last_t
                df.at[count+1,"time_cumul"] = new_t2
                last_t = new_t2
            else:
                t2 = df.at[1,"time"]
                diff_t = t2-t
                diff_t = str(divmod(diff_t, 3600)[0])
                diff_t = int(diff_t.split("00:00:")[1])
                df.at[count,"time_cumul"] = 0.0
                df.at[count+1,"time_cumul"] = diff_t
                last_t = diff_t
                
            count +=1
  
    def seg_data_ph(self):
        
        inc_df = self.full_df
        count = 0
        last_i = inc_df.at[0,"NegA"] 

        for i in inc_df["NegA"]:
            
            if count == 0:
                inc_df.at[0,"phase_statut"] = 0
                last_i = i
                count +=1
                
            if i > 1 and i > last_i :
                inc_df.at[count-1,"phase_statut"] = 1
               
            elif i > 1 and i < last_i :
                inc_df.at[count-1,"phase_statut"] = -1
                  
            elif i == 1 and i == last_i :
                inc_df.at[count-1,"phase_statut"] = 0
                
            elif i == 1 and i < last_i :
                inc_df.at[count-1,"phase_statut"] = -1
            
            elif i > 1 and i == last_i :
                if count >=2:
                    inc_df.at[count-1,"phase_statut"] = inc_df.at[count-2,"phase_statut"]
                else:     
                    inc_df.at[count-1,"phase_statut"] = inc_df.at[0,"phase_statut"]
            else:
                inc_df.at[count-1,"phase_statut"] = 0

            last_i = i
            count +=1
            
        count = 0
        last_i = inc_df.at[0,"phase_statut"]
        # count_ph = 1

        # for i in inc_df["phase_statut"]:
        #     if i != last_i :
        #         count_ph +=1
        #         inc_df.at[count,"phase_nb"] = count_ph
        #     else :
        #         inc_df.at[count,"phase_nb"] = count_ph
        #     count += 1    
        #     last_i = i        

        return self.full_df

## tools out of class
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        D = None
        
