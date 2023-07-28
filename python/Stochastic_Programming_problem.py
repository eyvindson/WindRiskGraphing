from __future__ import division
from pyomo.environ import *
import argparse
from pyomo.opt import SolverStatus, TerminationCondition
import pandas as pd
import time
import pyutilib.services
import pickle
import random
import sqlite3
import itertools
import copy
import matplotlib.pyplot as plt
import numpy as np
from samternary.ternary import Ternary
import multiprocessing as mp
import pathos as pmp
import dill
from pathos.multiprocessing import ProcessingPool as Pool
import copy
import pandas as pd
import numpy as np
import statsmodels.api as sm
import random
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings('ignore')

#TODO -- Get cluster information and enable its use.

Results_folder = "./Results/"

import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--V',type=int,dest='V',help='which uncertainty, 0="Low_High",1="Mod_Mid",2="Mod_Low",3="High_Low",4="Low_Low"')
    parser.add_argument('--P',type=float,dest='SP',help='salvage price')
    parser.add_argument('--OPT',type=str,dest='OPT',help='Optimization problem')
    parser.add_argument('--INT',type=float,dest='INT',help='Interest Rate used')
    parser.add_argument('--SCEN',type=int,dest='SCEN',help='Number of scenarios used')
    
    args = parser.parse_args()
    INTEREST = args.INT
    SCENARIOS = args.SCEN
    np.random.seed(10)
    SALV_PRICE = args.SP
    SALVAGE = "_SALVAGE"
    if args.OPT == "NPV":
        OPT="_NPV"
    else:
        OPT = "_CVAR"
    OPT = OPT + "_"+str(SALV_PRICE)
    df = pd.DataFrame(np.random.randint(0,4,size=(100000, 6)), columns=["P1","P2","P3","P4","P5","P6"])
    df['SUM'] = df.sum(axis=1)

    df = df.drop_duplicates()
    df = df[df['SUM']<= 4]
    for kk in range(0,4):
        df['Countif='+str(kk)] = df[["P1","P2","P3","P4","P5","P6"]].apply(lambda s: (s == kk).sum(),axis=1)
    
    df = df.reset_index()
    dfb =df
    dfb = (dfb[dfb['Countif=3']!=1])
    dfb = dfb[dfb['Countif=2']!=2]
    dfb = dfb[dfb['Countif=1']!=4]
    dfb = dfb[(dfb['Countif=2']!=1)]
    dfb =  dfb[(dfb['Countif=1']<=3) & (dfb['SUM']<4) & (dfb['Countif=2']==0)& (dfb['Countif=3']==0)] # mid low
    
    ## INDICES
    Low_High =list(df[df['Countif=3']==1].index)
    Mod_Mid = list(df[(df['Countif=2']==2) & (df['Countif=3']!=1)].index)
    High_Low = list(df[(df['Countif=1']==4)  & (df['Countif=3']!=1) & (df['Countif=2']!=2)].index)
    Mod_Low = list(df[(df['Countif=2']==1) & (df['Countif=1']!=4)  & (df['Countif=3']!=1) & (df['Countif=2']!=2)].index)
    Low_Low = list(df[(df['Countif=1']<=3) & (df['SUM']<4) & (df['Countif=2']==0)& (df['Countif=3']==0) &(df['Countif=2']!=1) & (df['Countif=1']!=4)  & (df['Countif=3']!=1) & (df['Countif=2']!=2)].index)
    print(len(Low_High)+len(Mod_Mid)+len(High_Low)+len(Mod_Low)+len(Low_Low))

    ##DATA USED TO ASSESS GENERATED SOLUTION

    all_data_LH = pd.read_csv("./Data/alldata_Low_High.csv")
    all_data_LH = all_data_LH.sort_values(['id','branch','iteration','year'],ascending = [True, True,True,True])
    all_data_LH = all_data_LH.set_index(['id','branch','year','iteration'])
    all_data_LH['year'] = all_data_LH.index.get_level_values(2)

    all_data_MM = pd.read_csv("./Data/alldata_Mod_Mid.csv")
    all_data_MM = all_data_MM.sort_values(['id','branch','iteration','year'],ascending = [True, True,True,True])
    all_data_MM = all_data_MM.set_index(['id','branch','year','iteration'])
    all_data_MM['year'] = all_data_MM.index.get_level_values(2)
    
    all_data_ML = pd.read_csv("./Data/alldata_Mod_Low.csv")
    all_data_ML = all_data_ML.sort_values(['id','branch','iteration','year'],ascending = [True, True,True,True])
    all_data_ML = all_data_ML.set_index(['id','branch','year','iteration'])
    all_data_ML['year'] = all_data_ML.index.get_level_values(2)
    
    all_data_HL = pd.read_csv("./Data/alldata_High_Low.csv")
    all_data_HL = all_data_HL.sort_values(['id','branch','iteration','year'],ascending = [True, True,True,True])
    all_data_HL = all_data_HL.set_index(['id','branch','year','iteration'])
    all_data_HL['year'] = all_data_HL.index.get_level_values(2)
    
    all_data_LL = pd.read_csv("./Data/alldata_Low_Low.csv")
    all_data_LL = all_data_LL.sort_values(['id','branch','iteration','year'],ascending = [True, True,True,True])
    all_data_LL = all_data_LL.set_index(['id','branch','year','iteration'])
    all_data_LL['year'] = all_data_LL.index.get_level_values(2)
    
    #REQUIRES CBC TO BE INSTALLED ON THE MACHINE USED!!!  See https://github.com/coin-or/Cbc -- OTHER SOLVERS CAN BE USED -- 
    #THEN CHANGE this line: "opt = SolverFactory('cbc') #Here we use the cbc solver -- open source software"
    
    path = "/scratch/project_2003638/PLAN_REPLAN/DB/"#ADJUST TO OWN PATH #"c:/mytemp/avohakkutpois/Files_for_optimization/temp/"
    path_cluster = "./Data/"
    path2 = "/scratch/project_2000611/KYLE/AVO2/Files_for_optimization/"
    pyutilib.services.TempfileManager.tempdir = path2
    
    VV = [[Low_High,"Low_High"],[Mod_Mid,"Mod_Mid"],[Mod_Low,"Mod_Low"],[High_Low,"High_Low"],[Low_Low,"Low_Low"]][args.V]
    
    class optimization:
        def __init__(self):
            c = 0
            
            data_opt =  pd.read_csv("./Data/alldata2.csv")
            data_opt['branch'] = [int(str(i)[-2:]) for i in data_opt['id']]
            data_opt['id'] = [int(str(i)[:-2]) for i in data_opt['id']]
            
            CLUSTER_1 = pd.read_csv(path_cluster+"cluster_1.csv",header=None)
            CLUSTER_2 = pd.read_csv(path_cluster+"cluster_2.csv",header=None)
            CLUSTER_3 = pd.read_csv(path_cluster+"cluster_3.csv",header=None)
            
            
            for k in range(0,SCENARIOS): ##HERE WE USE 50 Scenarios --- this may need to be made a variable.
                if k == 0:
                    dat = data_opt[(data_opt['id'].isin(list(CLUSTER_1[0]))) & (data_opt["iteration"] == random.choice(VV[0])) | (data_opt['id'].isin(list(CLUSTER_2[0]))) & (data_opt["iteration"] == random.choice(VV[0]))| (data_opt['id'].isin(list(CLUSTER_3[0]))) & (data_opt["iteration"] == random.choice(VV[0]))]
                    dat['iteration']=k
                else:
                    dat1 = data_opt[(data_opt['id'].isin(list(CLUSTER_1[0]))) & (data_opt["iteration"] == random.choice(VV[0])) | (data_opt['id'].isin(list(CLUSTER_2[0]))) & (data_opt["iteration"] == random.choice(VV[0]))| (data_opt['id'].isin(list(CLUSTER_3[0]))) & (data_opt["iteration"] == random.choice(VV[0]))]
                    dat1['iteration']=k
                    dat =pd.concat([dat,dat1])
            data_opt = dat
            combinations = 1
            
            #CREATE replicates with varying iterations
            
            all_data = data_opt
            
            Index_values = all_data.set_index(['id','branch']).index.unique()
            all_data = all_data.set_index(['id','branch','year','iteration'])
            AREA = all_data.loc[slice(None),0,2016,all_data.index.get_level_values(3).min()]['AREA']
            all_data = all_data.fillna(0)
            all_data['year'] = all_data.index.get_level_values(2)
            self.data_opt=all_data
            
            self.combinations = 1
            
            #CREATE replicates with varying iterations
            
            self.all_data = self.data_opt
            
            self.Index_values = self.all_data.drop(['year'], axis=1).reset_index().set_index(['id','branch']).index.unique()#all_data.set_index(['id','branch']).index.unique()
            #self.all_data = self.all_data.set_index(['id','branch','year','iteration'])
            self.AREA = self.all_data.loc[slice(None),0,2016,self.all_data.index.get_level_values(3).min()]['AREA']
            self.all_data = self.all_data.fillna(0)
            
            self.createModel()
            
        def createModel(self):
            # Declare sets - These used to recongnize the number of stands, regimes and number of periods in the analysis.
            self.model1 = ConcreteModel()
            
            self.model1.stands = Set(initialize = list(set(self.all_data.index.get_level_values(0))))
            self.model1.year = Set(initialize = list(set(self.all_data.index.get_level_values(2))))
            self.model1.iteration = Set(initialize = list(set(self.all_data.index.get_level_values(3))))        
            self.model1.regimes = Set(initialize = list(set(self.all_data.index.get_level_values(1))))
            self.model1.scen_index = Set(initialize= [i for i in range(0,self.combinations)])
            self.model1.Index_values = self.Index_values
            
            # Indexes (stand, regime)-- excludes those combinations that have no regimes simulated
            
            def index_rule(model1):
                index = []
                for (s,r) in model1.Index_values: #stand_set
                    index.append((s,r))
                return index
            self.model1.index1 = Set(dimen=2, initialize=index_rule)
            
            self.model1.X1 = Var(self.model1.index1, within=NonNegativeReals, bounds=(0,1), initialize=1)
            
            self.all_data['year'] = self.all_data.index.get_level_values(2)
            
            #objective function:
            def outcome_rule(model1):
                return sum((self.all_data.Harvested_V.loc[(s,r,k,it)]*self.all_data.AREA.loc[(s,r,k,it)]* self.model1.X1[(s,r)])/((1+INTEREST)**(2.5+self.all_data.year[(s,r,k,it)]))  for (s,r) in self.model1.index1 for k in self.model1.year for it in self.model1.iteration)
            self.model1.OBJ = Objective(rule=outcome_rule, sense=maximize)
            
            def regime_rule(model1, s):
                row_sum = sum(model1.X1[(s,r)] for r in [x[1] for x in model1.index1 if x[0] == s])
                return row_sum == 1
            self.model1.regime_limit = Constraint(self.model1.stands, rule=regime_rule)
            
        def solve(self):
            opt = SolverFactory('cbc') #Here we use the cbc solver -- open source software
            self.results = opt.solve(self.model1,tee=False) #We solve a problem, but do not show the solver output
    
    t1 = optimization()
    
    t2 = copy.deepcopy(t1)
    
    ### MAX MINIMUM NPV:
    #Max min
    try:
        t2.model1.del_component(t2.model1.NPV_INV)
    except:
        print("NONE")
    
    t2.model1.NPV= Var(within=NonNegativeReals)
    def NPV_INVENTORY(model1,it):
        row_sum = sum(((t2.all_data.income.loc[(s,r,k,it)]+t2.all_data.natural_rm_wind.loc[(s,r,k,it)]*SALV_PRICE)*t2.all_data.AREA.loc[(s,r,k,it)]* t2.model1.X1[(s,r)])/((1+INTEREST)**(2.5+t2.all_data.year[(s,r,k,it)]-2016))  for (s,r) in t2.model1.index1 for k in t2.model1.year)+sum((t2.all_data.PV.loc[(s,r,max(t2.all_data.year[(s,r,slice(None),t2.all_data.index.get_level_values(3).min())]),it)]*t2.all_data.AREA.loc[(s,r,max(t2.all_data.year[(s,r,slice(None),t2.all_data.index.get_level_values(3).min())]),it)]* t2.model1.X1[(s,r)])/((1+INTEREST)**(2.5+max(t2.all_data.year[(s,r,slice(None),t2.all_data.index.get_level_values(3).min())])-2016))  for (s,r) in t2.model1.index1)
        return t2.model1.NPV<=row_sum
    t2.model1.NPV_INV= Constraint(t2.model1.iteration,rule=NPV_INVENTORY)
    
    try:
        t2.model1.del_component(t2.model1.EVEN_inc)
        t2.model1.del_component(t2.model1.EF)
    except:
        print("NONE")
    
    ### CVAR
    #NPV DOWNSIDE
    t2 = copy.deepcopy(t1)
    t2.model1.CVaR = Var(t2.model1.year,within=Reals, initialize=1)
    t2.model1.Z = Var(t2.model1.year,within=Reals, initialize=1)
    t2.model1.posZ = Var(t2.model1.year,t2.model1.iteration, within=NonNegativeReals, initialize=1)
    t2.model1.negZ = Var(t2.model1.year,t2.model1.iteration, within=NonNegativeReals, initialize=1)
    t2.model1.mod_L_plus = Var(t2.model1.year,t2.model1.iteration, within=NonNegativeReals, initialize=1)
    t2.model1.mod_L_neg = Var(t2.model1.year,t2.model1.iteration, within=NonNegativeReals, initialize=1)
    t2.model1.alpha_risk = Param(default=0.05, mutable=True)
    t2.model1.min_CVaR = Param(default=0, mutable=True)
    t2.model1.max_CVaR = Param(default=1, mutable=True)
    t2.model1.target = Param(default=200000, mutable=True)
    
    #CVAR Constraints LOG
    def CVAR_constraint(model1,k):
        CVaR = t2.model1.Z[k] + (1/((1-t2.model1.alpha_risk)*t2.combinations))*sum(t2.model1.posZ[k,scenario] for scenario in t2.model1.iteration)
        return t2.model1.CVaR[k] == CVaR
    t2.model1.CVAR_constraint = Constraint(t2.model1.year,rule=CVAR_constraint)
    
    def MOD_L_P_constraint(model1,k,it):
        VAL = t2.model1.mod_L_plus[k,it] -t2.model1.Z[k]+t2.model1.negZ[k,it]-t2.model1.posZ[k,it]
        return VAL ==0
    t2.model1.MOD_L_P = Constraint(t2.model1.year,t2.model1.iteration,rule=MOD_L_P_constraint)
    
    def MOD_TARGET_constraint(model1,k,it):
        row_sum = sum(((t2.all_data.income.loc[(s,r,k,it)]+t2.all_data.natural_rm_wind.loc[(s,r,k,it)]*SALV_PRICE)*t2.all_data.AREA.loc[(s,r,k,it)]* t2.model1.X1[(s,r)]) for (s,r) in t2.model1.index1)
        VAL = t2.model1.target-row_sum - t2.model1.mod_L_plus[k,it]+t2.model1.mod_L_neg[k,it]
        return VAL == 0
    t2.model1.MOD_Target = Constraint(t2.model1.year,t2.model1.iteration,rule=MOD_TARGET_constraint)
    
    t2.model1.NPV= Var(within=NonNegativeReals)
    def NPV_INVENTORY(model1,it):
        row_sum = sum(((t2.all_data.income.loc[(s,r,k,it)]+t2.all_data.natural_rm_wind.loc[(s,r,k,it)]*SALV_PRICE)*t2.all_data.AREA.loc[(s,r,k,it)]* t2.model1.X1[(s,r)])/((1+INTEREST)**(2.5+t2.all_data.year[(s,r,k,it)]-2016))  for (s,r) in t2.model1.index1 for k in t2.model1.year)+sum((t2.all_data.PV.loc[(s,r,max(t2.all_data.year[(s,r,slice(None),t2.all_data.index.get_level_values(3).min())]),it)]*t2.all_data.AREA.loc[(s,r,max(t2.all_data.year[(s,r,slice(None),t2.all_data.index.get_level_values(3).min())]),it)]* t2.model1.X1[(s,r)])/((1+INTEREST)**(2.5+max(t2.all_data.year[(s,r,slice(None),t2.all_data.index.get_level_values(3).min())])-2016))  for (s,r) in t2.model1.index1)
        return t2.model1.NPV<=row_sum
    t2.model1.NPV_INV= Constraint(t2.model1.iteration,rule=NPV_INVENTORY)
    
    t2.model1.CVAR_MIN= Var(within=NonNegativeReals)
    def CVAR_MIN_V(model1,it):
        return t2.model1.CVAR_MIN>=t2.model1.CVaR[it]
    t2.model1.CVAR_MIN_V= Constraint(t2.model1.year,rule=CVAR_MIN_V)
    
    def outcome_rule(model1):
        if args.OPT == "NPV":
            return t2.model1.NPV-t2.model1.CVAR_MIN/1000
        else:
            return t2.model1.NPV-t2.model1.CVAR_MIN*1000000
        #return t2.model1.NPV-t2.model1.CVAR_MIN/100
    t2.model1.OBJ = Objective(rule=outcome_rule, sense=maximize)
    t2.solve()
    
    #GET DECISION DATA:
    def GET_DECISION_DATA():
        st = []
        reg = []
        vals = []
        for (s,r) in t2.model1.index1:
            st = st+[s]
            reg = reg+[r] 
            vals = vals+[t2.model1.X1[(s,r)].value]
        data = {"id":st,"branch":reg,"value":vals}
        df= pd.DataFrame(data)
        df = df.set_index(['id','branch'])
        return df
    
    df = GET_DECISION_DATA()
    NPV_CVAR = [t2.model1.NPV.value,t2.model1.CVAR_MIN.value]
    print(t2.model1.NPV.value)
    for it in range(0,SCENARIOS):
        if it == 0:
            min_MAX_NPV_solved = [sum((df.value*(t2.all_data.income.loc[slice(None),slice(None),slice(None),it]+t2.all_data.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*t2.all_data.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+t2.all_data.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*t2.all_data.PV.loc[slice(None),slice(None),2041,0]*t2.all_data.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]#[(sum((t2.all_data.income.loc[(s,r,k,it)]*t2.all_data.AREA.loc[(s,r,k,it)]* t2.model1.X1[(s,r)].value)/((1+INTEREST)**(2.5+t2.all_data.year[(s,r,k,it)]-2016))  for (s,r) in t2.model1.index1 for k in t2.model1.year) +sum((t2.all_data.PV.loc[(s,r,max(t2.all_data.year[(s,r,slice(None),it)]),it)]* t2.all_data.AREA.loc[(s,r,max(t2.all_data.year[(s,r,slice(None),it)]),it)]*t2.model1.X1[(s,r)].value)/((1+INTEREST)**(2.5+max(t2.all_data.year[(s,r,slice(None),it)])-2016))  for (s,r) in t2.model1.index1))]
            EF_min_NPV_solved  = [sum((df.value*(t2.all_data.income.loc[slice(None),slice(None),y,it]+t2.all_data.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*t2.all_data.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
        else:
            min_MAX_NPV_solved = min_MAX_NPV_solved +[sum((df.value*(t2.all_data.income.loc[slice(None),slice(None),slice(None),it]+t2.all_data.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*t2.all_data.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+t2.all_data.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*t2.all_data.PV.loc[slice(None),slice(None),2041,0]*t2.all_data.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]#[(sum((t2.all_data.income.loc[(s,r,k,it)]*t2.all_data.AREA.loc[(s,r,k,it)]* t2.model1.X1[(s,r)].value)/((1+INTEREST)**(2.5+t2.all_data.year[(s,r,k,it)]-2016))  for (s,r) in t2.model1.index1 for k in t2.model1.year) +sum((t2.all_data.PV.loc[(s,r,max(t2.all_data.year[(s,r,slice(None),it)]),it)]* t2.all_data.AREA.loc[(s,r,max(t2.all_data.year[(s,r,slice(None),it)]),it)]*t2.model1.X1[(s,r)].value)/((1+INTEREST)**(2.5+max(t2.all_data.year[(s,r,slice(None),it)])-2016))  for (s,r) in t2.model1.index1))]
            EF_min_NPV_solved  = EF_min_NPV_solved + [sum((df.value*(t2.all_data.income.loc[slice(None),slice(None),y,it]+t2.all_data.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*t2.all_data.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
    
    for it in range(0,SCENARIOS):
        if it == 0:
            min_MAX_NPV_unsolved_LL = [sum((df.value*(all_data_LL.income.loc[slice(None),slice(None),slice(None),it]+all_data_LL.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_LL.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_LL.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_LL.PV.loc[slice(None),slice(None),2041,0]*all_data_LL.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            min_MAX_NPV_unsolved_LH = [sum((df.value*(all_data_LH.income.loc[slice(None),slice(None),slice(None),it]+all_data_LH.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_LH.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_LH.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_LH.PV.loc[slice(None),slice(None),2041,0]*all_data_LH.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            min_MAX_NPV_unsolved_ML = [sum((df.value*(all_data_ML.income.loc[slice(None),slice(None),slice(None),it]+all_data_ML.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_ML.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_ML.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_ML.PV.loc[slice(None),slice(None),2041,0]*all_data_ML.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            min_MAX_NPV_unsolved_HL = [sum((df.value*(all_data_HL.income.loc[slice(None),slice(None),slice(None),it]+all_data_HL.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_HL.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_HL.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_HL.PV.loc[slice(None),slice(None),2041,0]*all_data_HL.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            min_MAX_NPV_unsolved_MM = [sum((df.value*(all_data_MM.income.loc[slice(None),slice(None),slice(None),it]+all_data_MM.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_MM.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_MM.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_MM.PV.loc[slice(None),slice(None),2041,0]*all_data_MM.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            min_MAX_PV_unsolved_LL = [sum((df.value*all_data_LL.PV.loc[slice(None),slice(None),2041,it]*all_data_LL.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            min_MAX_PV_unsolved_LH = [sum((df.value*all_data_LH.PV.loc[slice(None),slice(None),2041,it]*all_data_LH.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            min_MAX_PV_unsolved_ML = [sum((df.value*all_data_ML.PV.loc[slice(None),slice(None),2041,it]*all_data_ML.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            min_MAX_PV_unsolved_HL = [sum((df.value*all_data_HL.PV.loc[slice(None),slice(None),2041,it]*all_data_HL.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            min_MAX_PV_unsolved_MM = [sum((df.value*all_data_MM.PV.loc[slice(None),slice(None),2041,it]*all_data_MM.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            EF_min_NPV_unsolved_LL  = [sum((df.value*(all_data_LL.income.loc[slice(None),slice(None),y,it]+all_data_LL.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_LL.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_min_NPV_unsolved_LH  = [sum((df.value*(all_data_LH.income.loc[slice(None),slice(None),y,it]+all_data_LH.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_LH.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_min_NPV_unsolved_ML  = [sum((df.value*(all_data_ML.income.loc[slice(None),slice(None),y,it]+all_data_ML.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_ML.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_min_NPV_unsolved_HL  = [sum((df.value*(all_data_HL.income.loc[slice(None),slice(None),y,it]+all_data_HL.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_HL.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_min_NPV_unsolved_MM  = [sum((df.value*(all_data_MM.income.loc[slice(None),slice(None),y,it]+all_data_MM.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_MM.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
        else:
            min_MAX_NPV_unsolved_LL = min_MAX_NPV_unsolved_LL +[sum((df.value*(all_data_LL.income.loc[slice(None),slice(None),slice(None),it]+all_data_LL.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_LL.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_LL.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_LL.PV.loc[slice(None),slice(None),2041,0]*all_data_LL.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            min_MAX_NPV_unsolved_LH = min_MAX_NPV_unsolved_LH +[sum((df.value*(all_data_LH.income.loc[slice(None),slice(None),slice(None),it]+all_data_LH.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_LH.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_LH.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_LH.PV.loc[slice(None),slice(None),2041,0]*all_data_LH.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            min_MAX_NPV_unsolved_ML = min_MAX_NPV_unsolved_ML +[sum((df.value*(all_data_ML.income.loc[slice(None),slice(None),slice(None),it]+all_data_ML.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_ML.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_ML.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_ML.PV.loc[slice(None),slice(None),2041,0]*all_data_ML.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            min_MAX_NPV_unsolved_HL = min_MAX_NPV_unsolved_HL +[sum((df.value*(all_data_HL.income.loc[slice(None),slice(None),slice(None),it]+all_data_HL.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_HL.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_HL.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_HL.PV.loc[slice(None),slice(None),2041,0]*all_data_HL.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            min_MAX_NPV_unsolved_MM = min_MAX_NPV_unsolved_MM +[sum((df.value*(all_data_MM.income.loc[slice(None),slice(None),slice(None),it]+all_data_MM.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_MM.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_MM.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_MM.PV.loc[slice(None),slice(None),2041,0]*all_data_MM.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            min_MAX_PV_unsolved_LL = min_MAX_PV_unsolved_LL+[sum((df.value*all_data_LL.PV.loc[slice(None),slice(None),2041,it]*all_data_LL.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            min_MAX_PV_unsolved_LH = min_MAX_PV_unsolved_LH+[sum((df.value*all_data_LH.PV.loc[slice(None),slice(None),2041,it]*all_data_LH.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            min_MAX_PV_unsolved_ML = min_MAX_PV_unsolved_ML+[sum((df.value*all_data_ML.PV.loc[slice(None),slice(None),2041,it]*all_data_ML.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            min_MAX_PV_unsolved_HL = min_MAX_PV_unsolved_HL+[sum((df.value*all_data_HL.PV.loc[slice(None),slice(None),2041,it]*all_data_HL.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            min_MAX_PV_unsolved_MM = min_MAX_PV_unsolved_MM+[sum((df.value*all_data_MM.PV.loc[slice(None),slice(None),2041,it]*all_data_MM.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            EF_min_NPV_unsolved_LL  = EF_min_NPV_unsolved_LL+[sum((df.value*(all_data_LL.income.loc[slice(None),slice(None),y,it]+all_data_LL.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_LL.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_min_NPV_unsolved_LH  = EF_min_NPV_unsolved_LH+[sum((df.value*(all_data_LH.income.loc[slice(None),slice(None),y,it]+all_data_LH.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_LH.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_min_NPV_unsolved_ML  = EF_min_NPV_unsolved_ML+[sum((df.value*(all_data_ML.income.loc[slice(None),slice(None),y,it]+all_data_ML.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_ML.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_min_NPV_unsolved_HL  = EF_min_NPV_unsolved_HL+[sum((df.value*(all_data_HL.income.loc[slice(None),slice(None),y,it]+all_data_HL.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_HL.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_min_NPV_unsolved_MM  = EF_min_NPV_unsolved_MM+[sum((df.value*(all_data_MM.income.loc[slice(None),slice(None),y,it]+all_data_MM.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_MM.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
    
    ##MAX AVERAGE NPV
    
    #Max average NPV
    
    try:
        t2.model1.del_component(t2.model1.NPV_INV)
    except:
        print("NONE")
    
    def NPV_INVENTORY(model1):
        row_sum = sum(((t2.all_data.income.loc[(s,r,k,it)]+t2.all_data.natural_rm_wind.loc[(s,r,k,it)]*SALV_PRICE)*t2.all_data.AREA.loc[(s,r,k,it)]* t2.model1.X1[(s,r)])/((1+INTEREST)**(2.5+t2.all_data.year[(s,r,k,it)]-2016))  for (s,r) in t2.model1.index1 for k in t2.model1.year for it in t2.model1.iteration)+sum((t2.all_data.PV.loc[(s,r,max(t2.all_data.year[(s,r,slice(None),t2.all_data.index.get_level_values(3).min())]),it)]*t2.all_data.AREA.loc[(s,r,max(t2.all_data.year[(s,r,slice(None),t2.all_data.index.get_level_values(3).min())]),it)]* t2.model1.X1[(s,r)])/((1+INTEREST)**(2.5+max(t2.all_data.year[(s,r,slice(None),t2.all_data.index.get_level_values(3).min())])-2016))  for (s,r) in t2.model1.index1 for it in t2.model1.iteration)
        return t2.model1.NPV==row_sum
    t2.model1.NPV_INV= Constraint(rule=NPV_INVENTORY)
    
    try:
        t2.model1.del_component(t2.model1.OPJ)
    except:
        print("NONE")
    
    def outcome_rule(model1):
        if args.OPT == "NPV":
            rho = 0.001
            return t2.model1.NPV-t2.model1.CVAR_MIN*rho
        else:
            rho = 0.0000001
            return rho*t2.model1.NPV-t2.model1.CVAR_MIN
    t2.model1.OBJ = Objective(rule=outcome_rule, sense=maximize)
    t2.solve()
    
    #GET DECISION DATA:
    
    df = GET_DECISION_DATA()
    NPV_CVAR = NPV_CVAR+[t2.model1.NPV.value,t2.model1.CVAR_MIN.value]
    print(t2.model1.NPV.value)
    for it in range(0,SCENARIOS):
        if it == 0:
            MAX_AVG_NPV_solved = [sum((df.value*(t2.all_data.income.loc[slice(None),slice(None),slice(None),it]+t2.all_data.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*t2.all_data.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+t2.all_data.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*t2.all_data.PV.loc[slice(None),slice(None),2041,0]*t2.all_data.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            EF_AVG_NPV_solved  = [sum((df.value*(t2.all_data.income.loc[slice(None),slice(None),y,it]+t2.all_data.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*t2.all_data.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            YEAR_IT = [str(int(y))+"_"+str(it) for y in t2.model1.year]
            
        else:
            MAX_AVG_NPV_solved = MAX_AVG_NPV_solved +[sum((df.value*(t2.all_data.income.loc[slice(None),slice(None),slice(None),it]+t2.all_data.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*t2.all_data.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+t2.all_data.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*t2.all_data.PV.loc[slice(None),slice(None),2041,0]*t2.all_data.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]#[(sum((t2.all_data.income.loc[(s,r,k,it)]*t2.all_data.AREA.loc[(s,r,k,it)]* t2.model1.X1[(s,r)].value)/((1+INTEREST)**(2.5+t2.all_data.year[(s,r,k,it)]-2016))  for (s,r) in t2.model1.index1 for k in t2.model1.year) +sum((t2.all_data.PV.loc[(s,r,max(t2.all_data.year[(s,r,slice(None),it)]),it)]* t2.all_data.AREA.loc[(s,r,max(t2.all_data.year[(s,r,slice(None),it)]),it)]*t2.model1.X1[(s,r)].value)/((1+INTEREST)**(2.5+max(t2.all_data.year[(s,r,slice(None),it)])-2016))  for (s,r) in t2.model1.index1))]
            EF_AVG_NPV_solved  = EF_AVG_NPV_solved + [sum((df.value*(t2.all_data.income.loc[slice(None),slice(None),y,it]+t2.all_data.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*t2.all_data.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            YEAR_IT = YEAR_IT + [str(int(y))+"_"+str(it) for y in t2.model1.year]
            
    for it in range(0,SCENARIOS):
        if it == 0:
            MAX_AVG_NPV_unsolved_LL = [sum((df.value*(all_data_LL.income.loc[slice(None),slice(None),slice(None),it]+all_data_LL.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_LL.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_LL.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_LL.PV.loc[slice(None),slice(None),2041,0]*all_data_LL.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            MAX_AVG_NPV_unsolved_LH = [sum((df.value*(all_data_LH.income.loc[slice(None),slice(None),slice(None),it]+all_data_LH.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_LH.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_LH.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_LH.PV.loc[slice(None),slice(None),2041,0]*all_data_LH.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            MAX_AVG_NPV_unsolved_ML = [sum((df.value*(all_data_ML.income.loc[slice(None),slice(None),slice(None),it]+all_data_ML.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_ML.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_ML.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_ML.PV.loc[slice(None),slice(None),2041,0]*all_data_ML.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            MAX_AVG_NPV_unsolved_HL = [sum((df.value*(all_data_HL.income.loc[slice(None),slice(None),slice(None),it]+all_data_HL.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_HL.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_HL.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_HL.PV.loc[slice(None),slice(None),2041,0]*all_data_HL.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            MAX_AVG_NPV_unsolved_MM = [sum((df.value*(all_data_MM.income.loc[slice(None),slice(None),slice(None),it]+all_data_MM.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_MM.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_MM.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_MM.PV.loc[slice(None),slice(None),2041,0]*all_data_MM.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            MAX_AVG_PV_unsolved_LL = [sum((df.value*all_data_LL.PV.loc[slice(None),slice(None),2041,it]*all_data_LL.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            MAX_AVG_PV_unsolved_LH = [sum((df.value*all_data_LH.PV.loc[slice(None),slice(None),2041,it]*all_data_LH.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            MAX_AVG_PV_unsolved_ML = [sum((df.value*all_data_ML.PV.loc[slice(None),slice(None),2041,it]*all_data_ML.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            MAX_AVG_PV_unsolved_HL = [sum((df.value*all_data_HL.PV.loc[slice(None),slice(None),2041,it]*all_data_HL.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            MAX_AVG_PV_unsolved_MM = [sum((df.value*all_data_MM.PV.loc[slice(None),slice(None),2041,it]*all_data_MM.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            EF_AVG_NPV_unsolved_LL  = [sum((df.value*(all_data_LL.income.loc[slice(None),slice(None),y,it]+all_data_LL.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_LL.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_AVG_NPV_unsolved_LH  = [sum((df.value*(all_data_LH.income.loc[slice(None),slice(None),y,it]+all_data_LH.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_LH.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_AVG_NPV_unsolved_ML  = [sum((df.value*(all_data_ML.income.loc[slice(None),slice(None),y,it]+all_data_ML.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_ML.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_AVG_NPV_unsolved_HL  = [sum((df.value*(all_data_HL.income.loc[slice(None),slice(None),y,it]+all_data_HL.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_HL.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_AVG_NPV_unsolved_MM  = [sum((df.value*(all_data_MM.income.loc[slice(None),slice(None),y,it]+all_data_MM.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_MM.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
        else:
            MAX_AVG_NPV_unsolved_LL = MAX_AVG_NPV_unsolved_LL+[sum((df.value*(all_data_LL.income.loc[slice(None),slice(None),slice(None),it]+all_data_LL.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_LL.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_LL.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_LL.PV.loc[slice(None),slice(None),2041,0]*all_data_LL.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            MAX_AVG_NPV_unsolved_LH = MAX_AVG_NPV_unsolved_LH+[sum((df.value*(all_data_LH.income.loc[slice(None),slice(None),slice(None),it]+all_data_LH.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_LH.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_LH.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_LH.PV.loc[slice(None),slice(None),2041,0]*all_data_LH.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            MAX_AVG_NPV_unsolved_ML = MAX_AVG_NPV_unsolved_ML+[sum((df.value*(all_data_ML.income.loc[slice(None),slice(None),slice(None),it]+all_data_ML.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_ML.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_ML.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_ML.PV.loc[slice(None),slice(None),2041,0]*all_data_ML.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            MAX_AVG_NPV_unsolved_HL = MAX_AVG_NPV_unsolved_HL+[sum((df.value*(all_data_HL.income.loc[slice(None),slice(None),slice(None),it]+all_data_HL.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_HL.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_HL.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_HL.PV.loc[slice(None),slice(None),2041,0]*all_data_HL.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            MAX_AVG_NPV_unsolved_MM = MAX_AVG_NPV_unsolved_MM+[sum((df.value*(all_data_MM.income.loc[slice(None),slice(None),slice(None),it]+all_data_MM.natural_rm_wind.loc[slice(None),slice(None),slice(None),it]*SALV_PRICE)*all_data_MM.AREA.loc[slice(None),slice(None),slice(None),it])/((1+INTEREST)**(2.5+all_data_MM.year.loc[slice(None),slice(None),slice(None),it]-2016)))+sum((df.value*all_data_MM.PV.loc[slice(None),slice(None),2041,0]*all_data_MM.AREA.loc[slice(None),slice(None),2041,0])/((1+INTEREST)**(30)))]
            MAX_AVG_PV_unsolved_LL = MAX_AVG_PV_unsolved_LL+[sum((df.value*all_data_LL.PV.loc[slice(None),slice(None),2041,it]*all_data_LL.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            MAX_AVG_PV_unsolved_LH = MAX_AVG_PV_unsolved_LH+[sum((df.value*all_data_LH.PV.loc[slice(None),slice(None),2041,it]*all_data_LH.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            MAX_AVG_PV_unsolved_ML = MAX_AVG_PV_unsolved_ML+[sum((df.value*all_data_ML.PV.loc[slice(None),slice(None),2041,it]*all_data_ML.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            MAX_AVG_PV_unsolved_HL = MAX_AVG_PV_unsolved_HL+[sum((df.value*all_data_HL.PV.loc[slice(None),slice(None),2041,it]*all_data_HL.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            MAX_AVG_PV_unsolved_MM = MAX_AVG_PV_unsolved_MM+[sum((df.value*all_data_MM.PV.loc[slice(None),slice(None),2041,it]*all_data_MM.AREA.loc[slice(None),slice(None),2041,it])/((1+INTEREST)**(30)))]
            EF_AVG_NPV_unsolved_LL  = EF_AVG_NPV_unsolved_LL+[sum((df.value*(all_data_LL.income.loc[slice(None),slice(None),y,it]+all_data_LL.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_LL.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_AVG_NPV_unsolved_LH  = EF_AVG_NPV_unsolved_LH+[sum((df.value*(all_data_LH.income.loc[slice(None),slice(None),y,it]+all_data_LH.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_LH.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_AVG_NPV_unsolved_ML  = EF_AVG_NPV_unsolved_ML+[sum((df.value*(all_data_ML.income.loc[slice(None),slice(None),y,it]+all_data_ML.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_ML.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_AVG_NPV_unsolved_HL  = EF_AVG_NPV_unsolved_HL+[sum((df.value*(all_data_HL.income.loc[slice(None),slice(None),y,it]+all_data_HL.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_HL.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
            EF_AVG_NPV_unsolved_MM  = EF_AVG_NPV_unsolved_MM+[sum((df.value*(all_data_MM.income.loc[slice(None),slice(None),y,it]+all_data_MM.natural_rm_wind.loc[slice(None),slice(None),y,it]*SALV_PRICE)*all_data_MM.AREA.loc[slice(None),slice(None),y,it])) for y in t2.model1.year]
    
    DATA = [MAX_AVG_NPV_solved,min_MAX_NPV_solved]
    DATA_unsolved = [min_MAX_NPV_unsolved_LL,MAX_AVG_NPV_unsolved_LL,min_MAX_NPV_unsolved_HL,MAX_AVG_NPV_unsolved_HL,min_MAX_NPV_unsolved_ML,MAX_AVG_NPV_unsolved_ML,min_MAX_NPV_unsolved_LH,MAX_AVG_NPV_unsolved_LH,min_MAX_NPV_unsolved_MM,MAX_AVG_NPV_unsolved_MM,MAX_AVG_PV_unsolved_LL,MAX_AVG_PV_unsolved_HL,MAX_AVG_PV_unsolved_LH,MAX_AVG_PV_unsolved_ML,MAX_AVG_PV_unsolved_MM,min_MAX_PV_unsolved_LL,min_MAX_PV_unsolved_HL,min_MAX_PV_unsolved_LH,min_MAX_PV_unsolved_ML,min_MAX_PV_unsolved_MM]
    DAT_FRAME = pd.DataFrame(DATA)
    DAT_FRAME_unsolved = pd.DataFrame(DATA_unsolved)
    DAT_FRAME = DAT_FRAME.T
    DAT_FRAME_unsolved=DAT_FRAME_unsolved.T
    DAT_FRAME.columns = ["MAX_AVG_NPV_solved","min_MAX_NPV_solved"]
    DAT_FRAME_unsolved.columns = ["min_MAX_NPV_unsolved_LL","MAX_AVG_NPV_unsolved_LL","min_MAX_NPV_unsolved_HL","MAX_AVG_NPV_unsolved_HL","min_MAX_NPV_unsolved_ML","MAX_AVG_NPV_unsolved_ML","min_MAX_NPV_unsolved_LH","MAX_AVG_NPV_unsolved_LH","min_MAX_NPV_unsolved_MM","MAX_AVG_NPV_unsolved_MM","MAX_AVG_PV_unsolved_LL","MAX_AVG_PV_unsolved_HL","MAX_AVG_PV_unsolved_LH","MAX_AVG_PV_unsolved_ML","MAX_AVG_PV_unsolved_MM","min_MAX_PV_unsolved_LL","min_MAX_PV_unsolved_HL","min_MAX_PV_unsolved_LH","min_MAX_PV_unsolved_ML","min_MAX_PV_unsolved_MM"]
    DAT_FRAME.to_csv(Results_folder+"SP_RESULTS_solved_EF_"+VV[1]+SALVAGE+OPT+".csv")
    DAT_FRAME_unsolved.to_csv(Results_folder+"SP_RESULTS_unsolved_EF_"+VV[1]+SALVAGE+OPT+".csv")
    
    DATA_EF = [YEAR_IT,EF_AVG_NPV_solved,EF_min_NPV_solved]
    DATA_EF_FRAME = pd.DataFrame(DATA_EF).T
    DATA_EF_FRAME.columns = ["EF_YEAR","Solved_Avg","Solved_Min"]
    DATA_EF_FRAME.to_csv(Results_folder+"SP_RESULTS_solved_EF_INCOME_"+VV[1]+SALVAGE+OPT+".csv")
    
    DATA_EF_UN = [YEAR_IT,EF_AVG_NPV_unsolved_LL,EF_AVG_NPV_unsolved_LH,EF_AVG_NPV_unsolved_ML,EF_AVG_NPV_unsolved_HL,EF_AVG_NPV_unsolved_MM,EF_min_NPV_unsolved_LL,EF_min_NPV_unsolved_LH,EF_min_NPV_unsolved_ML,EF_min_NPV_unsolved_HL,EF_min_NPV_unsolved_MM]
    DATA_EF_UN_FRAME = pd.DataFrame(DATA_EF_UN).T
    DATA_EF_UN_FRAME.columns = ["EF_YEAR","EF_AVG_NPV_unsolved_LL","EF_AVG_NPV_unsolved_LH","EF_AVG_NPV_unsolved_ML","EF_AVG_NPV_unsolved_HL","EF_AVG_NPV_unsolved_MM","EF_min_NPV_unsolved_LL","EF_min_NPV_unsolved_LH","EF_min_NPV_unsolved_ML","EF_min_NPV_unsolved_HL","EF_min_NPV_unsolved_MM"]
    DATA_EF_UN_FRAME.to_csv(Results_folder+"SP_RESULTS_unsolved_EF_INCOME_"+VV[1]+SALVAGE+OPT+".csv")
    NC = pd.DataFrame(NPV_CVAR,columns=["Objectives"])
    NC.to_csv(Results_folder+"OBJ_OUTPUT_"+VV[1]+SALVAGE+OPT+".csv")