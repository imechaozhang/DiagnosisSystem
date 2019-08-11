import pandas as pd
import numpy as np

WM = pd.read_csv('../../WeightMatrix/Dis_Sym_30.csv', index_col=0)

dis2sym = pd.read_csv('../../UMLS/dis_symptom.csv', header=None)

dis2sym.fillna(method='ffill',inplace=True)

umls_dis = {}
umls_sym = {}
dis_num = {}
for i in dis2sym.index:
    temp = dis2sym.loc[i][0]
    items = temp.split('^')
    item = items[0].strip('UMLS:').split('_')
    if len(item) != 2: continue
    umls_dis[item[0]] = item[1]
    dis_num[item[0]] = int(dis2sym.loc[i][1])
for i in dis2sym.index:
    temp = dis2sym.loc[i][2]
    items = temp.split('^')
    item = items[0].strip('UMLS:').split('_')
    if len(item) != 2: continue
    umls_sym[item[0]] = item[1]
    
rev_sym = {v: k for k, v in umls_sym.items()}
rev_dis = {v: k for k, v in umls_dis.items()}

patients = {}

for i in WM.index:
    patients[i] = []
    for j in WM.columns:
        if WM.loc[i][j] != 0:
            patients[i].append(j)

def SelectedMatrix(sym):
    selected = WM[WM[sym] != 0]
    selected = selected.drop(columns=[sym])
    for c in selected.columns:
        if sum(selected[c]) == 0:
            selected.drop(columns=[c],inplace=True)
    return selected
   

T = 0.03
def renorm(dia):
    #for c in dia.index:
    #    dia[c] *= dis_num[c]**(1/3)
    dia.sort_values(ascending=False, inplace=True)
    dia.reset_index(drop=True)
    s = sum([np.exp(ai/T) for ai in dia])
    return np.exp(dia/T)/s

def diagnosis(dis, sym):
        
    selected = SelectedMatrix(sym)
    
    #The response vector
    res = pd.Series(index=WM.columns, data=[0]*len(WM.columns))
    res[sym] = 1
    
    #Diagnosis process
    output = [0, 0, 0, 0] # result: accuracy, number of questions asked, confidance, top 5 confidence
    while True:
        dia = WM.dot(res)
        if len(selected) == 1:
            #dia[selected.index[0]] = 1
            #dia.sort_values(ascending=False, inplace=True)
            dia = renorm(dia)
            output[2] = dia.iloc[0]
            output[3] = sum(dia.iloc[:5])
            if dia.keys()[0] == dis:
                output[0] = 1
                return output
            else:
                output[0] = -1
                return output
        elif len(selected.columns) == 1:
            output[0] = 0
            return output
        
        output[1] += 1
        #choose the most relevant symptom to ask: The symptom that are least shared with other diseases
        next_i = selected.columns[0]
        s = 100       
        
        for i in selected.columns:
            if min(selected[i]) != 0 or max(selected[i])==0:
                selected.drop(columns=[i], inplace = True)
        
        for i in selected.columns:   
            if 0 in selected[i].value_counts():
                pri = abs(selected[i].value_counts()[0] - len(selected)/2)
                if pri < s:
                    s = pri
                    next_i = i      
            else:
                res[next_i] = 1
                selected = selected[selected[next_i]!=0]  
        
        if next_i in patients[dis]:
            res[next_i] = 1
            selected = selected[selected[next_i]!=0]
        else:
            res[next_i] = 0
            selected = selected[selected[next_i]==0]
            
        selected.drop(columns=[next_i], inplace = True)
  

perf = pd.DataFrame(columns = ['result' ,'number of question', 'confidence', 'T5 confidence'])
N = 0
for d, v in patients.items():
    for s in v:
        N += 1
        perf.loc[N] = diagnosis(d,s)
        #if N%100 == 0: print(N)
perf.to_csv('testlog.csv')
