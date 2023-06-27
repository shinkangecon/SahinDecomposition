# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:40:55 2015

@author: bart
"""

import classes.BHData as bhd
import classes.BHCharts as bhc
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import copy as cp
import os

# ========================================================================================
# Turn the following dictionary items to TRUE/FALSE to generate different parts of output
# ----------------------------------------------------------------------------------------
settings = { 'Gender' : 'Total' ,
             'Load' : {'Fresh': False, 'vintage' : 'V20181227' } ,
             'Closeplots' : True , 
             'Area chart' : {'Make' : True , 'Web' : False } ,
             'Line chart' : {'Make' : False } ,
             'Decomposition chart' : {'Make' : False , 'Web' : False },
             'Decomposition2 chart' : {'Make' : False , 'Web' : False },
             'Decomposition3 chart' : {'Make' : True , 'Web' : False , "Animate" : False}}

# Fred Mnemonics for [ 'E' , 'U' , 'N' , 'UE' , 'UN' , 'EU' , 'EN' , 'NU' , 'NE' , 'OE' , 'ON' , 'OU' , 'EO' , 'NO' , 'UO' ] by gender
seriesplotted = { "Total" : [ 'CE16OV' , 'UNEMPLOY' , 'LNS15000000' , 'LNS17100000' , 'LNS17900000' , 'LNS17400000' , 'LNS17800000' , 'LNS17600000' , 'LNS17200000' , 'LNS17300000' , 'LNS18100000' , 'LNS17700000' , 'LNS18200000' , 'LNS18400000' , 'LNS18300000'  ] ,
                  "Men"   : [ 'LNS12000001' , 'LNS13000001' , 'LNS15000001' , 'LNS17100001' , 'LNS17900001' , 'LNS17400001' , 'LNS17800001' , 'LNS17600001' , 'LNS17200001' , 'LNS17300001' , 'LNS18100001' , 'LNS17700001' , 'LNS18200001' , 'LNS18400001' , 'LNS18300001'  ] ,
                  "Women" : [ 'LNS12000002' , 'LNS13000002' , 'LNS15000002' , 'LNS17100002' , 'LNS17900002' , 'LNS17400002' , 'LNS17800002' , 'LNS17600002' , 'LNS17200002' , 'LNS17300002' , 'LNS18100002' , 'LNS17700002' , 'LNS18200002' , 'LNS18400002' , 'LNS18300002'  ] }

areachrty = { "Total" : [57.5,67.5] ,
               "Men" : [50,80] ,
               "Women" : [50,65] }

debug0 = 0
debug1 = 1

# ==============================================================================================
# AUXILIARY PROCEDURES USED FOR THE CALCULATIONS
# ==============================================================================================

# ==============================================================================================
# MARGIN ADJUSTMENT
# Apply margin adjustment from EHS (2015) to make the flows and the stocks consistent with the
# flow dynamics identity used in the paper.
# ----------------------------------------------------------------------------------------------
def marginadjustment(thisdate):
    
    prevmonth = thisdate.month-1
    if prevmonth == 0:
        prevmonth = 12
        prevyear = thisdate.year-1
    else:
        prevyear = thisdate.year
    
    prevmonth = dt.datetime(prevyear,prevmonth,1)
    
    thisdata = results.data.loc[thisdate]
    prevdata = results.data.loc[prevmonth]
    
    pUE = thisdata['pUE']
    pUN = thisdata['pUN']
    pNE = thisdata['pNE']
    pNU = thisdata['pNU']
    pEU = thisdata['pEU']
    pEN = thisdata['pEN']
    
    E = thisdata['E']/thisdata['POP']
    N = thisdata['N']/thisdata['POP']
    U = thisdata['U']/thisdata['POP']
    E_1 = prevdata['E']/prevdata['POP']
    N_1 = prevdata['N']/prevdata['POP']
    U_1 = prevdata['U']/prevdata['POP']
    
    # Change in the state vector
    dst = np.zeros([2])
    dst[0] = E - E_1
    dst[1] = U - U_1
    
    # Constructing the weighting matrix
    W_t = np.zeros([6,6])
    W_t[0,0] = pEU*(1-pEU)/E_1
    W_t[0,1] = -pEU*pEN/E_1
    W_t[1,0] = W_t[0,1]
    W_t[1,1] = pEN*(1-pEN)/E_1
    W_t[2,2] = pUE*(1-pUE)/U_1
    W_t[2,3] = -pUE*pUN/U_1
    W_t[3,2] = W_t[2,3]
    W_t[3,3] = pUN*(1-pUN)/U_1
    W_t[4,4] = pNE*(1-pNE)/N_1
    W_t[4,5] = -pNE*pNU/N_1
    W_t[5,4] = W_t[4,5]
    W_t[5,5] = pNU*(1-pNU)/N_1
    W_t = np.linalg.inv(W_t)
    
    # Constructing the X matrix for the margin adjustment WRLS solution
    X_t = np.zeros([2,6])
    X_t[0,0] = -1*E_1
    X_t[0,1] = -1*E_1
    X_t[0,2] = U_1
    X_t[0,4] = N_1
    X_t[1,0] = E_1
    X_t[1,2] = -1*U_1
    X_t[1,3] = -1*U_1
    X_t[1,5] = N_1
    
    phat = np.matrix([pEU,pEN,pUE,pUN,pNE,pNU]).transpose()
    
    Wphat = np.matmul(W_t,phat)

    Xtop = np.concatenate((W_t,X_t.transpose()),axis=1)
    Xbot = np.concatenate((X_t,np.zeros([2,2])),axis=1)    
    XX = np.concatenate((Xtop,Xbot),axis=0)
    y = np.concatenate((Wphat,np.matrix(dst).transpose()),axis=0)
    b = np.linalg.solve(XX,y)
    
    results.data.loc[thisdate,'pEUm'] = b[0]
    results.data.loc[thisdate,'pENm'] = b[1]
    results.data.loc[thisdate,'pUEm'] = b[2]
    results.data.loc[thisdate,'pUNm'] = b[3]
    results.data.loc[thisdate,'pNEm'] = b[4]
    results.data.loc[thisdate,'pNUm'] = b[5]
    
    return b

# ==============================================================================================
# GET TRANSITION MATRIX AND D
# ----------------------------------------------------------------------------------------------
def getPdandSS(thisdate):
    
    prevmonth = thisdate.month-1
    if prevmonth == 0:
        prevmonth = 12
        prevyear = thisdate.year-1
    else:
        prevyear = thisdate.year
    
    prevmonth = dt.datetime(prevyear,prevmonth,1)
    
    thisdata = results.data.loc[thisdate]
    prevdata = results.data.loc[prevmonth]
    
    pUE = thisdata['pUEm']
    pUN = thisdata['pUNm']
    pNE = thisdata['pNEm']
    pNU = thisdata['pNUm']
    pEU = thisdata['pEUm']
    pEN = thisdata['pENm']
    
    E = thisdata['E']
    N = thisdata['N']
    U = thisdata['U']
    E_1 = prevdata['E']
    N_1 = prevdata['N']
    U_1 = prevdata['U']
    
    d = np.matrix([[pNE],[pNU]])
    
    P = np.matrix(np.zeros([2,2]))    
    P[0,0] = -pEU-pEN-pNE
    P[0,1] = pUE-pNE
    P[1,0] = pEU-pNU
    P[1,1] = -pUE-pUN-pNU
    
    dd = {}
    dd['pUE'] = np.matrix(np.zeros([2,1]))
    dd['pUN'] = np.matrix(np.zeros([2,1]))
    dd['pNE'] = np.matrix([[1],[0]])
    dd['pNU'] = np.matrix([[0],[1]])
    dd['pEU'] = np.matrix(np.zeros([2,1]))
    dd['pEN'] = np.matrix(np.zeros([2,1]))  
    
    #BH: I am here 11/05/18@10am
    dP = {}
    dP['pUE'] = np.matrix([[ 0 , 1 ],[ 0 , -1 ]])
    dP['pUN'] = np.matrix([[ 0 , 0 ],[ 0 , -1 ]])
    dP['pNE'] = np.matrix([[ -1 , -1 ],[ 0 , 0 ]])
    dP['pNU'] = np.matrix([[ 0 , 0 ],[ -1 , -1 ]])
    dP['pEU'] = np.matrix([[ -1 , 0 ],[ 1 , 0 ]])
    dP['pEN'] = np.matrix([[ -1 , 0 ],[ 0 , 0 ]])  
    
    PdSS = {}
    PdSS['d'] = d
    PdSS['P'] = P
    PdSS['dd'] = dd
    PdSS['dP'] = dP
    
    SS = -1*np.matmul(np.linalg.inv(P),d)
    
    PdSS['SS'] = {}
    PdSS['SS']['EPOP'] = SS[0,0]
    PdSS['SS']['UPOP'] = SS[1,0]
    PdSS['SS']['LFPR'] = SS[0,0]+SS[1,0]
    
    return PdSS

# ==============================================================================================
# FLOW DECOMPOSITION
# Calculate the matrices for the decomposition for a particular period and saving them in the
# results object    
# ----------------------------------------------------------------------------------------------
def flowdecomposition(thisdate):
    
    prevmonth = thisdate.month-1
    if prevmonth == 0:
        prevmonth = 12
        prevyear = thisdate.year-1
    else:
        prevyear = thisdate.year
    
    prevmonth = dt.datetime(prevyear,prevmonth,1)
    
    thisdata = results.data.loc[thisdate]
    prevdata = results.data.loc[prevmonth]
    
    thisPdSS = getPdandSS(thisdate)
    prevPdSS = getPdandSS(prevmonth)
    
    d = thisPdSS['d']
    d_1 = prevPdSS['d']
    
    P = thisPdSS['P']
    P_1 = prevPdSS['P']
    
    SS = np.matrix([[thisPdSS['SS']['EPOP']],[thisPdSS['SS']['UPOP']]])
    SS_1 = np.matrix([[prevPdSS['SS']['EPOP']],[prevPdSS['SS']['UPOP']]])
    SSsum = SS+SS_1
    
    dS = np.matrix([[thisdata['dEPOP']],[thisdata['dUPOP']]])
    dS_1 = np.matrix([[prevdata['dEPOP']],[prevdata['dUPOP']]])
    
    M = np.matmul(np.matmul(P,(np.eye(2)+P_1)),np.linalg.inv(P_1))
    
    results.data.loc[thisdate,'dEPOPdEPOP_1'] = M[0,0]
    results.data.loc[thisdate,'dEPOPdUPOP_1'] = M[0,1]
    results.data.loc[thisdate,'dUPOPdEPOP_1'] = M[1,0]
    results.data.loc[thisdate,'dUPOPdUPOP_1'] = M[1,1]
    
    results.data.loc[thisdate,'dEPOPdEPOP_1mult'] = M[0,0]*prevdata['dEPOP']
    results.data.loc[thisdate,'dEPOPdUPOP_1mult'] = M[0,1]*prevdata['dUPOP']
    results.data.loc[thisdate,'dUPOPdEPOP_1mult'] = M[1,0]*prevdata['dEPOP']
    results.data.loc[thisdate,'dUPOPdUPOP_1mult'] = M[1,1]*prevdata['dUPOP']
    
    S_1 = np.matrix([[prevdata['EPOP']],[prevdata['UPOP']]])
    S = np.matrix([[thisdata['EPOP']],[thisdata['UPOP']]])
    dSCheck = d + np.matmul(P,S_1)
    results.data.loc[thisdate,'dEPOPcheck'] = dSCheck[0,0]
    results.data.loc[thisdate,'dUPOPcheck'] = dSCheck[1,0]
    
    dd = -0.5*np.matmul(P-P_1,SS+SS_1) - 0.5*np.matmul(P+P_1,SS-SS_1)
    dddev = np.sqrt(np.power(dd-(d-d_1),2).sum())
    results.data.loc[thisdate,'ddcheck'] = dddev
    
    dS_ = np.matmul(M,dS_1)-np.matmul(P,(SS-SS_1))
    dSdev = np.sqrt(np.power(dS_- dS,2).sum())
    results.data.loc[thisdate,'dScheck'] = dSdev
    
    ps = ['pUE','pUN','pEU','pEN','pNE','pNU']
    
    M = np.matmul(P,np.linalg.inv(P+P_1))
    for thisp in ps:
        M2 = 2*thisPdSS['dd'][thisp]+np.matmul(thisPdSS['dP'][thisp],SSsum)
        M2 = M2*(thisdata[thisp+'m']-prevdata[thisp+'m'])
        M2 = np.matmul(M,M2)
        
        results.data.loc[thisdate,'dEPOPd'+thisp] = M2[0,0]
        results.data.loc[thisdate,'dUPOPd'+thisp] = M2[1,0]

# ========================================================================================
# Loading the data from Fred or from the saved vintage
# ----------------------------------------------------------------------------------------

start = dt.datetime(1990, 1, 1)
end = dt.datetime(dt.datetime.now().year, 12, 31)
rawdata = bhd.TSData(start,end)
rawdata.name = settings['Gender']+'_'+settings['Load']['vintage']+'_rawdata_'
rawdata.path = 'xlsx/'
rawdata.info = 'Flow decomposition of the labor force participation rate (EHS 2018)'

if settings['Load']['Fresh'] == True:
    rawdata.interpolate = True
    rawdata.seriesplotted = seriesplotted[settings['Gender']] 
    rawdata.constructdata()
    rawdata.data.columns = [ 'E' , 'U' , 'N' , 'UE' , 'UN' , 'EU' , 'EN' , 'NU' , 'NE' , 'OE' , 'ON' , 'OU' , 'EO' , 'NO' , 'UO' ]
    rawdata.toExcel()
else:
    rawdata.fromExcel()

del rawdata.wb

results = cp.deepcopy(rawdata)
results.name = results.name+'_results'

# Calculating the stocks in terms of percentages of the population
results.data['POP'] = ( results.data['E'] + results.data['U'] + results.data['N'] )
results.data['LF'] = ( results.data['E'] + results.data['U'] )
results.data['UPOP'] = results.data['U']/results.data['POP']
results.data['EPOP'] = results.data['E']/results.data['POP']
results.data['LFPR'] = results.data['LF']/results.data['POP']
results.data['URATE'] = results.data['U']/results.data['LF']

# Calculating the changes in the stocks that we decompose
results.data['dUPOP'] = results.data['UPOP']-results.data['UPOP'].shift(1)
results.data['dEPOP'] = results.data['EPOP']-results.data['EPOP'].shift(1)
results.data['dLFPR'] = results.data['LFPR']-results.data['LFPR'].shift(1)

# Calculating the transition probabilities from the BLS labor force flows
results.data['pUE'] = results.data['UE']/results.data['U'].shift(1)
results.data['pUN'] = results.data['UN']/results.data['U'].shift(1)
results.data['pNE'] = results.data['NE']/results.data['N'].shift(1)
results.data['pNU'] = results.data['NU']/results.data['N'].shift(1)
results.data['pEU'] = results.data['EU']/results.data['E'].shift(1)
results.data['pEN'] = results.data['EN']/results.data['E'].shift(1)

# ==============================================================================================
# Adding columns with missings to results that we will fill with results we calculate later
# ==============================================================================================
# Making the columns for the margin-adjusted transition probabilities
pMs = ['pUEm','pUNm','pEUm','pENm','pNUm','pNEm']
for thisp in pMs:
    results.data[thisp] = np.nan
    
# Making the columns for the steady state EPOP, UPOP, and LFPR
SSs = ['EPOPSS','UPOPSS','LFPRSS']    
for thisss in SSs:
    results.data[thisss] = np.nan

# Making the period-by-period columns for the decomposition
decomp = ['dEPOPdEPOP_1','dEPOPdUPOP_1','dUPOPdEPOP_1','dUPOPdUPOP_1','dEPOPdEPOP_1mult','dEPOPdUPOP_1mult','dUPOPdEPOP_1mult','dUPOPdUPOP_1mult','dEPOPdpUE','dUPOPdpUE','dEPOPdpUN','dUPOPdpUN',
          'dEPOPdpEU','dUPOPdpEU','dEPOPdpEN','dUPOPdpEN','dEPOPdpNU','dUPOPdpNU','dEPOPdpNE','dUPOPdpNE','dEPOPcheck','dUPOPcheck', 'ddcheck' , 'dScheck' ]
for thisd in decomp:
    results.data[thisd] = np.nan
    
cumdecomp = ['cumdEPOPdEPOPinit','cumdUPOPdEPOPinit','cumdEPOPdUPOPinit','cumdUPOPdUPOPinit','cumdEPOPdpUE','cumdUPOPdpUE','cumdEPOPdpUN','cumdUPOPdpUN',
             'cumdEPOPdpEU','cumdUPOPdpEU','cumdEPOPdpEN','cumdUPOPdpEN','cumdEPOPdpNU','cumdUPOPdpNU','cumdEPOPdpNE','cumdUPOPdpNE',
             'cumdLFPRdpUE','cumdLFPRdpUN','cumdLFPRdpEU','cumdLFPRdpEN','cumdLFPRdpNU','cumdLFPRdpNE']
for thisc in cumdecomp:
    results.data[thisc] = np.nan

# ==============================================================================================
# MARGIN ADJUSTMENT OF FLOW PROBABILITIES
# Looping over all dates to do margin adjustment    
# ----------------------------------------------------------------------------------------------
for thisdate_ in results.data.index:
    
    if (thisdate_ >= dt.datetime(1990,2,1)) and (thisdate_ <= results.data.last_valid_index()):
        
        pms = marginadjustment(thisdate_)  
                
# ==============================================================================================
# FLOW STEADY STATE
# Calculating the flow steady state for all months
# ==============================================================================================
for thisdate_ in results.data.index:
    
    if (thisdate_ >= dt.datetime(1990,2,1)) and (thisdate_ <= results.data.last_valid_index()):
        
        PdSS = getPdandSS(thisdate_)
        results.data.loc[thisdate_,'EPOPSS'] = PdSS['SS']['EPOP']
        results.data.loc[thisdate_,'UPOPSS'] = PdSS['SS']['UPOP']
        results.data.loc[thisdate_,'LFPRSS'] = PdSS['SS']['LFPR']              

# ==============================================================================================
# FLOW DECOMPOSITION OF THE LABOR FORCE PARTICIPATION RATE
# Calculate the month-by-month terms of the decomposition
# ----------------------------------------------------------------------------------------------
for thisdate_ in results.data.index:
    
    if (thisdate_ >= dt.datetime(1990,3,1)) and (thisdate_ <= results.data.last_valid_index()):
        
        flowdecomposition(thisdate_)
        
results.data['checkdEPOP'] = results.data[['dEPOPdEPOP_1mult','dEPOPdUPOP_1mult','dEPOPdpUE','dEPOPdpUN','dEPOPdpEN','dEPOPdpEU','dEPOPdpNE','dEPOPdpNU']].sum(axis=1)
results.data['checkdUPOP'] = results.data[['dUPOPdEPOP_1mult','dUPOPdUPOP_1mult','dUPOPdpUE','dUPOPdpUN','dUPOPdpEN','dUPOPdpEU','dUPOPdpNE','dUPOPdpNU']].sum(axis=1)

# ==============================================================================================
# CALCULATING CUMULATIVE CONTRIBUTIONS TO THE CHANGES IN THE STOCKS
# ----------------------------------------------------------------------------------------------
ps_ = ['pUE','pUN','pEU','pEN','pNU','pNE']

for thisdate_ in results.data.index:
    
    prevmonth_ = thisdate_.month-1
    if prevmonth_ == 0:
        prevmonth_ = 12
        prevyear_ = thisdate_.year-1
    else:
        prevyear_ = thisdate_.year
    
    prevmonth_ = dt.datetime(prevyear_,prevmonth_,1)

    if (thisdate_ == dt.datetime(1990,3,1)):
        
        for dstock in ['EPOP','UPOP']:
            
            for dstock2 in ['EPOP','UPOP']:
            
                results.data.loc[thisdate_,'cumd'+dstock+'d'+dstock2+'init'] = results.data.loc[thisdate_,'d'+dstock+'d'+dstock2+'_1mult']                            
                
            for dp in ps_:

                results.data.loc[thisdate_,'cumd'+dstock+'d'+dp] = results.data.loc[thisdate_,'d'+dstock+'d'+dp]                        
    
    if (thisdate_ > dt.datetime(1990,3,1)) and (thisdate_ <= results.data.last_valid_index()):
        
        results.data.loc[thisdate_,'cumdEPOPdEPOPinit'] = \
            results.data.loc[thisdate_,'dEPOPdEPOP_1']*results.data.loc[prevmonth_,'cumdEPOPdEPOPinit'] +\
            results.data.loc[thisdate_,'dEPOPdUPOP_1']*results.data.loc[prevmonth_,'cumdUPOPdEPOPinit']
        
        results.data.loc[thisdate_,'cumdEPOPdUPOPinit'] = \
            results.data.loc[thisdate_,'dEPOPdEPOP_1']*results.data.loc[prevmonth_,'cumdEPOPdUPOPinit'] +\
            results.data.loc[thisdate_,'dEPOPdUPOP_1']*results.data.loc[prevmonth_,'cumdUPOPdUPOPinit']
        
        results.data.loc[thisdate_,'cumdUPOPdUPOPinit'] = \
            results.data.loc[thisdate_,'dUPOPdEPOP_1']*results.data.loc[prevmonth_,'cumdEPOPdUPOPinit'] +\
            results.data.loc[thisdate_,'dUPOPdUPOP_1']*results.data.loc[prevmonth_,'cumdUPOPdUPOPinit']
            
        results.data.loc[thisdate_,'cumdUPOPdEPOPinit'] = \
            results.data.loc[thisdate_,'dUPOPdEPOP_1']*results.data.loc[prevmonth_,'cumdEPOPdEPOPinit'] +\
            results.data.loc[thisdate_,'dUPOPdUPOP_1']*results.data.loc[prevmonth_,'cumdUPOPdEPOPinit']   
    
        for dstock in ['EPOP','UPOP']:
                        
            for dp in ps_:

                results.data.loc[thisdate_,'cumd'+dstock+'d'+dp] = results.data.loc[thisdate_,'d'+dstock+'d'+dp] + \
                    results.data.loc[thisdate_,'d'+dstock+'dEPOP_1']*results.data.loc[prevmonth_,'cumdEPOPd'+dp] + \
                    results.data.loc[thisdate_,'d'+dstock+'dUPOP_1']*results.data.loc[prevmonth_,'cumdUPOPd'+dp]
        
for dstock in ['EPOP','UPOP']:
            
    results.data['cumdLFPRd'+dstock+'init'] = results.data['cumdEPOPd'+dstock+'init']+results.data['cumdUPOPd'+dstock+'init']    
    
for dp in ps_:

    results.data['cumdLFPRd'+dp] = results.data['cumdEPOPd'+dp]+results.data['cumdUPOPd'+dp]

# ==============================================================================================
# Selecting part of results used for output
# ----------------------------------------------------------------------------------------------
try:
    del results.wb    
except:
    pass

decomp = cp.deepcopy(results)    
decompcols = ['dLFPR','cumdLFPRdEPOPinit' , 'cumdLFPRdUPOPinit' , 'cumdLFPRdpEU' , 'cumdLFPRdpEN' , 'cumdLFPRdpUE' , 'cumdLFPRdpUN' , 'cumdLFPRdpNE' , 'cumdLFPRdpNU' ]
decomp.data = decomp.data[decompcols]
newcols = decompcols = ['dLFPR','dEPOPinit' , 'dUPOPinit' , 'dpEU' , 'dpEN' , 'dpUE' , 'dpUN' , 'dpNE' , 'dpNU' ]        
decomp.data.columns = newcols

# ==============================================================================================
# AREA CHART:
# Area chart with the labor force participation rate as the sum of the employment-population
# ratio and the unemployment-population ratio
# ----------------------------------------------------------------------------------------------
if settings['Area chart']['Make'] == True:
    
    d = 100*results.data[['LFPR' , 'EPOP' , 'UPOP' ]]
    d['UPOP'] = d['EPOP']+d['UPOP']
    r = results.recessions
    
    chrt = bhc.Chart()
    chrt.name = settings['Gender']+'LFPRAreaChart'
    chrt.series.colors = [ 'k' , 'navy' , 'orange' ]
    chrt.series.linewidth = 3
    chrt.series.styles = [ '-' , '--' , '--' ]
    chrt.legend.labels = [ 'LFPR' , 'EPOP' , 'UPOP'  ]
    chrt.fonttype.type = 'sans-serif'
    chrt.fonttype.usetex = False
    chrt.template.type = 'slide169'
    chrt.recessionshading.visible = True
    
    chrt.titles.title.text = 'Labor force participation rate and its components'
    chrt.titles.subtitle.text = 'Monthly observations; seasonally adjusted; share of CNI population age 16+; '+settings["Gender"]
    chrt.titles.source.text = 'Source: Bureau of Labor Statistics'
    chrt.titles.ylabel.text = 'Percent' 
    
    chrt.notes.title= chrt.name
    chrt.notes.description = 'Labor force participation rate and employment- and unemployment-population ratio'
    chrt.notes.unitofmeasurement = 'Share of CNI population age 16+ (percent)'
    chrt.notes.datatransformation = 'Seasonally adjusted monthy observations'
    chrt.notes.interpretation = 'NA'
    chrt.notes.sourceURL = 'https://www.bls.gov'
    chrt.notes.sourceTitle = 'Bureau of Labor Statistics'
    chrt.notes.save(os.getcwd())
    
    # Generates the chart that is being saved for publication on the web
    if settings['Area chart']['Web'] == True:
        webchrt = cp.deepcopy(chrt)
        webchrt.template.type = 'mooc'
        webchrt.series.marker = 'o' 
        webchrt.initialize()
        webchrt.tsbuild(d.round(2),r)
        for seriesno in [1]:
            webchrt.plt.fill_between(webchrt.series.data.index, areachrty[settings["Gender"]][0] , webchrt.series.data[webchrt.series.data.columns[seriesno]], facecolor=webchrt.series.colors[seriesno]  , alpha = 0.5)
        for seriesno in [2]:
            webchrt.plt.fill_between(webchrt.series.data.index, webchrt.series.data[webchrt.series.data.columns[seriesno-1]], webchrt.series.data[webchrt.series.data.columns[seriesno]], facecolor=webchrt.series.colors[seriesno]  , alpha = 0.5)
        webchrt.plt.axes.set_xlim(results.startdate, results.enddate)
        webchrt.plt.axes.set_ylim(areachrty[settings["Gender"]][0],areachrty[settings["Gender"]][1])
        webchrt.arrange()
        webchrt.webtooltips.LinePointToolTips()
        webchrt.series.data = d.round(2)
        webchrt.webexport.save()
        
    pprchrt = cp.deepcopy(chrt)
    pprchrt.template.type = 'paper'
    pprchrt.fonttype.type = 'sans-serif'
    pprchrt.series.colors = [ 'k' , 'dimgray' , 'lightgray' ]
    pprchrt.series.marker = 0
    pprchrt.fonttype.usetex = False
    pprchrt.initialize()
    pprchrt.template.figsize = [12,3.5]
    pprchrt.legend.fontsize = 14
    pprchrt.titles.ylabel.fontsize = 16
    pprchrt.template.xticksize = 13
    pprchrt.template.yticksize = 13
    pprchrt.tsbuild(d,r)
    for seriesno in [1]:
        pprchrt.plt.fill_between(pprchrt.series.data.index, areachrty[settings["Gender"]][0] , pprchrt.series.data[pprchrt.series.data.columns[seriesno]], facecolor=pprchrt.series.colors[seriesno]  , alpha = 0.5)
    for seriesno in [2]:
        pprchrt.plt.fill_between(pprchrt.series.data.index, pprchrt.series.data[pprchrt.series.data.columns[seriesno-1]], pprchrt.series.data[pprchrt.series.data.columns[seriesno]], facecolor=pprchrt.series.colors[seriesno]  , alpha = 0.5)
    pprchrt.plt.axes.set_xlim(results.startdate, results.enddate)
    pprchrt.plt.axes.set_ylim(areachrty[settings["Gender"]][0],areachrty[settings["Gender"]][1])
    pprchrt.arrange()
    pprchrt.export.save()
    
    chrt.template.type = 'slide169'
    chrt.series.marker = 0
    chrt.fonttype.usetex = False
    chrt.initialize()
    chrt.tsbuild(d,r)
    for seriesno in [1]:
        chrt.plt.fill_between(chrt.series.data.index, areachrty[settings["Gender"]][0] , chrt.series.data[chrt.series.data.columns[seriesno]], facecolor=chrt.series.colors[seriesno]  , alpha = 0.5)
    for seriesno in [2]:
        chrt.plt.fill_between(chrt.series.data.index, chrt.series.data[chrt.series.data.columns[seriesno-1]], chrt.series.data[chrt.series.data.columns[seriesno]], facecolor=chrt.series.colors[seriesno]  , alpha = 0.5)
    chrt.plt.axes.set_xlim(results.startdate, results.enddate)
    chrt.plt.axes.set_ylim(areachrty[settings["Gender"]][0],areachrty[settings["Gender"]][1])
    chrt.arrange()
    chrt.export.save()
    
if settings['Line chart']['Make'] == True:
    
    d = 100*results.data['LFPR']
    urate = 100*results.data['URATE']
    r = results.recessions
    
    chrt = bhc.Chart()
    chrt.name = settings['Gender']+'LFPRLineChart'
    chrt.series.colors = [ 'k' , 'navy' , 'orange' ]
    chrt.series.linewidth = 3
    chrt.series.styles = [ '-' , '--' , '--' ]
    chrt.legend.labels = [ 'LFPR' , 'Unemployment rate' ]
    chrt.legend.visible = False
    chrt.fonttype.type = 'sans-serif'
    chrt.fonttype.usetex = False
    chrt.template.type = 'slide169'
    chrt.recessionshading.visible = True
    
    chrt.titles.title.text = 'Labor force participation rate and unemployment rate'
    chrt.titles.subtitle.text = 'Monthly observations; seasonally adjusted; share of CNI population age 16+; '+settings["Gender"]
    chrt.titles.source.text = 'Source: Bureau of Labor Statistics'
    chrt.titles.ylabel.text = 'Percent' 
        
    pprchrt = cp.deepcopy(chrt)
    pprchrt.template.type = 'paper'
    pprchrt.fonttype.type = 'sans-serif'
    pprchrt.series.colors = [ 'k' , 'dimgray' , 'lightgray' ]
    pprchrt.titles.ylabel.color = 'k'
    pprchrt.series.marker = 0
    pprchrt.fonttype.usetex = False
    pprchrt.initialize()
    pprchrt.template.figsize = [12,3.5]
    pprchrt.legend.fontsize = 14
    pprchrt.titles.ylabel.fontsize = 16
    pprchrt.template.xticksize = 13
    pprchrt.template.yticksize = 13
    pprchrt.tsbuild(d,r)
    plt2 = results.data['URATE'].plot(kind='line', color='dimgray',secondary_y=True)
    plt2.set_ylabel(ylabel='Unemployment rate (percent)',fontsize=pprchrt.titles.ylabel.fontsize,color = 'dimgray')
    plt2.spines["top"].set_visible(False)  
    plt2.spines["bottom"].set_visible(False)  
    plt2.spines["right"].set_visible(False)  
    plt2.spines["left"].set_visible(False)
    pprchrt.arrange()
    pprchrt.export.save()
    mpl.pyplot.close('all')
    
    chrt.initialize()
    chrt.template.type = 'slide169'
    chrt.series.marker = 0
    chrt.titles.ylabel.color = 'k'
    chrt.fonttype.usetex = False
    chrt.initialize()
    chrt.tsbuild(d,r)
    plt2 = results.data['URATE'].plot(kind='line', color='darkred',secondary_y=True)
    plt2.set_ylabel(ylabel='Unemployment rate (percent)',fontsize=chrt.titles.ylabel.fontsize,color = 'darkred')
    plt2.spines["top"].set_visible(False)  
    plt2.spines["bottom"].set_visible(False)  
    plt2.spines["right"].set_visible(False)  
    plt2.spines["left"].set_visible(False)
    chrt.arrange()
    chrt.export.save()    
    mpl.pyplot.close('all')
    
# ==============================================================================================
# DECOMPOSITION CHART:
# Line chart with the components of the decomposition
# ----------------------------------------------------------------------------------------------
if settings['Decomposition chart']['Make'] == True:
    
    d = 100*decomp.data.cumsum()
    r = decomp.recessions
    
    chrt = bhc.Chart()
    chrt.name = settings['Gender']+'LFPRDecomposed1'
    chrt.series.colors = [ 'k' , 'navy' , 'orange' , 'lightblue' , 'maroon' , 'brown' , 'grey' , 'pink' , 'green' ]
    chrt.series.linewidth = 3
    chrt.series.styles = [ '-' ]
    chrt.legend.labels =  ['Change in LFPR','Initial change in EPOP' , 'Initial change in UPOP' , 'pEU' , 'pEN' , 'pUE' , 'pUN' , 'pNE' , 'pNU' ]
    chrt.fonttype.type = 'sans-serif'
    chrt.fonttype.usetex = False
    chrt.template.type = 'slide169'
    chrt.recessionshading.visible = True
    
    chrt.titles.title.text = 'Change in Labor force participation rate decomposed'
    chrt.titles.subtitle.text = 'Monthly observations; seasonally adjusted; Percentage point change since March 1993; '+settings["Gender"]
    chrt.titles.source.text = 'Source: Bureau of Labor Statistics'
    chrt.titles.ylabel.text = 'Percentage point' 
    
    chrt.notes.title= chrt.name
    chrt.notes.description = 'Labor force participation rate decomposed'
    chrt.notes.unitofmeasurement = 'Percentage point change since March 1993'
    chrt.notes.datatransformation = 'Seasonally adjusted monthy observations'
    chrt.notes.interpretation = 'NA'
    chrt.notes.sourceURL = 'https://www.bls.gov'
    chrt.notes.sourceTitle = 'Bureau of Labor Statistics'
    chrt.notes.save(os.getcwd())
    
    # Generates the chart that is being saved for publication on the web
    if settings['Decomposition chart']['Web'] == True:
        webchrt = cp.deepcopy(chrt)
        webchrt.template.type = 'mooc'
        webchrt.series.marker = 'o' 
        webchrt.initialize()
        webchrt.tsbuild(d.round(2),r)
        webchrt.plt.axes.set_xlim(results.startdate, results.enddate)
        webchrt.arrange()
        webchrt.webtooltips.LinePointToolTips()
        webchrt.series.data = d.round(2)
        webchrt.webexport.save()
        
    pprchrt = cp.deepcopy(chrt)
    pprchrt.template.type = 'paper'
    pprchrt.fonttype.type = 'sans-serif'
#    pprchrt.series.colors = [ 'k' , 'dimgray' , 'lightgray' ]
    pprchrt.series.marker = 0
    pprchrt.fonttype.usetex = False
    pprchrt.initialize()
    pprchrt.tsbuild(d,r)
    pprchrt.plt.axes.set_xlim(results.startdate, results.enddate)
    pprchrt.arrange()
    pprchrt.export.save()
    
    chrt.template.type = 'slide169'
    chrt.series.marker = 0
    chrt.fonttype.usetex = False
    chrt.initialize()
    chrt.tsbuild(d,r)
    chrt.plt.axes.set_xlim(results.startdate, results.enddate)
    chrt.arrange()
    chrt.export.save() 
    
if settings['Decomposition chart']['Make'] == True:
    
    d = d.subtract(d.loc[dt.datetime(2007,1,1)])
    r = decomp.recessions
    
    chrt = bhc.Chart()
    chrt.name = settings['Gender']+'LFPRDecomposed1a'
    chrt.series.colors = [ 'k' , 'navy' , 'orange' , 'lightblue' , 'maroon' , 'brown' , 'grey' , 'pink' , 'green' ]
    chrt.series.linewidth = 3
    chrt.series.styles = [ '-' ]
    chrt.legend.labels =  ['Change in LFPR','Initial change in EPOP' , 'Initial change in UPOP' , 'pEU' , 'pEN' , 'pUE' , 'pUN' , 'pNE' , 'pNU' ]
    chrt.fonttype.type = 'sans-serif'
    chrt.fonttype.usetex = False
    chrt.template.type = 'slide169'
    chrt.recessionshading.visible = True
    
    chrt.titles.title.text = 'Change in Labor force participation rate decomposed'
    chrt.titles.subtitle.text = 'Monthly observations; seasonally adjusted; Percentage point change since January 2007; '+settings["Gender"]
    chrt.titles.source.text = 'Source: Bureau of Labor Statistics'
    chrt.titles.ylabel.text = 'Percentage point' 
    
    chrt.notes.title= chrt.name
    chrt.notes.description = 'Labor force participation rate decomposed'
    chrt.notes.unitofmeasurement = 'Percentage point change since January 2007'
    chrt.notes.datatransformation = 'Seasonally adjusted monthy observations'
    chrt.notes.interpretation = 'NA'
    chrt.notes.sourceURL = 'https://www.bls.gov'
    chrt.notes.sourceTitle = 'Bureau of Labor Statistics'
    chrt.notes.save(os.getcwd())
    
    # Generates the chart that is being saved for publication on the web
    if settings['Decomposition chart']['Web'] == True:
        webchrt = cp.deepcopy(chrt)
        webchrt.template.type = 'mooc'
        webchrt.series.marker = 'o' 
        webchrt.initialize()
        webchrt.tsbuild(d.round(2),r)
        webchrt.plt.axes.set_xlim(results.startdate, results.enddate)
        webchrt.arrange()
        webchrt.webtooltips.LinePointToolTips()
        webchrt.series.data = d.round(2)
        webchrt.webexport.save()
        
    pprchrt = cp.deepcopy(chrt)
    pprchrt.template.type = 'paper'
    pprchrt.fonttype.type = 'sans-serif'
#    pprchrt.series.colors = [ 'k' , 'dimgray' , 'lightgray' ]
    pprchrt.series.marker = 0
    pprchrt.fonttype.usetex = False
    pprchrt.initialize()
    pprchrt.tsbuild(d,r)
    pprchrt.plt.axes.set_xlim(results.startdate, results.enddate)
    pprchrt.arrange()
    pprchrt.export.save()
    
    chrt.template.type = 'slide169'
    chrt.series.marker = 0
    chrt.fonttype.usetex = False
    chrt.initialize()
    chrt.tsbuild(d,r)
    chrt.plt.axes.set_xlim(results.startdate, results.enddate)
    chrt.arrange()
    chrt.export.save()     


decomp.data['Change in LFPR'] = decomp.data['dLFPR']
decomp.data['Initial state'] = decomp.data['dEPOPinit']+decomp.data['dUPOPinit'] 
decomp.data['Churn'] = decomp.data['dpEU']+decomp.data['dpUE'] 
decomp.data['Net entry to employment'] = decomp.data['dpNE']+decomp.data['dpEN']
decomp.data['Net entry to unemployment'] = decomp.data['dpNU']+decomp.data['dpUN']
decomp.data['Entry'] = decomp.data['dpNE']+decomp.data['dpNU']
decomp.data['Exit'] = decomp.data['dpEN']+decomp.data['dpUN']

plotvars = ['Change in LFPR','Initial state','Churn','Net entry to employment','Net entry to unemployment']

# ==============================================================================================
# DECOMPOSITION CHART:
# Line chart with the components of the decomposition
# ----------------------------------------------------------------------------------------------

if settings['Decomposition2 chart']['Make'] == True:
    
    d = 100*decomp.data[plotvars].cumsum()
    r = decomp.recessions
    
    chrt = bhc.Chart()
    chrt.name = settings['Gender']+'LFPRDecomposed2'
    chrt.series.colors = [ 'k' , 'navy' , 'orange' , 'lightblue' , 'maroon' , 'brown' , 'grey' , 'pink' , 'green' ]
    chrt.series.linewidth = 3
    chrt.series.styles = [ '-' ]
    chrt.legend.labels =  plotvars
    chrt.fonttype.type = 'sans-serif'
    chrt.fonttype.usetex = False
    chrt.template.type = 'slide169'
    chrt.recessionshading.visible = True
    
    chrt.titles.title.text = 'Change in Labor force participation rate decomposed'
    chrt.titles.subtitle.text = 'Monthly observations; seasonally adjusted; Percentage point change since March 1993; '+settings["Gender"]
    chrt.titles.source.text = 'Source: Bureau of Labor Statistics'
    chrt.titles.ylabel.text = 'Percentage point' 
    
    chrt.notes.title= chrt.name
    chrt.notes.description = 'Labor force participation rate decomposed'
    chrt.notes.unitofmeasurement = 'Percentage point change since March 1993'
    chrt.notes.datatransformation = 'Seasonally adjusted monthy observations'
    chrt.notes.interpretation = 'NA'
    chrt.notes.sourceURL = 'https://www.bls.gov'
    chrt.notes.sourceTitle = 'Bureau of Labor Statistics'
    chrt.notes.save(os.getcwd())
    
    # Generates the chart that is being saved for publication on the web
    if settings['Decomposition2 chart']['Web'] == True:
        webchrt = cp.deepcopy(chrt)
        webchrt.template.type = 'mooc'
        webchrt.series.marker = 'o' 
        webchrt.initialize()
        webchrt.tsbuild(d.round(2),r)
        webchrt.plt.axes.set_xlim(results.startdate, results.enddate)
        webchrt.arrange()
        webchrt.webtooltips.LinePointToolTips()
        webchrt.series.data = d.round(2)
        webchrt.webexport.save()
        
    pprchrt = cp.deepcopy(chrt)
    pprchrt.template.type = 'paper'
    pprchrt.fonttype.type = 'sans-serif'
#    pprchrt.series.colors = [ 'k' , 'dimgray' , 'lightgray' ]
    pprchrt.series.marker = 0
    pprchrt.fonttype.usetex = False
    pprchrt.initialize()
    pprchrt.tsbuild(d,r)
    pprchrt.plt.axes.set_xlim(results.startdate, results.enddate)
    pprchrt.arrange()
    pprchrt.export.save()
    
    chrt.template.type = 'slide169'
    chrt.series.marker = 0
    chrt.fonttype.usetex = False
    chrt.initialize()
    chrt.tsbuild(d,r)
    chrt.plt.axes.set_xlim(results.startdate, results.enddate)
    chrt.arrange()
    chrt.export.save()
    
# ==============================================================================================
# DECOMPOSITION CHART:
# Line chart with the components of the decomposition
# ----------------------------------------------------------------------------------------------

if settings['Decomposition2 chart']['Make'] == True:
    
    d = d = d.subtract(d.loc[dt.datetime(2007,1,1)])
    r = decomp.recessions
    
    chrt = bhc.Chart()
    chrt.name = settings['Gender']+'LFPRDecomposed2a'
    chrt.series.colors = [ 'k' , 'navy' , 'orange' , 'lightblue' , 'maroon' , 'brown' , 'grey' , 'pink' , 'green' ]
    chrt.series.linewidth = 3
    chrt.series.styles = [ '-' ]
    chrt.legend.labels =  plotvars
    chrt.fonttype.type = 'sans-serif'
    chrt.fonttype.usetex = False
    chrt.template.type = 'slide169'
    chrt.recessionshading.visible = True
    
    chrt.titles.title.text = 'Change in Labor force participation rate decomposed'
    chrt.titles.subtitle.text = 'Monthly observations; seasonally adjusted; Percentage point change since January 2007; '+settings["Gender"]
    chrt.titles.source.text = 'Source: Bureau of Labor Statistics'
    chrt.titles.ylabel.text = 'Percentage point' 
    
    chrt.notes.title= chrt.name
    chrt.notes.description = 'Labor force participation rate decomposed'
    chrt.notes.unitofmeasurement = 'Percentage point change since January 2007'
    chrt.notes.datatransformation = 'Seasonally adjusted monthy observations'
    chrt.notes.interpretation = 'NA'
    chrt.notes.sourceURL = 'https://www.bls.gov'
    chrt.notes.sourceTitle = 'Bureau of Labor Statistics'
    chrt.notes.save(os.getcwd())
    
    # Generates the chart that is being saved for publication on the web
    if settings['Decomposition2 chart']['Web'] == True:
        webchrt = cp.deepcopy(chrt)
        webchrt.template.type = 'mooc'
        webchrt.series.marker = 'o' 
        webchrt.initialize()
        webchrt.tsbuild(d.round(2),r)
        webchrt.plt.axes.set_xlim(results.startdate, results.enddate)
        webchrt.arrange()
        webchrt.webtooltips.LinePointToolTips()
        webchrt.series.data = d.round(2)
        webchrt.webexport.save()
        
    pprchrt = cp.deepcopy(chrt)
    pprchrt.template.type = 'paper'
    pprchrt.fonttype.type = 'sans-serif'
#    pprchrt.series.colors = [ 'k' , 'dimgray' , 'lightgray' ]
    pprchrt.series.marker = 0
    pprchrt.fonttype.usetex = False
    pprchrt.initialize()
    pprchrt.tsbuild(d,r)
    pprchrt.plt.axes.set_xlim(results.startdate, results.enddate)
    pprchrt.arrange()
    pprchrt.export.save()
    
    chrt.template.type = 'slide169'
    chrt.series.marker = 0
    chrt.fonttype.usetex = False
    chrt.initialize()
    chrt.tsbuild(d,r)
    chrt.plt.axes.set_xlim(results.startdate, results.enddate)
    chrt.arrange()
    chrt.export.save()    

plotvars = ['Change in LFPR','Exit','Entry','Churn']

# ==============================================================================================
# DECOMPOSITION CHART:
# Line chart with the components of the decomposition
# ----------------------------------------------------------------------------------------------

if settings['Decomposition3 chart']['Make'] == True:
    
    d = 100*decomp.data[plotvars].cumsum()
    r = decomp.recessions
    
    chrt = bhc.Chart()
    chrt.name = settings['Gender']+'LFPRDecomposed3'
    chrt.series.colors = [ 'k' , 'navy' , 'orange' , 'maroon' , 'brown' , 'grey' , 'pink' , 'green' ]
    chrt.series.linewidth = 3
    chrt.series.styles = [ '-' ]
    chrt.legend.labels =  plotvars
    chrt.fonttype.type = 'sans-serif'
    chrt.fonttype.usetex = False
    chrt.template.type = 'slide169'
    chrt.recessionshading.visible = True
    
    chrt.titles.title.text = 'Change in Labor force participation rate decomposed'
    chrt.titles.subtitle.text = 'Monthly observations; seasonally adjusted; Percentage point change since March 1993; '+settings["Gender"]
    chrt.titles.source.text = 'Source: Bureau of Labor Statistics'
    chrt.titles.ylabel.text = 'Percentage point' 
    
    chrt.notes.title= chrt.name
    chrt.notes.description = 'Labor force participation rate decomposed'
    chrt.notes.unitofmeasurement = 'Percentage point change since March 1993'
    chrt.notes.datatransformation = 'Seasonally adjusted monthy observations'
    chrt.notes.interpretation = 'NA'
    chrt.notes.sourceURL = 'https://www.bls.gov'
    chrt.notes.sourceTitle = 'Bureau of Labor Statistics'
    chrt.notes.save(os.getcwd())
    
    # Generates the chart that is being saved for publication on the web
    if settings['Decomposition3 chart']['Web'] == True:
        webchrt = cp.deepcopy(chrt)
        webchrt.template.type = 'mooc'
        webchrt.series.marker = 'o' 
        webchrt.initialize()
        webchrt.tsbuild(d.round(2),r)
        webchrt.plt.axes.set_xlim(results.startdate, results.enddate)
        webchrt.arrange()
        webchrt.webtooltips.LinePointToolTips()
        webchrt.series.data = d.round(2)
        webchrt.webexport.save()
        
    pprchrt = cp.deepcopy(chrt)
    pprchrt.template.type = 'paper'
    pprchrt.fonttype.type = 'sans-serif'
#    pprchrt.series.colors = [ 'k' , 'dimgray' , 'lightgray' ]
    pprchrt.series.marker = 0
    pprchrt.fonttype.usetex = False
    pprchrt.initialize()
    pprchrt.tsbuild(d,r)
    pprchrt.plt.axes.set_xlim(results.startdate, results.enddate)
    pprchrt.arrange()
    pprchrt.export.save()
    
    chrt.template.type = 'slide169'
    chrt.series.marker = 0
    chrt.fonttype.usetex = False
    chrt.initialize()
    chrt.tsbuild(d,r)
    chrt.plt.axes.set_xlim(results.startdate, results.enddate)
    chrt.arrange()
    chrt.export.save()
    
# ==============================================================================================
# DECOMPOSITION CHART:
# Line chart with the components of the decomposition
# ----------------------------------------------------------------------------------------------

if settings['Decomposition3 chart']['Make'] == True:
    
    d = d.subtract(d.loc[dt.datetime(2007,1,1)])
    r = decomp.recessions
    
    chrt = bhc.Chart()
    chrt.name = settings['Gender']+'LFPRDecomposed3a'
    chrt.series.colors = [ 'k' , 'navy' , 'orange' , 'maroon' , 'brown' , 'grey' , 'pink' , 'green' , 'lightblue' ]
    chrt.series.linewidth = 3
    chrt.series.styles = [ '-' , '--' , ':' , '-.' ]
    chrt.legend.labels =  plotvars
    chrt.fonttype.type = 'sans-serif'
    chrt.fonttype.usetex = False
    chrt.template.type = 'slide169'
    chrt.recessionshading.visible = True
    
    chrt.titles.title.text = 'Change in Labor force participation rate decomposed'
    chrt.titles.subtitle.text = 'Monthly observations; seasonally adjusted; Percentage point change since January 2007; '+settings["Gender"]
    chrt.titles.source.text = 'Source: Bureau of Labor Statistics'
    chrt.titles.ylabel.text = 'Percentage point' 
    
    chrt.notes.title= chrt.name
    chrt.notes.description = 'Labor force participation rate decomposed'
    chrt.notes.unitofmeasurement = 'Percentage point change since January 2007'
    chrt.notes.datatransformation = 'Seasonally adjusted monthy observations'
    chrt.notes.interpretation = 'NA'
    chrt.notes.sourceURL = 'https://www.bls.gov'
    chrt.notes.sourceTitle = 'Bureau of Labor Statistics'
    chrt.notes.save(os.getcwd())
    
    # Generates the chart that is being saved for publication on the web
    if settings['Decomposition3 chart']['Web'] == True:
        webchrt = cp.deepcopy(chrt)
        webchrt.template.type = 'mooc'
        webchrt.series.marker = 'o' 
        webchrt.initialize()
        webchrt.tsbuild(d.round(2),r)
        webchrt.plt.axes.set_xlim(results.startdate, results.enddate)
        webchrt.arrange()
        webchrt.webtooltips.LinePointToolTips()
        webchrt.series.data = d.round(2)
        webchrt.webexport.save()
        
    pprchrt = cp.deepcopy(chrt)
    pprchrt.template.type = 'paper'
    pprchrt.fonttype.type = 'sans-serif'
    pprchrt.series.colors = [ 'k' , 'lightgray' , 'dimgray' , 'gray'  ]
    pprchrt.series.styles = [ '-' , ':' , '--' , '-.'  ]
    pprchrt.series.marker = 0
    pprchrt.fonttype.usetex = False
    pprchrt.initialize()
    pprchrt.template.figsize = [12,3.5]
    pprchrt.legend.fontsize = 13
    if settings["Gender"] == "Total":
        pprchrt.legend.visible = True
    else:
        pprchrt.legend.visible = False
    pprchrt.titles.ylabel.fontsize = 16
    pprchrt.template.xticksize = 13
    pprchrt.template.yticksize = 13
    pprchrt.tsbuild(d,r)
    pprchrt.plt.axes.set_xlim(results.startdate, results.enddate)
    pprchrt.arrange()
    pprchrt.export.save()
    
    chrt.template.type = 'slide169'
    chrt.series.marker = 0
    chrt.fonttype.usetex = False
    chrt.initialize()
    chrt.tsbuild(d,r)
    chrt.plt.axes.set_xlim(results.startdate, results.enddate)
    chrt.arrange()
    chrt.export.save()  
    
    ylims = chrt.plt.axes.get_ylim()
    
    if  settings['Decomposition3 chart']['Animate'] == True:
        
        for anfrm in range(1,5):
            
            dfrm = d[plotvars[:anfrm]]
    
            chrt.name = settings['Gender']+'LFPRDecomposed3a'+"_frame"+str(anfrm)        
            mpl.pyplot.close(chrt.fig)
            chrt.legend.labels =  plotvars[:anfrm]
            chrt.tsbuild(dfrm,r)
            chrt.plt.axes.set_xlim(results.startdate, results.enddate)
            chrt.plt.axes.set_ylim(ylims[0], ylims[1])
            chrt.arrange()
            chrt.export.save()
            mpl.pyplot.close(chrt.fig)
        
 
# ==============================================================================================
# Saving a table with the summary statistics
# ============================================================================================== 
    
totable = results.data[['EPOP','UPOP','pUEm', 'pUNm','pEUm', 'pENm', 'pNUm', 'pNEm']]    
totable['NPOP'] = 1-totable['EPOP']-totable['UPOP']
totable = 100*totable.mean(axis=0)
idx = ['E','U','N']
cols = ['from E','from U','from N','stock']
table = pd.DataFrame(index=idx,columns=cols)    
table.iloc[0,0] = (100 - totable['pEUm']- totable['pENm'])
table.iloc[1,0] = (totable['pEUm'])
table.iloc[2,0] = (totable['pENm'])
table.iloc[0,1] = (totable['pUEm'])
table.iloc[1,1] = (100 - totable['pUEm']- totable['pUNm'])
table.iloc[2,1] = (totable['pUNm'])
table.iloc[0,2] = (totable['pNEm'])
table.iloc[1,2] = (totable['pNUm'])
table.iloc[2,2] = (100 - totable['pNEm']- totable['pNUm'])
table.iloc[0,3] = (totable['EPOP'])
table.iloc[1,3] = (totable['UPOP'])
table.iloc[2,3] = (totable['NPOP'])

with open('./tex/'+settings['Gender']+settings['Load']['vintage']+'_table.tex', 'w') as tf:
     tf.write(table.astype(float).round(2).to_latex())
    
# ==============================================================================================
# Saving the results to an excel file
# ==============================================================================================
results.toExcel() 
res = results.data   

if settings["Closeplots"] == True:
    mpl.pyplot.close('all')

    
    

