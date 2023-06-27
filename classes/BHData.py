# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 10:55:03 2015

BHData:
This module contains classes that allow you to load data from different statistical agencies and online
data repositories into Pandas Dataframes. Currently supported are

1) FRED at the St. Louis Fed
2) NIPA data and underlying detail from the BEA
3) BLS time series data
4) Quandl data
5) BEA NIPA tables
6) OECD data

@author: bart
"""

import datetime as dt
import pandas as pd
import numpy as np 
try:
    import pandas_datareader.data as web
    from pandas_datareader.io import read_jsdmx
except:
    pd.core.common.is_list_like = pd.api.types.is_list_like
    import pandas_datareader.data as web
    from pandas_datareader.io import read_jsdmx

# from pandas_datareader import data as pdr
# import fix_yahoo_finance as yf
try:
    import BHBEA
except:
    from classes import BHBEA
import requests
import json
import time

try:
    import Quandl as qd
except:
    try:
        import quandl as qd
    except:
        print("No Quandl installed")
from pandas.io.json import json_normalize

sourcetypes = [ 'fred' , 'BEANIPA' , 'BLSTS' , 'Quandl', 'Yahoo' , 'OECD' ]
" Time series data class "

def quarterstr(tmstmp):
    
    qstr = str(tmstmp.year)+'Q'+str(tmstmp.quarter)    
    
    return qstr
    
def monthstr(tmstmp):
    
    mstr = tmstmp.strftime("%B-%Y")
    
    return mstr        

class TSData(object):
    
    def __init__(self,start,end,freq='MS'):        
        
        # yf.pdr_override() # <== that's all it takes :-)
        
        self.recessions = 0        
        
        self.interpolate = True
        self.interpolatelimit = 2
        self.startdate = start
        self.enddate =  end
        self.frequency = freq
        self.dates = pd.date_range(start=self.startdate,end=self.enddate,freq=self.frequency)        
        
        dum = pd.Series(index=self.dates)
        self.data = pd.DataFrame.from_dict(dum)
        self.data.columns = ['dum'] 
        
        self.name = ""
        self.path = ""
        self.date = ""
        self.info = ""
        
        self.seriesplotted = False
        self.transformation = ['level']
        self.source = [ 'fred' ]
        
        self.recessionseries = 'USREC'
        
        self.debug = 0
        
    def interpolatedata(self):

        self.__interpolatedata()
        
        return self.data

    def constructdata(self):
        
        self.date = dt.datetime.now()        
        
        dum = pd.Series(index=self.dates)
        self.data = pd.DataFrame.from_dict(dum)
        self.data.columns = ['dum'] 
        self.data['dum'] = 0        
        
        if not ( self.seriesplotted == False ):
            
            for seriesno in range( 0 , len(self.seriesplotted) ):
                
                if len(self.source) == 1:                
                    thissource = self.source[0]            
                else:                
                    thissource = self.source[seriesno]
            
                if (thissource in sourcetypes):
                    
                    if ( thissource == 'fred' ):
                        __series = web.DataReader(self.seriesplotted[seriesno],thissource,self.startdate,self.enddate)
                    elif ( thissource == 'BEANIPA' ):
                        __series = self.__loadBEANIPAdata(self.seriesplotted[seriesno][0],self.seriesplotted[seriesno][1])
                    elif ( thissource == 'BLSTS' ):
                        __series = self.__loadBLSTSdata(self.seriesplotted[seriesno])
#                        self.debug = __series
                    elif ( thissource == 'Quandl' ):
                        __series = self.__loadQuandldata(self.seriesplotted[seriesno])                    
                    elif ( thissource == 'Yahoo' ):
                        __series = web.DataReader(self.seriesplotted[seriesno],'yahoo',self.startdate,self.enddate)                       
                        __series = pd.DataFrame(__series['Close'])
                        __series.columns = [self.seriesplotted[seriesno]]
                    elif ( thissource == 'OECD' ):
                        __series = self.__loadOECDdata(self.seriesplotted[seriesno])
                    else:
                        print('This should not be possible')
                           
                    self.debug = self.data       
                    self.data = pd.merge(self.data, __series , left_index=True, right_index=True , how='left')                    
                    
            self.data = self.data.drop('dum',1)        
        
        self.recessions = web.DataReader(self.recessionseries,'fred',self.startdate, self.enddate)
        firstrecdate = self.recessions.index.min()
        lastrecdate = self.recessions.index.max()
        
        self.data = pd.merge(self.data, self.recessions , left_index=True, right_index=True , how='left')
        if self.seriesplotted == 0:
            self.recessions = self.data.drop('dum',1)
        else:
            self.recessions = self.data[[self.recessionseries]]
            
        self.data = self.data.drop(self.recessionseries,1)

        self.recessions = self.recessions.interpolate()
        self.recessions[self.recessionseries][self.recessions.index < firstrecdate] = 0
        self.recessions[self.recessionseries][self.recessions.index > lastrecdate] = 0

        if ( self.interpolate ):
            self.__interpolatedata()

    def __interpolatedata(self):
        """
        Interpolates the data and then fills in missings beyond and before first observations.
        """
        for seriesno in range( 0 , self.data.columns.size ):
            if (seriesno == 0):
                firstobs = [ self.data[self.data.columns[seriesno]].first_valid_index() ]
                lastobs = [ self.data[self.data.columns[seriesno]].last_valid_index() ]
            else:
                firstobs.append( self.data[self.data.columns[seriesno]].first_valid_index() )
                lastobs.append( self.data[self.data.columns[seriesno]].last_valid_index() )
            
        self.data = self.data.interpolate(limit=self.interpolatelimit)
        for seriesno in range( 0 , self.data.columns.size ):
            
            self.data[self.data.columns[seriesno]][self.data.index < firstobs[seriesno]] = np.nan
            self.data[self.data.columns[seriesno]][self.data.index > lastobs[seriesno]] = np.nan
            
    def __loadBEANIPAdata(self,NIPATable,linenumbers):
        
        freq = self.frequency[0]
        if freq == 'Y':
            freq = 'A'
        startyr = self.startdate.year
        endyr = self.enddate.year
        
        BHBEA.set_user_id('3A5B39F9-C5E8-4DA0-BD4F-F9A1EAC569A4')
        
        Yearslist = list(range(startyr,endyr+1))
        DataSetList = BHBEA.get_data_set_list()
        
        if NIPATable.endswith('U'):
            DataSetSelected = DataSetList['DatasetName'][2]
        else:
            DataSetSelected = DataSetList['DatasetName'][1]
        
        ListOfParameters = BHBEA.get_parameter_list(DataSetName=DataSetSelected)
        
        ListOfTables = BHBEA.get_parameter_values(DataSetName=DataSetSelected,
                                               ParameterName='TableID')
        TablesDescriptionString = ListOfTables['Description'].astype(str)                                       
                                               
        TableSelected = ListOfTables['TableID'][TablesDescriptionString.str.contains(NIPATable)]
        TableDescription = ListOfTables['Description'][TablesDescriptionString.str.contains(NIPATable)]
        
        Data = BHBEA.get_data(DataSetName=DataSetSelected,TableIDs=[ TableSelected ],Frequency=freq,Year=Yearslist)                                     
        Data2 = Data
        
        if freq == 'Q':
            yrqtr = Data['TimePeriod'].str.split("Q",expand=True)   
            # Generating a column with the date
            yrqtr = yrqtr.convert_objects(convert_numeric=True)
            Data['Date'] = pd.to_datetime(yrqtr[0]*10000 + (3*(yrqtr[1]-1)+1)*100 + 1, format='%Y%m%d')
        elif freq == 'M':
            yrqtr = Data['TimePeriod'].str.split("M",expand=True)   
            # Generating a column with the date
            yrqtr = yrqtr.convert_objects(convert_numeric=True)
            Data['Date'] = pd.to_datetime(yrqtr[0]*10000 + yrqtr[1]*100 + 1, format='%Y%m%d')
        elif freq == 'A':
            yrqtr = Data['TimePeriod'].convert_objects(convert_numeric=True)
            Data['Date'] = pd.to_datetime(yrqtr*10000 + 100 + 1, format='%Y%m%d')
        
        LineDescriptions = Data[['LineNumber','LineDescription','SeriesCode']][Data['LineNumber'].isin(linenumbers)]    
        Data = Data[['Date','LineNumber','DataValue']][ Data['LineNumber'].isin(linenumbers) ]
        LineDescriptions['LineNumber'] = NIPATable + "." + LineDescriptions['LineNumber'].astype(str)
        LineDescriptions = LineDescriptions.drop_duplicates(subset='LineNumber')
        LineDescriptions.index = LineDescriptions['LineNumber']
        LineDescriptions = LineDescriptions[['LineDescription','SeriesCode']]
        Data['LineNumber'] = NIPATable + "." + Data['LineNumber'].astype(str)
        Data = Data.pivot('Date','LineNumber','DataValue')
        Data = Data.replace(to_replace=',', value='', regex=True )
        Data = Data.convert_objects(convert_numeric=True)


        self.debug = Data

        return Data      

    def __loadBLSTSdata(self,BLSseries):

        headers = {'Content-type': 'application/json'}
        data = json.dumps({"seriesid": [BLSseries],"startyear":str(self.startdate.year), "endyear":str(self.enddate.year),"registrationKey":"ed07b46421604d76a6b3895d285d9325"})
        p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
        json_data = json.loads(p.text)
        Data2 = pd.DataFrame.from_dict(self.data['dum'])
        Data2.columns = ['dum']
        
        for series in json_data['Results']['series']:
            self.debug = BLSseries
            x = json_normalize(series['data'])
            x = x[['year' , 'period' , 'value']]
            x['freq'] = x['period'].str[:1]
            x['period'] = x['period'].str[1:]
            for s in ['year' , 'value' , 'period' ]: x[s] = pd.to_numeric(x[s])
            mth = x['period']
            mth[x['freq']=='Q'] = (3*(mth[x['freq']=='Q']-1)+1)
            x['month'] = mth
            x.index = pd.to_datetime(x['year']*10000 + x['month']*100 + 1, format='%Y%m%d')
            x = x.rename(columns = {'value':series['seriesID']})
            x = x[series['seriesID']]
            x = pd.DataFrame.from_dict(x)
            x.columns = [series['seriesID']]
            x = x.sort_index()
            self.debug = x
            Data2 = pd.merge(Data2, x , left_index=1, right_index=1 , how='left')
            self.debug2 = Data2
        Data2 = Data2.drop('dum',1)   
           
        return Data2  
        
    def __loadQuandldata(self,QuandlSeries):

        freq = self.frequency[0]
        api_id = "1GbrYar_zzUwB8tyUHEk"
        startdtstr = self.startdate.strftime("%Y-%m-%d")
        enddtstr = self.enddate.strftime("%Y-%m-%d")
        
        if freq == "M":
            collapse = "monthly"
        elif freq == "Q":
            collapse = "quarterly"
        elif freq == "A":
            collapse = "annual"
        else:
            collapse = ""
        
        Data = qd.get(QuandlSeries, collapse="", start_date=startdtstr , end_date=enddtstr , api_key=api_id)
        
        self.debug = Data

        if freq == "M":
            Data.index = Data.index + pd.offsets.MonthBegin(-1)
        elif freq == "Q":
            Data.index = Data.index + pd.offsets.QuarterBegin()
            # Data.index = Data.index + pd.offsets.MonthBegin(-2)
        elif freq == "A":
            Data.index = Data.index + pd.offsets.YearBegin(-1)

        Data.columns = [ QuandlSeries ]

        self.debug2 = Data

        return Data
        
    def __loadOECDdata(self,OECDSeries):

        dataset = OECDSeries[0]
        dataitems = OECDSeries[1]
        
        urlstr = 'http://stats.oecd.org/restsdmx/sdmx.ashx/GetData/'+dataset+'/'+dataitems+'/all?'

        Data = read_jsdmx(urlstr)
        try:
            idx_name = Data.index.name  # hack for pandas 0.16.2
            Data.index = pd.to_datetime(Data.index)
            Data = Data.sort_index()
            Data.index.name = idx_name
        except ValueError:
            pass        
        
        return Data
    
    def toExcel(self):
        
        self.data.columns = pd.MultiIndex.from_tuples([ (x,y,z) for x in [ self.info ] for y in [ self.date ] for z in self.data.columns.values ] )
        self.data.columns.names = ['Info' , 'Date' , 'Variable' ]
    
        self.recessions.columns = pd.MultiIndex.from_tuples([ (x,y,z) for x in [ self.info ] for y in [ self.date ] for z in self.recessions.columns.values ] )
        self.recessions.columns.names = ['Info' , 'Date' , 'RecessionInd' ]
        
        self.wb = pd.ExcelWriter(self.path+self.name+'.xlsx', engine='xlsxwriter')
            
        self.data.to_excel(self.wb,'Data')
        self.recessions.to_excel(self.wb,'Recessions')
            
        self.wb.save()
        
        self.data.columns = self.data.columns.droplevel('Info')
        self.data.columns = self.data.columns.droplevel('Date')
        self.recessions.columns = self.recessions.columns.droplevel('Info')
        self.recessions.columns = self.recessions.columns.droplevel('Date')        
        
    def fromExcel(self):
        
        if self.name != "" :
            
            self.wb = pd.ExcelFile(self.path+self.name+'.xlsx')
            
            self.data = pd.read_excel(self.wb,'Data',index_col=0,header=[0,1,2]) 
            self.recessions = pd.read_excel(self.wb,'Recessions',index_col=0,header=[0,1,2])
            
            self.info = self.data.columns.get_level_values('Info')[0]
            self.date = self.data.columns.get_level_values('Date')[0]
            
            self.data.columns = self.data.columns.droplevel('Info')
            self.data.columns = self.data.columns.droplevel('Date')
            self.recessions.columns = self.recessions.columns.droplevel('Info')
            self.recessions.columns = self.recessions.columns.droplevel('Date') 
 
            
        