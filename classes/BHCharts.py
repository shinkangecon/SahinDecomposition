# -*- coding: utf-8 -*-
"""
update on BHCharts

"""

import pandas as pd
import matplotlib as mpl
import numpy as np
import datetime as dt
import plotly.plotly as py
import sys
import mpld3
import copy as cp
import os

os.environ['PATH'] = os.environ['PATH'] + ':/Library/Tex/texbin'

templatetypes = [ 'slide', 'slide169' , 'mooc' , 'paper' ]
fonttypes = ['serif','sans-serif']

class Chart(object):
    
    def __init__(self):
    
        self.name = 'NoName'
        
        # Setting the sub-objects that have to do with formatting, data, and titles       
        
        self.fonttype = Fonttype(self)
        self.template = Template(self)        
        self.legend = Legend(self) 
        self.series = Series(self)
        self.titles = Titles(self)
        self.recessionshading = Recessionshading(self)
        self.eventlines = Eventlines(self)
        self.export = Export(self)
        self.webexport = WebExport(self)
        self.webtooltips = WebToolTips(self)
        self.notes = Notes(self)
        # self.map = Map(self)
        
        self.fig = 0
        self.plt = 0
    
    
    def initialize(self):    
    
        self.fonttype.apply()
        self.template.apply()
    
    def bind(self,myplt):
        # Method that binds the Chart object to an existing matplotlib.axes.AxesSubplot object
        # Generate this plot after calling the initialize method that sets necessary matplotlib variables.

        self.plt = myplt 
        self.fig = self.plt.get_figure()                

    def tsbuild(self,data,r=0):

        self.series.data = data

        if self.series.marker == 0 :
            self.plt = self.series.data.plot(legend=False, linewidth = self.series.linewidth, color = self.series.colors , style = self.series.styles , figsize = self.template.figsize , rot=0 , fontsize=self.template.fontsize )
        else:
            self.plt = self.series.data.plot(legend=False, linewidth = self.series.linewidth, color = self.series.colors , style = self.series.styles , marker = self.series.marker , markeredgecolor = 'none' , markersize = self.series.markersize , figsize = self.template.figsize , rot=0 , fontsize=self.template.fontsize )
            
        self.recessionshading.series = r    
            
        self.fig = self.plt.get_figure() 
        
    def xybuild(self,x,y):        
        # This still needs to be implemented
    
        self.plt = 0
        self.fig = 0

    def get_bound(self):
        
        bound = isinstance( self.fig , mpl.figure.Figure )
        
        return bound

    def arrange(self):

        if self.get_bound():
            
            rect = self.fig.patch
            rect.set_facecolor('white')
                        
            self.template.arrange()
            self.titles.arrange()
            
            if self.titles.xlabel.visible:
                xlabelpadding = 0.0375
            else:
                xlabelpadding = 0
            
            pos = [self.template.leftpadding, self.template.bottompadding + xlabelpadding,  1-self.template.leftpadding-self.template.rightpadding , (1-self.template.toppadding)-self.template.bottompadding-xlabelpadding] 
            self.plt.axes.set_position(pos)            
            
            self.legend.arrange()
            self.recessionshading.arrange()
            self.eventlines.arrange()
        
            self.plt.get_xaxis().tick_bottom()  
            self.plt.get_yaxis().tick_left()

class Notes(object): #added by Kevin
    
    def __init__(self,parent):  
        self.parent = parent
        self.name = 'notes'        
        
        self.title = ''
        self.description = ''
        self.unitofmeasurement = ''
        self.datatransformation = ''
        self.interpretation = ''
        self.sourceURL = ''
        self.sourceTitle= ''
        
    def save(self,file_location):
        
        HTMLdesc="<p><strong>Description:</strong> "+self.description+" </p>"
        HTMLuom="<p><strong>Unit of measurement:</strong> "+self.unitofmeasurement+" </p>"
        HTMLtransform="<p><strong>Data transformation:</strong> "+self.datatransformation+"</p>"
        HTMLinterp="<p><strong>Interpretation:</strong> "+self.interpretation+"</p>"
        date = dt.date.today()
        today = date.strftime('%B %d, %Y')
        HTMLsrc="<p><strong>Source:</strong> <a href=\""+self.sourceURL+"\" target=\"_blank\">"+self.sourceTitle+"</a>"+". Data retrieved on "+today+".</p>"
        
        HTMLtext=HTMLdesc+HTMLuom+"\n\n"+HTMLtransform+"\n\n"+HTMLinterp

        if self.sourceURL!=''  or self.sourceTitle!='':  
            HTMLtext+="\n\n"+HTMLsrc
        
        if sys.platform == 'win32':
            output=open('.\\html\\'+self.title+'_mooc_notes.html','w')
        else:
            output=open('./html/'+self.title+'_mooc_notes.html','w')
        
        output.write(HTMLtext) 
        output.close()
#        return file_location+'\\'+self.title+'.html'
#        print("Output file location: "+file_location+self.title+'.html')                       
                        
class Fonttype(object):
    def __init__(self,parent):
    
        self.parent = parent
        self.name = 'Fonttype'
        self.type = fonttypes[0]
        self.usetex = False

    def apply(self):
        
        if not (self.type in fonttypes ):
            
            self.type = fonttypes[0]

        if (self.type == fonttypes[0] ):
            
            mpl.rc('text', usetex = self.usetex )
            mpl.rc('font', **{'family' : "serif"})            
            
        else:
            
            mpl.rc('text', usetex = self.usetex )
            mpl.rc('font', **{'family' : "sans-serif"})
            
class Series(object):
    
    def __init__(self,parent):
        self.parent = parent
        self.name = 'Series'      
        self.data = 0
        self.colors = [ 'b' ]
        self.styles = [ '-' ]
        self.marker = 0
        self.linewidth = 1.5
        self.markersize = 2.5
        
    def hasdata(self):
        # Property to check whether data has been loaded into object
        return ( type(self.data) == pd.DataFrame )
        
    def xisvalues(self):
        
        xisvalues = not(self.xisdates())
    
        return xisvalues

class Titles(object):
    
    def __init__(self,parent):
        self.parent = parent
        self.name = 'Titles'
        
        # These are the strings that determine the text in the different title boxes
        self.title = Titlebox(self,'Title')
        self.subtitle = Titlebox(self,'Subtitle')
        self.source = Titlebox(self,'Source')
        self.xlabel = Axislabel(self,'Xlabel')
        self.ylabel = Axislabel(self,'Ylabel')
        
    def arrange(self):
    
        if self.parent.get_bound():
            
            if self.title.visible:
                self.parent.fig.suptitle(self.title.text, fontsize=self.title.fontsize, fontweight='bold', horizontalalignment='left', x=self.parent.template.leftpadding )
            if self.subtitle.visible:
                self.parent.plt.set_title(self.subtitle.text,loc='left',fontsize=self.subtitle.fontsize)
            if self.source.visible:
                self.source.object = self.parent.fig.text(x=self.parent.template.leftpadding,y=0.025,s=self.source.text,fontsize=self.source.fontsize)

            if self.parent.template.timeseries:
                self.xlabel.visible = False
    
            if self.xlabel.visible:
                self.parent.plt.set_xlabel(self.xlabel.text, fontsize=self.xlabel.fontsize)                
            else:
                self.parent.plt.set_xlabel('', fontsize=0)                

            self.parent.plt.set_ylabel(self.ylabel.text,fontsize=self.ylabel.fontsize)
        
class Titlebox(object):
    
    def __init__(self,parent,name):
        self.parent = parent
        self.name = name
        self.text = ''
        self.fontsize = 12
        self.visible = True
        self.object = 0

class Axislabel(object):
    
    def __init__(self,parent,name):
        
        self.parent = parent
        self.name = name
        self.visible = True
        self.text = name
        self.fontsize = 14
        
class Template(object):
    
    def __init__(self,parent):
        self.parent = parent
        self.name = 'Template'
        self.type = 'slide'
        self.figsize = [10,6]
        self.bottompadding = 0.1
        self.toppadding = 0.1
        self.leftpadding = 0.1
        self.rightpadding = 0.1
        self.xticksize = 14
        self.yticksize = 14        
        self.fontsize = 12
        self.timeseries = True
        self.hzeroline = True
        self.vzeroline = True
        
    def apply(self):
        
        if not (self.type in templatetypes ):
            
            self.type = templatetypes[0]
        
        if ( self.type == templatetypes[0] ):
            
            mpl.rcParams['figure.autolayout'] = False
            mpl.rcParams['xtick.major.pad'] = 5
            mpl.rcParams['ytick.major.pad'] = 3
            mpl.rcParams['lines.linewidth'] = 2            
            
            self.parent.titles.title.visible = True
            self.parent.titles.subtitle.visible = True
            self.parent.titles.source.visible = True           
            
            self.figsize = [10,6]
            self.bottompadding = 0.05
            self.toppadding = 0.12
            self.leftpadding = 0.075
            self.rightpadding = 0.05
            
            self.parent.titles.title.fontsize = 24
            self.parent.titles.subtitle.fontsize = 14
            self.parent.titles.source.fontsize = 10
            self.parent.titles.xlabel.fontsize = 14            
            self.parent.titles.ylabel.fontsize = 14
            self.parent.legend.fontsize = 12
            
            self.parent.titles.title.visible = True
            self.parent.titles.subtitle.visible = True
            self.parent.titles.source.visible = True
            self.parent.titles.xlabel.visible = True            
            self.parent.titles.ylabel.visible = True

            self.xticksize = 14
            self.yticksize = 14
            
            self.parent.eventlines.visible = True
            
        elif ( self.type == templatetypes[1] ):
            
            mpl.rcParams['figure.autolayout'] = False
            mpl.rcParams['xtick.major.pad'] = 5
            mpl.rcParams['ytick.major.pad'] = 3
            mpl.rcParams['lines.linewidth'] = 2            
            
            self.parent.titles.title.visible = True
            self.parent.titles.subtitle.visible = True
            self.parent.titles.source.visible = True           
            
            self.figsize = [13.5,6]
            self.bottompadding = 0.05
            self.toppadding = 0.12
            self.leftpadding = 0.075
            self.rightpadding = 0.05
            
            self.parent.titles.title.fontsize = 24
            self.parent.titles.subtitle.fontsize = 14
            self.parent.titles.source.fontsize = 10
            self.parent.titles.xlabel.fontsize = 14            
            self.parent.titles.ylabel.fontsize = 14
            self.parent.legend.fontsize = 12
            
            self.parent.titles.title.visible = True
            self.parent.titles.subtitle.visible = True
            self.parent.titles.source.visible = True
            self.parent.titles.xlabel.visible = True            
            self.parent.titles.ylabel.visible = True

            self.xticksize = 14
            self.yticksize = 14  
            
            self.parent.eventlines.visible = True
            
        elif ( self.type == templatetypes[2] ):           
            
            mpl.rcParams['figure.autolayout'] = False
            mpl.rcParams['xtick.major.pad'] = 5
            mpl.rcParams['ytick.major.pad'] = 5
            mpl.rcParams['lines.linewidth'] = 1.5
            
            self.parent.titles.title.visible = True
            self.parent.titles.subtitle.visible = True
            self.parent.titles.source.visible = True
            
            self.figsize = [7,3.15]
            self.bottompadding = 0.025
            self.toppadding = 0.025
            self.leftpadding = 0.075
            self.rightpadding = 0.025            
            
            self.parent.titles.title.fontsize = 16
            self.parent.titles.subtitle.fontsize = 12
            self.parent.titles.source.fontsize = 8
            self.parent.titles.xlabel.fontsize = 12            
            self.parent.titles.ylabel.fontsize = 12
            self.parent.legend.fontsize = 12            
            
            self.parent.titles.title.visible = False
            self.parent.titles.subtitle.visible = False
            self.parent.titles.source.visible = False
            self.parent.titles.xlabel.visible = True            
            self.parent.titles.ylabel.visible = True

            self.xticksize = 12
            self.yticksize = 12
            
            self.hzeroline = False
            self.vzeroline = False
            
            self.parent.eventlines.visible = False
            
        else:
        
            mpl.rcParams['figure.autolayout'] = True
            mpl.rcParams['xtick.major.pad'] = 10
            mpl.rcParams['ytick.major.pad'] = 10
            mpl.rcParams['lines.linewidth'] = 1.5
            
            self.parent.titles.title.visible = True
            self.parent.titles.subtitle.visible = True
            self.parent.titles.source.visible = True
            
            self.figsize = [8,6]
            self.bottompadding = 0.025
            self.toppadding = 0.05
            self.leftpadding = 0.1
            self.rightpadding = 0.05            
            
            self.parent.titles.title.fontsize = 24
            self.parent.titles.subtitle.fontsize = 14
            self.parent.titles.source.fontsize = 10
            self.parent.titles.xlabel.fontsize = 12            
            self.parent.titles.ylabel.fontsize = 12
            self.parent.legend.fontsize = 10
            
            self.parent.titles.title.visible = False
            self.parent.titles.subtitle.visible = False
            self.parent.titles.source.visible = False
            self.parent.titles.xlabel.visible = True            
            self.parent.titles.ylabel.visible = True

            self.xticksize = 10
            self.yticksize = 10  
            
            self.parent.eventlines.visible = True
            
    def arrange(self):

        if not (self.type in templatetypes ):
            
            self.type = templatetypes[0]

        if self.parent.get_bound():

            xl = self.parent.plt.get_xticklabels()
            for label in xl:
                label.set_fontsize(self.xticksize)
                label.set_horizontalalignment('center')
    
            yl = self.parent.plt.get_yticklabels()
            for label in yl:
                label.set_fontsize(self.xticksize)
                label.set_horizontalalignment('right')            

        xlims = self.parent.plt.axes.get_xlim()
        ylims = self.parent.plt.axes.get_ylim()

        if self.parent.template.timeseries:
            self.parent.plt.xaxis.grid(False)
            self.parent.plt.yaxis.grid(True,linestyle=':',color='black')
            if self.hzeroline:
                self.parent.plt.axhline(y=0, color='black')
            self.parent.plt.spines["top"].set_visible(False)  
            self.parent.plt.spines["bottom"].set_visible(True)  
            self.parent.plt.spines["right"].set_visible(False)  
            self.parent.plt.spines["left"].set_visible(False)
        else:
            self.parent.plt.xaxis.grid(False)
            self.parent.plt.yaxis.grid(False)
            if self.hzeroline:
                self.parent.plt.axhline(y=0, color='black')
            if self.vzeroline:
                self.parent.plt.axvline(x=0, color='black')
            self.parent.plt.spines["top"].set_visible(False)  
            self.parent.plt.spines["bottom"].set_visible(True)  
            self.parent.plt.spines["right"].set_visible(False)  
            self.parent.plt.spines["left"].set_visible(True)
            self.parent.plt.xaxis.set_ticks_position('bottom')
            self.parent.plt.yaxis.set_ticks_position('left')                
               
        self.parent.plt.axes.set_xlim( [ xlims[0] , xlims[1] ] )
        self.parent.plt.axes.set_ylim( [ ylims[0] , ylims[1] ] )               
               
        if self.parent.titles.xlabel.visible:
            
            self.bottompadding = self.bottompadding+0.05
            
class Eventlines(object):
    
    def __init__(self,parent):
        
        self.parent = parent
        
        self.visible = True
        
        self.linestyle = '--'
        self.linewidth = 1
        self.color = 'k'
        
        self.text = True
        self.padding = 0.05
        self.fontsize = 10 
        
        self.events = None

    def arrange(self):
    
        if (self.events != None) and (self.visible == True) and (self.parent.template.timeseries == True):
            
            ylims = self.parent.plt.axes.get_ylim()
        
            for event in self.events:                         
                
                eventdate = event[0]
                
                try:
                    eventtext = event[1]
                except:
                    eventtext = ''

                if (self.text == True) and (eventtext != ''):
                
                    ylimline =  [ ylims[0] , ylims[0]+(1-self.padding)*(ylims[1]-ylims[0]) ]
                
                else:
                
                    ylimline =  [ ylims[0] , ylims[1] ]


                self.parent.plt.vlines(x=eventdate , ymin=ylimline[0] , ymax=ylimline[1], color=self.color ,linewidth=self.linewidth,linestyle=self.linestyle)
                
                if (self.text == True) and (eventtext != ''):
                    
                    self.parent.plt.text(eventdate, ylims[0]+(1-0.5*self.padding)*(ylims[1]-ylims[0]),eventtext, horizontalalignment='center', verticalalignment='center')

            self.parent.plt.axes.set_ylim(ylims[0], ylims[1])
   
class Export(object):
    
    def __init__(self,parent):
        self.parent = parent
        self.name = 'ChartSave'
        
        if sys.platform == 'win32':
            self.pdfpath = '.\\pdf\\'
            self.svgpath = '.\\svg\\'
            self.pngpath = '.\\png\\'
            self.htmlpath = '.\\html\\'
        else:
            self.pdfpath = './pdf/'
            self.svgpath = './svg/'
            self.pngpath = './png/'
            self.htmlpath = './html/'
            
        self.createdirs()

    def filename(self):

        if self.parent.fonttype.usetex:
            
            usetex = '_usetex'
            
        else:
            
            usetex =''

        filename = self.parent.name.replace(' ','_') + '_' + self.parent.template.type + '_' + self.parent.fonttype.type + usetex

        return filename

    def createdirs(self):
    
        dirs = [ self.pdfpath , self.svgpath , self.pngpath , self.htmlpath ]
        
        for directory in dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def save(self):

        svgfilename = self.svgpath + self.filename() + '.svg'
        pdffilename = self.pdfpath + self.filename() + '.pdf'
        pngfilename = self.pngpath + self.filename() + '.png'
    
        self.parent.fig.savefig(svgfilename, format='svg', dpi=1000, transparent=True)
        self.parent.fig.savefig(pdffilename, format='pdf', dpi=1000, transparent=True)
        self.parent.fig.savefig(pngfilename, format='png', dpi=600, transparent=True)
        
    def posttoplotly(self,uname,apikey):
        
        py.sign_in(uname,apikey)        
        url = py.plot_mpl(self.parent.fig)
        
        return url

class WebToolTips(object):
    
    def __init__(self,parent):
        self.parent = parent
        self.name = 'WebToolTips'
        
    def LinePointToolTips(self):
        
        if self.parent.series.data.index.freq == 'MS':
            datelabels = self.parent.series.data.index.map(lambda x: x.strftime('%Y-%b'))
            datelabels = pd.DataFrame(datelabels,columns=["date"])
        elif self.parent.series.data.index.freq == 'QS':
            datelabels = cp.deepcopy(self.parent.series.data)
            datelabels['year'] = self.parent.series.data.index.year.astype(str)
            datelabels['quarter'] = self.parent.series.data.index.quarter.astype(str)
            datelabels['date'] = datelabels['year'].str.cat(datelabels['quarter'].astype(str),sep="Q")
            datelabels = datelabels[ ['date'] ]
        elif self.parent.series.data.index.freq == 'AS':
            datelabels = self.parent.series.data.index.map(lambda x: x.strftime('%Y'))
            datelabels = pd.DataFrame(datelabels,columns=["date"])
        else:
            datelabels = self.parent.series.data.index.map(lambda x: x.strftime('%Y-%m-%d'))
            datelabels = pd.DataFrame(datelabels,columns=["date"])
        for i, line in enumerate(self.parent.plt.get_lines()):
            labels2 = datelabels["date"].str.cat(self.parent.series.data[line.get_label()].astype(str), sep=': ')
            labels = labels2.values.tolist()
            tooltip = mpld3.plugins.PointLabelTooltip(line, labels=labels)
            mpld3.plugins.connect(self.parent.fig, tooltip)
        
class WebExport(object):
    
    def __init__(self,parent):
        self.parent = parent
        self.name = 'WebChartSave'
        
        if sys.platform == 'win32':
            self.htmlpath = '.\\html\\'
        else:
            self.htmlpath = './html/'

    def filename(self):

        filename = self.parent.name.replace(' ','_') + '_' + self.parent.template.type

        return filename

    def save(self):
        
        self.savefig()
        self.savedata()

    def savefig(self):

        htmlfilename = self.htmlpath + self.filename() + '.html'
        
        try:
            mpld3.save_html(self.parent.fig,htmlfilename)
        except:
            mpld3.save_html(self.parent.fig,htmlfilename)
        
        
    def savedata(self,sort=True):
        
        htmldatafilename = self.htmlpath + self.filename() + '_data.html'
    
        expdata = cp.deepcopy(self.parent.series.data)
        frequency=0
        try:
           frequency=expdata.index.freq
           
        except:
            print('No frequency in savedata()')

        if (frequency == "QS"):
            if(sort):
                expdata = expdata.sort_index(ascending=False)            
            datelabels = cp.deepcopy(expdata)
            datelabels['year'] = expdata.index.year.astype(str)
            datelabels['quarter'] = expdata.index.quarter.astype(str)
            datelabels['date'] = datelabels['year'].str.cat(datelabels['quarter'].astype(str),sep="Q")
            datelabels = datelabels[ ['date'] ]           
            expdata.index = datelabels['date']
        elif  (frequency == "AS"):
            if(sort):
                expdata = expdata.sort_index(ascending=False)            
            datelabels = cp.deepcopy(expdata)
            datelabels['year'] = expdata.index.year.astype(str)
            datelabels = datelabels[ ['year'] ]           
            expdata.index = datelabels['year']
        elif  (frequency == "MS"):
            if(sort):
                expdata = expdata.sort_index(ascending=False)
            datelabels = expdata.index.map(lambda x: x.strftime('%Y-%b'))
            datelabels = pd.DataFrame(datelabels,columns=["month"])          
            expdata.index = datelabels['month']
        else:
            print("None of these frequencies")                        
        
                    
        copyData=cp.deepcopy(expdata)
#        copyData=copyData.drop(copyData.index[copyData.last_valid_index()+1:len(copyData.index)])# +1 added- Kevin (replaced by following line)
        copyData=copyData[copyData.first_valid_index():copyData.last_valid_index()]#Updated by Kevin to save only valid data        
        expdata=expdata[expdata.first_valid_index():expdata.last_valid_index()]        
        copyData.index=list(range(0,len(copyData)))
        copyData.index=expdata.index.values[0:len(copyData)]

        expdata=cp.deepcopy(copyData) 
        expdata.to_html(open(htmldatafilename, 'w'),na_rep='')
        
        
        self.formatTable(htmldatafilename,frequency)#added by Kevin
    
    '''Added by Kevin to format tables into mooc format'''
    def formatTable(self,filelocation,freq):
      
        file=open(filelocation,'r',encoding='utf8')
        text=file.read()
        frequency=freq

        
        if (frequency=="MS"):
            text=text.replace('<th></th>','<th>Month</th>',1)
        elif (frequency=="AS"):
            text=text.replace('<th></th>','<th>Year</th>',1)
        elif (frequency=="QS"):
            text=text.replace('<th></th>','<th>Quarter</th>',1)
        elif (frequency=="D"):
            text=text.replace('<th></th>','<th>Date</th>',1)
        elif (frequency=="W"):
            text=text.replace('<th></th>','<th>Week</th>',1) 
        else:
            print('None of these frequencies: '+str(frequency))

            
            
        text=text.replace('border=\"1\"','border=\"0\"')
        text=text.replace('dataframe','data_table')
        text=text.replace('<tbody>','<tbody tabindex=\"0\">')
        output=open(filelocation,'w')
        output.write(text)
        output.close()
        file.close()
        print('Table formatted for MOOC upload')
             
        
class Recessionshading(object):
    
    def __init__(self,parent):
        
        self.parent = parent
        self.name = 'Recessionshading'
        self.series = 0
        self.visible = True
        
    def arrange(self):
        
        if self.visible and self.parent.template.timeseries and isinstance(self.series,pd.core.frame.DataFrame):
            
            recessionindicator = self.series.ix[:,0]            
            
            lims = self.parent.plt.axes.get_ylim()
            self.parent.plt.fill_between(self.parent.series.data.index,lims[0], lims[1], where=(recessionindicator.values==1), edgecolor='#BBBBBB', facecolor='#BBBBBB', alpha=0.3)
            self.parent.plt.axes.set_ylim( [ lims[0] , lims[1] ] )
        
class Legend(object):

    def __init__(self,parent):

        self.parent = parent
        self.name = 'Legend'
        self.visible = True
        self.location = 0
        self.fontsize = 12 
        self.legend = 0
        self.labels = 0 
        self.positions = ['NE','NW','SW','SE','E','W','E','S','N','C']
        
        '''
        Legend.locations
        |2  9   1|
        |6 10 5,7|
        |3  8   4|
        or
        |NW N NE|
        |W  C  E|
        |SW S SE|
        '''
        
    def arrange(self):
        if not (type(self.location)==int):
            self.location = self.positions.index(self.location)+1
            
        if self.parent.get_bound() and self.visible:        
        
            self.legend = self.parent.plt.legend(loc=self.location,prop={'size': self.fontsize },title='')
            
            if not ( self.labels == 0 ):
                for seriesnumber in range(0,len(self.labels)):
                    self.legend.get_texts()[seriesnumber].set_text(self.labels[seriesnumber])
                    
class Colors(object):

    def __init__(self):

        self.palette = { 'black' : (0 ,0 , 0) ,
                        'gray dark' : (0.1,0.1,0.1) ,
                       'teal dark' : (0 , 0.345 , 0.345 ) ,
                       'teal light' : (0 , 0.686 , 0.686 ) ,
                       'purple dark' : ( 0.286 , 0 , 0.572 ),
                       'blue medium' : ( 0 , 0.427 , 0.859 ),
                       'blue light' : ( 0.427 , 0.714 , 1) ,
                       'brown dark' : ( 0.573 , 0 , 0 ),
                       'brown medium' : ( 0.573 , 0.286 , 0 ),
                       'brown light' : ( 0.859 , 0.820 , 0 ) }
        
        self.lines = { 'solid' : (0, ()) ,
                       'loosely dotted' : (0, (1, 10)) , 
                       'dotted' : ( 0, (1, 5)) ,
                       'densely dotted': (0, (1, 1)) ,
                       'loosely dashed': (0, (5, 10)),
                       'dashed': (0, (5, 5)),
                       'densely dashed': (0, (5, 1)),
                       'loosely dashdotted': (0, (3, 10, 1, 10)),
                       'dashdotted': (0, (3, 5, 1, 5)),
                       'densely dashdotted':  (0, (3, 1, 1, 1)),
                       'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
                       'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
                       'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1)) }
        
        self.source = "http://www.somersault1824.com/tips-for-designing-scientific-figures-for-color-blind-readers/"        
        
    def names(self):
        
        return list(self.palet.keys())
