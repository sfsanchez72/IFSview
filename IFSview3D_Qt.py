#!/usr/bin/env python
""" 
This example illustrates embedding a visvis figure in a Qt application.
"""

from PyQt4 import QtGui, QtCore
import visvis as vv


import matplotlib
matplotlib.use('TkAgg')

from multiprocessing import Process
#import threading
import visvis as vv
#from PyQt4 import QtGui, QtCore

from matplotlib.widgets import Cursor
from tkFileDialog import askopenfilename

from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

from matplotlib.colors import colorConverter
import Tkinter as Tk
import sys

def destroy(e): sys.exit()

import sys
import pyfits


import matplotlib.pyplot as plt
#plt.jet()
import numpy as np
from mpl_toolkits.mplot3d  import Axes3D
import math, copy
from matplotlib import pyplot, colors, cm

global nx_med,ny_med,nz_med,j
global mapFig,filename,ismark,mark
global specFig,isSmark,Smark
global isSpecFig
global click1
global W1,W2,vW1,vW2,Y1,Y2,vY1,vY2,fixY,var_fix
global var_invert
global fitsdata
global listObj,Obj,cObj
global type_spectra
global Wmin,Wmax,Wmin0,Wmax0
global xdata_old,ydata_old
global bright, contrast
global app,root


# Create a visvis app instance, which wraps a qt4 application object.
# This needs to be done *before* instantiating the main window. 
app = vv.use()

class MainWindow(QtGui.QWidget):
    def __init__(self, *args):
        QtGui.QWidget.__init__(self, *args)
        
        # Make a panel with a button
        self.panel = QtGui.QWidget(self)
        but = QtGui.QPushButton(self.panel)
        but.setText('Push me')
        
        # Make figure using "self" as a parent
        self.fig = vv.backends.backend_qt4.Figure(self)
        
        # Make sizer and embed stuff
        self.sizer = QtGui.QHBoxLayout(self)
        self.sizer.addWidget(self.panel, 1)
        self.sizer.addWidget(self.fig._widget, 2)
        
        # Make callback
        but.pressed.connect(self._Plot)
        
        # Apply sizers        
        self.setLayout(self.sizer)
        
        # Finish
        self.resize(560, 420)
        self.setWindowTitle('Embedding in Qt')
        self.show()
    
    
    def _Plot(self):
        
        # Make sure our figure is the active one. 
        # If only one figure, this is not necessary.
        #vv.figure(self.fig.nr)
        
        # Clear it
        vv.clf()
        
        # Plot
        vv.plot([1,2,3,1,6])
        vv.legend(['this is a line'])        
        #self.fig.DrawNow()







#import threading

#class MyTkApp(threading.Thread):
#    def run(self):
#        app = vv.use()
#        app.Create()
#        app.Run()


#now = MyTkApp()
#now.start()

        

root = Tk.Tk()
root.wm_title("Cube Explorer")


#appTk = MyTkApp()
#appTk.init



#class MyThread ( threading.Thread ):
#   def run ( self ):
#       app = vv.use('')
#       app.Run()
#       global app
#       self.app.Run()



bright=0.5
contrast=0.5

fixY=0
type_spectra=1
cObj=0
xdata_old=0
ydata_old=0

maps=[m for m in plt.cm.datad if not m.endswith("_r")]
maps.sort()
l=len(maps)+1

new_cube=1

######## Simulate Bright and Contrast ###########
def cmap_powerlaw_adjust(cmap, a):
    '''
    returns a new colormap based on the one given
    but adjusted via power-law:

    newcmap = oldcmap**a
    '''

    if a < 0.:
        return cmap

    #cdict = cmap #._segmentdata
    cdict = copy.copy(cmap._segmentdata)
    #print cdict['red']
    fn = lambda x : (x[0]**a, x[1], x[2])
    for key in ('red','green','blue'):
        cdict[key] = map(fn, cdict[key])
        cdict[key].sort()
        assert (cdict[key][0]<0 or cdict[key][-1]>1), \
            "Resulting indices extend out of the [0, 1] segment."
    return colors.LinearSegmentedColormap('colormap',cdict,1024)

def cmap_center_adjust(cmap, center_ratio):
    '''
    returns a new colormap based on the one given
    but adjusted so that the old center point higher
    (>0.5) or lower (<0.5)
    '''
    if not (0. < center_ratio) & (center_ratio < 1.):
        return cmap
    a = math.log(center_ratio) / math.log(0.5)
    return cmap_powerlaw_adjust(cmap, a)

def cmap_center_point_adjust(cmap, range, center):
    '''
    converts center to a ratio between 0 and 1 of the
    range given and calls cmap_center_adjust(). returns
    a new adjusted colormap accordingly
    '''
    if not ((range[0] < center) and (center < range[1])):
        return cmap
    return cmap_center_adjust(cmap,
        abs(center - range[0]) / abs(range[1] - range[0]))


def cmap_map(function,cmap):
    """ Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous     points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red','green','blue'):         step_dict[key] = map(lambda x: x[0], cdict[key])
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(map( reduced_cmap, step_list))
    new_LUT = np.array(map( function, old_LUT))
    
#    new_LUT=new_LUT_tmp[-1::]
   # new_LUT=np.reverse(new_LUT)
#    print new_LUT.shape()
#    new_LUT=new_LUT[-1:-1:-1]
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i,key in enumerate(('red','green','blue')):
        this_cdict = {}
        for j,step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j,i]
            elif new_LUT[j,i]!=old_LUT[j,i]:
                this_cdict[step] = new_LUT[j,i]
        colorvector=  map(lambda x: x + (x[1], ), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)



######### Create Colours ######################
def pastel(colour, weight=2.4):
    """ Convert colour into a nice pastel shade"""
    rgb = np.asarray(colorConverter.to_rgb(colour))
    # scale colour
    maxc = max(rgb)
    if maxc < 1.0 and maxc > 0:
        # scale colour
        scale = 1.0 / maxc
        rgb = rgb * scale
    # now decrease saturation
    total = rgb.sum()
    slack = 0
    for x in rgb:
        slack += 1.0 - x

    # want to increase weight from total to weight
    # pick x s.t.  slack * x == weight - total
    # x = (weight - total) / slack
    x = (weight - total) / slack

    rgb = [0.6*(c + (x * (1.0-c))) for c in rgb]
    return rgb

def get_colours(n):
    """ Return n pastel colours. """
    base = np.asarray([[1,0,0], [0,1,0], [0,0,1]])

    if n <= 3:
        return base[0:n]

    # how many new colours to we need to insert between
    # red and green and between green and blue?
    needed = (((n - 3) + 1) / 2, (n - 3) / 2)

    colours = []
    for start in (0, 1):
        for x in np.linspace(0, 1, needed[start]+2):
            colours.append((base[start] * (1.0 - x)) +
                           (base[start+1] * x))
            
    return [pastel(c) for c in colours[0:n]]



###############################################
def test():
    print 'Test'

######
def LoadFile():
    global j,nx_med,ny_med,nz_med,mask,click1,Obj,listObj
    global filename,fitsdata,fitscube,fitshdr
    global Wmin,Wmax,Wmin0,Wmax0
    filename = askopenfilename(filetypes=[("allfiles","*"),("pythonfiles","*.py")])
    fitscube,fitshdr=rfits_cube(filename)
#    j=47
    j=58
    click1=0
    nx_med=int(nx/2)
    ny_med=int(ny/2)
    nz_med=int(nz/2)
    fitsdata=fitscube[nz_med,:,:]
    out=np.zeros((ny,nx))
    infinite=np.isfinite(fitsdata,out)
    fitsdata=fitsdata*out
    fitsdata=np.nan_to_num(fitsdata)
    mask=np.zeros((ny,nx))
    for ii in range(0,nx-1):
        for jj in range(0,ny-1):
            val=fitsdata[jj,ii]
            if (abs(val)<1e308):
                mask[jj,ii]=1
    isSmark=0
    ismark=0
    Obj=np.zeros((ny,nx))
    listObj = []
    listObj.append(Obj)
    animate(j,l)
    plot_spec(nx_med,ny_med)


######

###### Create Menubar
def makeMenuBar(self,frame):
    menubar = Tk.Frame(frame,relief='raised',borderwidth=1)
    menubar.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)
		
    mb_file = Tk.Menubutton(menubar,text='File')
    mb_file.pack(side=Tk.LEFT)
    mb_file.menu = Tk.Menu(mb_file)
    
    mb_file.menu.add_command(label='open',command=LoadFile)
    mb_file.menu.add_command(label='close')
    
#    mb_edit = Tk.Menubutton(menubar,text='edit')
#   mb_edit.pack(side=Tk.LEFT)
#    mb_edit.menu = Tk.Menu(mb_edit)
#    mb_edit.menu.add_command(label='copy')
#    mb_edit.menu.add_command(label='paste')
    
    mb_help = Tk.Menubutton(menubar,text='help')
    mb_help.pack(padx=25,side=Tk.LEFT)
    
    button1 = Tk.Button(menubar, text='Quit', command=sys.exit)
    button1.pack(side=Tk.RIGHT)

    mb_file['menu'] = mb_file.menu
    #mb_edit['menu'] = mb_edit.menu
    return menubar

###### Delete Menubar


###### Create Menubar
def makeOptFrame(self,frame):
    menubar = Tk.Frame(frame,relief='raised',borderwidth=1)
    menubar.pack(side=Tk.TOP, fill=Tk.Y, expand=1)
		
#    mb_file = Tk.Menubutton(menubar,text='file')
#    mb_file.pack(side=Tk.LEFT)

#    label = Tk.Label(menubar, text='Color Map', width=20)
#    label.pack(side=Tk.TOP)
    button1 = Tk.Button(menubar, text='Quit', command=sys.exit)
    button1.pack(side=Tk.BOTTOM)

    return menubar

### START rfits_img
def rfits_img(filename):
    global nx,ny
    # READ FITS FILE
    fitsdata=pyfits.getdata(filename);
    fitshdr=pyfits.getheader(filename);
    nx = fitshdr['NAXIS1']
    ny = fitshdr['NAXIS2']
    out=np.zeros((ny,nx))
    infinite=np.isfinite(fitsdata,out)
    fitsdata=fitsdata*out
    fitsdata=np.nan_to_num(fitsdata)
    return fitsdata,fitshdr
### END rfits_img

### START rfits_img
def rfits_cube(filename):
    global nx,ny,nz,crval,cdelt,crpix
    global Wmin,Wmax,Wmin0,Wmax0
    global new_cube
    new_cube=1
    # READ FITS FILE
    print "Reading cube ",filename
    fitscube=pyfits.getdata(filename);
    fitshdr=pyfits.getheader(filename);
    nx = fitshdr['NAXIS1']
    ny = fitshdr['NAXIS2']
    nz = fitshdr['NAXIS3']
    crval = fitshdr['CRVAL3']
    cdelt = fitshdr['CDELT3']
    crpix = 1.0
    #fitshdr['CRPIX3']
    out=np.zeros((nz,ny,nx))    
    infinite=np.isfinite(fitscube,out)
    fitscube=fitscube*out
    fitscube=np.nan_to_num(fitscube)
    Wmin=crval
    Wmax=crval+nz*cdelt
    Wmin0=Wmin
    Wmax0=Wmax
    print "done"
    return fitscube,fitshdr
### END rfits_img

### Create Fake cube
def create_cube(NX,NY,NZ,CRVAL,CDELT,CRPIX):
    global nx,ny,nz,crval,cdelt,crpix    
    global Wmin,Wmax,Wmin0,Wmax0
    global new_cube
    new_cube=1
    print "Creating a fake cube..."
    fitscube=np.ones((NZ,NY,NX))
#    fitshdr=pyfits.getheader(filename);
    nx = NX
    ny = NY
    nz = NZ
    crval = CRVAL
    cdelt = CDELT
    crpix = CRPIX
    fitshdr = {'NAXIS':3}
#    fitshdr['NAXIS']=3
    fitshdr['NAXIS1']=NX
    fitshdr['NAXIS2']=NY
    fitshdr['NAXIS3']=NZ
    fitshdr['CRPIX1']=1
    fitshdr['CDELT1']=1
    fitshdr['CRVAL1']=1
    fitshdr['CRPIX2']=1
    fitshdr['CDELT2']=1
    fitshdr['CRVAL2']=1
    fitshdr['CRPIX3']=CRPIX
    fitshdr['CDELT3']=CDELT
    fitshdr['CRVAL3']=CRVAL
    Wmin=CRVAL
    Wmax=CRVAL+NZ*CDELT
    Wmin0=Wmin
    Wmax0=Wmax
    return fitscube,fitshdr

#print sys.argv

def plot_img(fig,ax,fitsdata,alpha,m):
    fitsdata=fitsdata*mask
    cax = ax.imshow(fitsdata,cmap=plt.get_cmap(m),alpha=alpha,interpolation='nearest')    
    plt.hold=True
    return




def animate(k,l):    
    global mapFig
    global specFig,isSmark,Smark,isSpecFig,Obj
    global Y1,Y2,fixY,var_fix
    global nx_med,ny_med
    global bright, contrast
    global var_invert
    global caxMap,new_cube
    invert=var_invert.get()

    fixY=var_fix.get()
    m=maps[k]
    cmap_name=plt.get_cmap(m)    
    my_str=cmap_name.name
    #print my_str
    if (invert==1):
        my_str=my_str+'_r'
    cmap=cm.get_cmap(my_str)
#    cmap=cm.get_cmap(cmap)
#    print cmap
#    cmap=cm.get_cmap(cmap_name.name)
    mod_cmap=cmap_center_adjust(cmap, contrast)

    if (new_cube==1):
        fig1.clear()
        fig1.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.99)
        mapFig = fig1.add_subplot(1,1,1)
        mapFig.clear()
        if (fixY==0):   
            caxMap = mapFig.imshow(fitsdata,cmap=mod_cmap,interpolation='nearest')    
            Y1=caxMap.norm.vmin
            Y2=caxMap.norm.vmax

        else:
            caxMap = mapFig.imshow(fitsdata,cmap=mod_cmap,interpolation='nearest',vmin=Y1,vmax=Y2)    
        fig1.colorbar(caxMap, orientation='vertical', shrink=0.925)
    
    
    caxMap.set_clim([Y1*(bright-0.5),Y2/(0.5+bright)])
    caxMap.set_cmap(mod_cmap) 

    if (new_cube!=1):
        mapFig.get_figure().canvas.draw()

    mapFig.hold=True
    levels = arange(0,1,2)
    if (new_cube==1):
        mapFig.contour(Obj, levels, hold='on', colors = 'k')
    k=0
    XX=[]
    YY=[]
    for ii in range(0,nx-1):
        for jj in range(0,ny-1):
            val=Obj[jj,ii]
            if (val>0):
                XX.append(ii)
                YY.append(jj)
    nsum=int(Obj.sum())
    if (nsum>0):
        mapFig.scatter(XX,YY,alpha=0.5,s=3,color='black')
    mapFig.set_xlim(0,nx-1)
    mapFig.set_ylim(ny-1,0)
    mapFig.get_figure().canvas.draw()
    XX=np.zeros(2)
    YY=np.zeros(2)
    XX[0]=crval+cdelt*nz_med
    XX[1]=crval+cdelt*nz_med
    ymin,ymax=specFig.get_ylim()
    xmin,xmax=specFig.get_xlim()
    YY[0]=ymin
    YY[1]=ymax
    new_cube=0
    
#    plot_spec(nx_med,ny_med)
#    specFig.plot(XX,YY,"o-",color='orange',lw=2,ms=3,alpha=0.4)
#    specFig.set_xlim(xmin,xmax)
#    specFig.set_ylim(ymin,ymax)
#    specFig.get_figure().canvas.draw()


global ii_last,jj_last
ii_last=0
jj_last=0
global count
count=0
def plot_spec(ii,jj):    
    global fitscube,mapFig,ismark,mark
    global specFig,isSpecFig
    global Y1,Y2,fixY,var_fix,count
    global ii_last,jj_last
    global Wmin,Wmax
#    if ((ii_last==ii) and (jj_last==jj)):
#        return

    fixY=var_fix.get()
#    fig2.subplots_adjust(top=0.95,bottom=0.1,left=0.05,right=0.99)
#    specFig = fig2.add_subplot(1,1,1)


#    onClick = specFig.get_figure().canvas.mpl_connect('button_press_event',click)
#    onMove = specFig.get_figure().canvas.mpl_connect('motion_notify_event',move)


    specFig.clear()    
    count=count+1
    s=fitscube[:,jj,ii]
    sout=np.zeros(nz)
    w=np.zeros(nz)
    infinite=np.isfinite(s,sout)
    s=s*sout
    s=np.nan_to_num(s)   
    for iii in range(0,nz):
        val=s[iii]
        w[iii]=crval+cdelt*iii
        if (abs(val)>1e30):
            s[iii]=0
    specFig.plot(w,s,color='black',lw=2.1)
    if (fixY==1):
        specFig.set_ylim(Y1,Y2)
    specFig.set_xlim(Wmin,Wmax)
#    specFig.grid()
    specFig.get_figure().canvas.draw()
    XX=np.zeros(1)
    YY=np.zeros(1)
    XX[0]=ii*1.0;
    YY[0]=jj*1.0;
    ismark=ismark+1
    isSpecFig=isSpecFig+1
    ii_last=ii
    jj_last=jj
#    return

def plot_spectra(obj_now,Type,kcolor):    
    global fitscube,mapFig,ismark,mark
    global specFig,isSpecFig
    global Y1,Y2,fixY,var_fix
    global Wmin,Wmax
    global bright,contrast
    global var_invert,j
    invert=var_invert.get()
    
#    nsum=int(obj_now.sum())
    nsum=0
    for ii in range(0,nx-1):
        for jj in range(0,ny-1):
            val=obj_now[jj,ii]
            if (abs(val)>0):
                nsum=nsum+1

    if (nsum==0):
        return

    fig2.subplots_adjust(top=0.95,bottom=0.1,left=0.05,right=0.99)
    specFig = fig2.add_subplot(1,1,1)
    specFig.clear()    
    colours=get_colours(nsum)
    spectra=np.zeros((nsum,nz))
    yp=np.zeros(nsum)
    for i in range(0,nsum-1):
        yp[i]=i    
    m=maps[kcolor]
            
    wmap=np.zeros((nsum,nz))
    ypmap=np.zeros((nsum,nz))
    
    k=0
    w=np.zeros(nz)
    spec_sum=np.zeros(nz)

    fixY=var_fix.get()
    for ii in range(0,nx-1):
        for jj in range(0,ny-1):
            val=obj_now[jj,ii]
            if (abs(val)>0):
                s=fitscube[:,jj,ii]
                sout=np.zeros(nz)
                infinite=np.isfinite(s,sout)
                s=s*sout
                s=np.nan_to_num(s)   
                for iii in range(0,nz):
                    val=s[iii]
                    w[iii]=crval+cdelt*iii
                    wmap[k,iii]=w[iii]
                    ypmap[k,iii]=k
                    if (abs(val)>1e30):
                        s[iii]=0
                    kk=val-k
                    spectra[k,iii]=s[iii]                    
#                    print k,iii,spectra[k,iii];
                if (Type==2):
                    specFig.plot(w,s,color=colours[k],lw=1)
                    Z=k/nsum
                    if (fixY==1):
                        specFig.set_ylim(Y1,Y2)
                    specFig.set_xlim(Wmin,Wmax)
                plt.hold=True
                k=k+1

    if (Type==0):
        if (k>0):
            for iii in range(0,nz):
                spec_sum[iii]=0
                for kk in range(0,k):
                    spec_sum[iii]=spec_sum[iii]+spectra[kk,iii]/k        
        specFig.plot(w,spec_sum,color='Black',lw=2.1)
        if (fixY==1):
            specFig.set_ylim(Y1,Y2)
        specFig.set_xlim(Wmin,Wmax)
    plt.hold=True


                
#    print spectra
    if (Type==1):
        extent=np.zeros(4)
        extent[0]=Wmin
        extent[1]=Wmax
        extent[2]=0
        extent[3]=nsum-1

        i0=int((Wmin-crval)/cdelt)
        i1=int((Wmax-crval)/cdelt)

        m=maps[j]
        cmap_name=plt.get_cmap(m)    
        my_str=cmap_name.name
        if (invert==1):
            my_str=my_str+'_r'
        cmap=cm.get_cmap(my_str)
        mod_cmap=cmap_center_adjust(cmap, contrast)
        print 'i0='+str(i0)+',i1='+str(i1)
        map_now=spectra[:,i0:i1]
        mean=map_now.mean()
        levels=arange(0.01,3*mean,0.01*mean)


        if (fixY==0):  
           caxMap=specFig.imshow(map_now,interpolation='nearest',aspect='auto',extent=extent,vmin=-0.1*mean,vmax=3*mean,cmap=mod_cmap)    
        else:
           caxMap=specFig.imshow(map_now,interpolation='nearest',aspect='auto',extent=extent,cmap=mod_cmap,vmin=Y1,vmax=Y2)

    caxMap.set_clim([Y1*(bright-0.5),Y2/(0.5+bright)])
    caxMap.set_cmap(mod_cmap) 


    specFig.get_figure().canvas.draw()
    ismark=0



#    specFig.imshow(spectra,cmap=plt.get_cmap(m),alpha=1)    


#    print 'isSpecFig=',isSpecFig
# Press a Key
def key(event):
    print "pressed", repr(event.char)," at ", event.xdata, event.ydata

# Click Button
def click(event):
    global specFig,ismark,isSmark,click1,nx_med,ny_med,W1,W2,vW1,vW2
    global cdelt,Smark
    global nx_med,ny_med,count

#    nx_med_now=int(event.xdata);
#    ny_med_now=int(event.ydata);
#    print "PASO ",nx_med,nx_med_now,ny_med,ny_med_now
#    if ((nx_med_now==nx_med) and (ny_med_now==ny_med)):
#        return

    if ((event.key=='m') or (event.button==2)):
        if ((event.xdata>0) and (event.xdata<nx) and (event.ydata>0) and (event.ydata<ny)):
            nx_med=int(event.xdata);
            ny_med=int(event.ydata);
            isSmark=0
            plot_spec(nx_med,ny_med)
        else:
            if ((event.xdata>crval) and (event.xdata<(crval+cdelt*nz))):
                ismark=0
                W1=event.xdata-cdelt
                W2=event.xdata+cdelt
                plot_3d_nofill(W1,W2,j)
                Slice(event.xdata)

    if (event.button==3):
        if (not ((event.xdata>0) and (event.xdata<nx) and (event.ydata>0) and (event.ydata<ny))):
            if ((event.xdata>crval) and (event.xdata<(crval+cdelt*nz))):
                ismark=0
                if (W1>W2):
                    W=W2
                    W2=W1
                    W1=W
                vW1.set(W1)
                vW2.set(W2)                
                Slice_Range(W1,W2)
                plot_3d_nofill(W1,W2,j)

    if (event.button==3):
        if ((event.xdata>0) and (event.xdata<nx) and (event.ydata>0) and (event.ydata<ny)):
#            nx_med=int(event.xdata);
#            ny_med=int(event.ydata);
            isSmark=0
            plot_spectra(Obj,type_spectra,j)

    if (event.button==1):
        if ((event.xdata>0) and (event.xdata<nx) and (event.ydata>0) and (event.ydata<ny)):
            nx_med=int(event.xdata);
            ny_med=int(event.ydata);
            isSmark=1
            plot_spec(nx_med,ny_med)
        else:
            if ((event.xdata>crval) and (event.xdata<(crval+cdelt*nz))):
                 if (click1==2):
                    click1=0
                    plot_spec(nx_med,ny_med)
                 if (click1==0):
                    W1=event.xdata 
                    vW1.set(W1)
                    XX=np.zeros(2)
                    YY=np.zeros(2)
                    XX[0]=event.xdata
                    XX[1]=event.xdata
                    ymin,ymax=specFig.get_ylim()
                    xmin,xmax=specFig.get_xlim()
                    YY[0]=ymin
                    YY[1]=ymax                                
                    specFig.plot(XX,YY,"o-",color='blue',lw=2,ms=3,alpha=0.4)
                    specFig.set_xlim(xmin,xmax)
                    specFig.set_ylim(ymin,ymax)
                    specFig.get_figure().canvas.draw()                    
                 if (click1==1):
                    W2=event.xdata
                    vW2.set(W2)
                    plot_spec(nx_med,ny_med)
                    specFig.axvspan(W1,W2,color='blue',alpha=0.4)
                    ymin,ymax=specFig.get_ylim()
                    xmin,xmax=specFig.get_xlim()
                    specFig.set_xlim(xmin,xmax)
                    specFig.set_ylim(ymin,ymax)
                    specFig.get_figure().canvas.draw()                    

                 click1=click1+1
    
#                print '[',W1,',',W2,']'
#                Slice(event.xdata)

# Movesmod while clicking
def move(event):
    global mapFig,ismark,isSmark,Obj,mask
    global W1,W2,vW1,vW2
    global nx_med,ny_med
    global cObj
    global xdata_old,ydata_old
    

#    if ((xdata_old==xnow) and (ydata_old==ynow)):
#        return
#    x_now=toolbar.x()
#    print 'x= ',x_now
#    print 'x= ',event.xdata,' y= ',event.ydata,' key= ',event.key
    if ((event.key=='a') or (event.button==2)):
        if ((event.xdata>0) and (event.xdata<nx) and (event.ydata>0) and (event.ydata<ny)):
            xnow=event.xdata
            ynow=event.ydata
            if ((xnow!=xdata_old) or (ynow!=ydata_old)):
                nx_med=int(event.xdata)
                ny_med=int(event.ydata)
                isSmark=0
                plot_spec(nx_med,ny_med)
                xdata_old=xnow
                ydata_old=ynow
        else:
            if ((event.xdata>crval) and (event.xdata<(crval+cdelt*nz))):
                ismark=0
                xnow=event.xdata
                W1=xnow-0.5*cdelt
                W2=xnow+0.5*cdelt
                if ((xdata_old!=xnow)):
                    vW1.set(W1)
                    vW2.set(W2)
                    Slice(xnow)
                    xdata_old=xnow
                    plot_3d_nofill(W1,W2,j)



    if ((event.xdata>0) and (event.xdata<nx) and (event.ydata>0) and (event.ydata<ny)):
        if ((event.key=='s') or (event.button==1)):
            xnow=event.xdata
            ynow=event.ydata
            if ((xnow!=xdata_old) or (ynow!=ydata_old)):
                nx_med=int(event.xdata)
                ny_med=int(event.ydata)
                cObj=cObj+1
                Obj[ny_med,nx_med]=cObj
                Obj=Obj*mask
                animate(j,l)
                xdata_old=xnow
                ydata_old=ynow

        if ((event.key=='d')):
            xnow=event.xdata
            ynow=event.ydata
            if ((xnow!=xdata_old) or (ynow!=ydata_old)):
                nx_med=int(event.xdata)
                ny_med=int(event.ydata)
                Obj[ny_med,nx_med]=0
                Obj=Obj*mask
                xdata_old=xnow
                ydata_old=ynow
                animate(j,l)
#    xdata_old=xnow
#    ydata_old=ynow
                

def enter_figure(event):
    print 'enter_figure', event.canvas.figure
    event.canvas.figure.patch.set_facecolor('red')
    event.canvas.draw()

def leave_figure(event):
    print 'leave_figure', event.canvas.figure
    event.canvas.figure.patch.set_facecolor('grey')
    event.canvas.draw()

def next():
    global j
    if (j<l-2):
        animate(j+1,l)
    else:
        animate(0,l)
        j=0
    j=j+1

def previous():
    global j
    if (j>1):
        animate(j-1,l)
    else:
        animate(0,l)
        j=0
    j=j-1

def nSlice():
    global j,nz_med,fitsdata,ismark
    global new_cube
    new_cube=1
    ismark=0
    if (nz_med<nz-12):
        nz_med=nz_med+10
        fitsdata=fitscube[nz_med,:,:]
        out=np.zeros((ny,nx))
        infinite=np.isfinite(fitsdata,out)
        fitsdata=fitsdata*out
        fitsdata=np.nan_to_num(fitsdata)
        wave=crval+cdelt*nz_med
#        new_cube=1
#        print wave
        animate(j,l)

def pSlice():
    global j,nz_med,fitsdata,ismark
    global new_cube
    new_cube=1

    ismark=0
    if (nz_med>10):
        nz_med=nz_med-10
        fitsdata=fitscube[nz_med,:,:]
        out=np.zeros((ny,nx))
        infinite=np.isfinite(fitsdata,out)
        fitsdata=fitsdata*out
        fitsdata=np.nan_to_num(fitsdata)
        wave=crval+cdelt*nz_med
#        print wave
        animate(j,l)

def Slice(wave_now):
    global j,nz_med,fitsdata
    global new_cube
    new_cube=1

    i_med=int((wave_now-crval)/cdelt)
    if ((i_med>0) and (i_med<nz)):
        nz_med=i_med
        fitsdata=fitscube[nz_med,:,:]
        out=np.zeros((ny,nx))
        infinite=np.isfinite(fitsdata,out)
        fitsdata=fitsdata*out
        fitsdata=np.nan_to_num(fitsdata)
        wave=crval+cdelt*nz_med
#        print wave
        animate(j,l)

def Slice_Range(wave_1,wave_2):
    global j,nz_med,fitsdata
    global new_cube
    new_cube=1

    wave_now=0.5*(wave_2+wave_1)
    i_med=int((wave_now-crval)/cdelt)
    i1=int((wave_1-crval)/cdelt)
    i2=int((wave_2-crval)/cdelt)
    if (i1>i2):
        i=i2
        i2=i1
        i1=i

    if ((i1>0) and (i2<nz)):
        nz_med=i_med
        tmpdata=fitscube[i1:i2,:,:]
        fitsdata=np.apply_along_axis(sum,0,tmpdata)
#        print fitscube.shape
#        print fitsdata.shape
#        out=np.zeros((ny,nx))
#        infinite=np.isfinite(fitsdata,out)
#        fitsdata=fitsdata*out
        if (i2!=i1):
            fitsdata=fitsdata/(i2-i1)
        fitsdata=np.nan_to_num(fitsdata)
        wave=crval+cdelt*nz_med
#        print wave
        animate(j,l)

def movie():
    global j,nz_med,fitsdata
    global new_cube
    new_cube=1

    for i in range(0,nz-1):        
        fitsdata=fitscube[i,:,:]
        out=np.zeros((ny,nx))
        infinite=np.isfinite(fitsdata,out)
        fitsdata=fitsdata*out
        fitsdata=np.nan_to_num(fitsdata)
        wave=crval+cdelt*i
#        print wave
        animate(j,l)
    fitsdata=fitscube[nz_med,:,:]
    out=np.zeros((ny,nx))
    infinite=np.isfinite(fitsdata,out)
    fitsdata=fitsdata*out
    fitsdata=np.nan_to_num(fitsdata)
    animate(j,l)

def specX():
    global nx_med,ny_med    
    for i in range(0,nx-1):        
        isSmark=0
        plot_spec(i,ny_med)
    plot_spec(nx_med,ny_med)

def specY():
    global nx_med,ny_med    
    for i in range(0,ny-1):
        isSmark=0        
        plot_spec(nx_med,i)
    plot_spec(nx_med,ny_med)
def first():
    global j,ismark
    ismark=0
    j=0
    animate(0,l)

def last():
    global j,ismark
    ismark=0
    j=j-2
    animate(l-2,l)

def change_color(kk):
    global j,ismark
    ismark=0
    j=kk
    animate(kk,l)

#frame.pack()





#gui3d = Tk.Tk()
#gui3d.wm_title("Cube 3D")
fig1b = plt.figure(figsize=(6,6))
#TopFrame3D=Tk.Frame(gui3d)
#TopFrame3D.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
#BottomFrame3D=Tk.Frame(gui3d)
#BottomFrame3D.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
#OptFrame3D=Tk.Frame(gui3d,width=200)
#OptFrame3D=Tk.Frame(TopFrame3D,width=150)
#OptFrame3D.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)
#plotFrame3D=Tk.Frame(gui3d)
#plotFrame3D=Tk.Frame(TopFrame3D)
#plotFrame3D.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)
#canvas2 = FigureCanvasTkAgg(fig1b, master=gui3d)
#canvas2 = FigureCanvasTkAgg(fig1b, master=plotFrame3D)


TopFrame=Tk.Frame(root)
MenuBar=makeMenuBar(root,TopFrame)
TopFrame.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)



OptFrame=Tk.Frame(width=200)
OptMenu=makeOptFrame(root,OptFrame)
OptFrame.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)

MidFrame=Tk.Frame(root, relief='raised',borderwidth=2)
MidFrame.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

plotFrame1 = Tk.Frame(root,relief='raised',borderwidth=1)
plotFrame1.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
plotFrame11 = Tk.Frame(plotFrame1,relief='raised',borderwidth=1,width=5,height=5)
plotFrame11.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)
plotFrame12 = Tk.Frame(plotFrame1,relief='raised',borderwidth=1,width=5,height=5)
plotFrame12.pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)

plotFrame2 = Tk.Frame(root,relief='raised',borderwidth=1)
plotFrame2.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

#plotFrame3 = Tk.Frame(root,relief='raised',borderwidth=1)
#plotFrame3.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

#root.bind("<Destroy>", destroy)


# READ ARGUMENTS
nargs=len(sys.argv)
print nargs
if (nargs==2):
    filename=sys.argv[1]
    fitscube,fitshdr=rfits_cube(filename)
else:
    filename='NONE'
    fitscube,fitshdr=create_cube(50,50,50,3800,1,1)

ismark=0
isSmark=0
isSpecFig=0
#j=47
j=58
click1=0
nx_med=int(nx/2)
ny_med=int(ny/2)
nz_med=int(nz/2)
W1=crval+cdelt*(nz_med-1)
W2=crval+cdelt*(nz_med+1)

fitsdata=fitscube[nz_med,:,:]
out=np.zeros((ny,nx))
infinite=np.isfinite(fitsdata,out)
fitsdata=fitsdata*out
fitsdata=np.nan_to_num(fitsdata)
mask=np.zeros((ny,nx))
Obj=np.zeros((ny,nx))
listObj = []
listObj.append(Obj)
for ii in range(0,nx-1):
    for jj in range(0,ny-1):
        val=fitsdata[jj,ii]
        if (abs(val)<1e308):
            mask[jj,ii]=1

var_fix=Tk.IntVar()
var_invert=Tk.IntVar()



fig1 = plt.figure(figsize=(6,5))
fig1.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.99)

#canvas = plt.FigureCanvas(self,-1,fig1b)

#fig1b.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.99)
fig2 = plt.figure(figsize=(11,5))
fig2.subplots_adjust(top=0.95,bottom=0.1,left=0.1,right=0.99)


#fig1 = plt.figure(figsize=(8,5))
#fig1.subplots_adjust(top=0.95,bottom=0.05,left=0.01,right=0.99)

#j=12
#fig2.subplots_adjust(top=0.95,bottom=0.1,left=0.05,right=0.99)
specFig = fig2.add_subplot(1,1,1)


animate(j,l)
plot_spec(nx_med,ny_med)
#onClick = specFig.get_figure().canvas.mpl_connect('button_press_event',click)
#onMove = specFig.get_figure().canvas.mpl_connect('motion_notify_event',move)


#cursor = Tk.Cursor(mapFig, useblit=True, color='black', linewidth=2)

rangeFrame = Tk.Frame(OptMenu,relief='raised',borderwidth=1)
rangeFrame.pack(side=Tk.TOP, expand=1)
labelR = Tk.Label(rangeFrame, text='Plotting Range', width=15)
labelR.pack(side=Tk.TOP)

labW1=Tk.Label(rangeFrame,text="W1=",width=3,anchor='e')
labW1.pack(side=Tk.LEFT)
vW1=Tk.StringVar()
eb1=Tk.Entry(rangeFrame,textvariable=vW1,width=6)
eb1.pack(side=Tk.LEFT)
vW1.set(W1)
labW2=Tk.Label(rangeFrame,text="W2=",width=3,anchor='s')
labW2.pack(side=Tk.LEFT)
vW2=Tk.StringVar()
eb2=Tk.Entry(rangeFrame,textvariable=vW2,width=6)
eb2.pack(side=Tk.LEFT)
vW2.set(W2)
def setW():
    global mapFig,ismark,isSmark,Obj,mask
    global W1,W2,eb1,eb2
    W1=float(Tk.Entry.get(eb1))
    W2=float(Tk.Entry.get(eb2))
    if (W1>W2):
        W=W2
        W2=W1
        W1=W
        vW1.set(W1)
        vW2.set(W2)
    specFig.axvspan(W1,W2,color='blue',alpha=0.4)
    ymin,ymax=specFig.get_ylim()
    xmin,xmax=specFig.get_xlim()
    specFig.set_xlim(xmin,xmax)
    specFig.set_ylim(ymin,ymax)
    specFig.get_figure().canvas.draw()  
buttonW = Tk.Button(rangeFrame, text='SET', command=setW)
buttonW.pack(side=Tk.RIGHT)

def zoomW():
    global Wmin,Wmax
    global Wmin0,Wmax0
    global mapFig,ismark,isSmark,Obj,mask
    global W1,W2,eb1,eb2,Y1,Y2,fixY
    W1=float(Tk.Entry.get(eb1))
    W2=float(Tk.Entry.get(eb2))
    if (W1>W2):
        W=W2
        W2=W1
        W1=W    
    Wmin=W1
    Wmax=W2
    if (fixY==1):
        specFig.set_ylim(Y1,Y2)
    specFig.set_xlim(Wmin,Wmax)
    specFig.get_figure().canvas.draw()  

buttonZW = Tk.Button(rangeFrame, text='Zoom', command=zoomW)
buttonZW.pack(side=Tk.RIGHT)
def UzoomW():
    global Wmin,Wmax
    global Wmin0,Wmax0
    global W1,W2,Y1,Y2,fixY
    Wmin=Wmin0
    Wmax=Wmax0
    if (fixY==1):
        specFig.set_ylim(Y1,Y2)
    specFig.set_xlim(Wmin,Wmax)
    specFig.get_figure().canvas.draw()  


buttonUW = Tk.Button(rangeFrame, text='Release', command=UzoomW)
buttonUW.pack(side=Tk.RIGHT)



YrangeFrame = Tk.Frame(OptMenu,relief='raised',borderwidth=1)
YrangeFrame.pack(side=Tk.TOP, expand=1)
labelW = Tk.Label(YrangeFrame, text='Plotting Yrange', width=12)
labelW.pack(side=Tk.TOP)
YlabW1=Tk.Label(YrangeFrame,text="Y1=",width=3,anchor='e')
YlabW1.pack(side=Tk.LEFT)
vY1=Tk.StringVar()
yeb1=Tk.Entry(YrangeFrame,textvariable=vY1,width=6)
yeb1.pack(side=Tk.LEFT)
vY1.set(Y1)
YlabW2=Tk.Label(YrangeFrame,text="Y2=",width=3,anchor='s')
YlabW2.pack(side=Tk.LEFT)
vY2=Tk.StringVar()
yeb2=Tk.Entry(YrangeFrame,textvariable=vY2,width=6)
yeb2.pack(side=Tk.LEFT)
vY2.set(Y2)
def setY():
    global mapFig,ismark,isSmark,Obj,mask
    global Y1,Y2,yeb1,yeb2
    global j,l,nx_med,ny_med
    Y1=float(Tk.Entry.get(yeb1))
    Y2=float(Tk.Entry.get(yeb2))
    if (Y1>Y2):
        W=Y2
        Y2=Y1
        Y1=W
    vY1.set(Y1)
    vY2.set(Y2)
    animate(j,l)
    plot_spec(nx_med,ny_med)

buttonY = Tk.Button(YrangeFrame, text='SET', command=setY)
buttonY.pack(side=Tk.RIGHT)
CB_setY = Tk.Checkbutton(YrangeFrame, text="Fix Y", variable=var_fix)
CB_setY.pack(side=Tk.LEFT)

vBC1=Tk.StringVar()
vBC2=Tk.StringVar()

BCrangeFrame = Tk.Frame(OptMenu,relief='raised',borderwidth=1)
BCrangeFrame.pack(side=Tk.TOP, expand=1)
labelC = Tk.Label(BCrangeFrame, text='Bright/Contrast', width=40)
labelC.pack(side=Tk.TOP, expand=1, fill=Tk.X)
BCrangeFrame1 = Tk.Frame(BCrangeFrame,relief='raised',borderwidth=1)
BCrangeFrame1.pack(side=Tk.TOP, expand=1,fill=Tk.X)
BCrangeFrame1a = Tk.Frame(BCrangeFrame,relief='raised',borderwidth=1)
BCrangeFrame1a.pack(side=Tk.TOP, expand=1,fill=Tk.X)
BCrangeFrame2 = Tk.Frame(BCrangeFrame,relief='raised',borderwidth=1)
BCrangeFrame2.pack(side=Tk.TOP, expand=1,fill=Tk.X)
BCrangeFrame2a = Tk.Frame(BCrangeFrame,relief='raised',borderwidth=1)
BCrangeFrame2a.pack(side=Tk.TOP, expand=1,fill=Tk.X)
BCrangeFrame3 = Tk.Frame(BCrangeFrame,relief='raised',borderwidth=1)
BCrangeFrame3.pack(side=Tk.TOP, expand=1,fill=Tk.X)
BClabW1=Tk.Label(BCrangeFrame1,text="Bright=",width=3,anchor='e')
BClabW1.pack(side=Tk.LEFT,expand=1,fill=Tk.X)
ceb1=Tk.Entry(BCrangeFrame1,textvariable=vBC1,width=6)
ceb1.pack(side=Tk.LEFT,expand=1,fill=Tk.X)
vBC1.set(bright)
BClabW2=Tk.Label(BCrangeFrame2,text="Contrast=",width=3,anchor='s')
BClabW2.pack(side=Tk.LEFT,expand=1,fill=Tk.X)
ceb2=Tk.Entry(BCrangeFrame2,textvariable=vBC2,width=6)
ceb2.pack(side=Tk.LEFT,expand=1,fill=Tk.X)
vBC2.set(contrast)
def setBC():
    global mapFig,ismark,isSmark,Obj,mask
    global bright,contrast,ceb1,ceb2
    global j,l,nx_med,ny_med
    bright=float(Tk.Entry.get(ceb1))
    contrast=float(Tk.Entry.get(ceb2))
    if (bright<0.01):
        bright=0.01
    if (bright>0.99):
        bright=1
    if (contrast<0.011):
        contrast=0.01
    if (contrast>0.989):
        contrast=0.99
    vBC1.set(bright)
    vBC2.set(contrast)
    animate(j,l)
#    plot_spec(nx_med,ny_med)

def setB_Scale(bright_now):
    global mapFig,ismark,isSmark,Obj,mask
    global bright,contrast,ceb1,ceb2
    global j,l,nx_med,ny_med
    global vBC1,vBC2
    vBC1.set(bright_now)
    bright_new=float(Tk.Entry.get(ceb1))
    if (bright != bright_new):
        bright=bright_new
        animate(j,l)
#    animate(j,l)
#    print bright,bright_now
#    animate(j,l)


def setC_Scale(contrast_now):
    global mapFig,ismark,isSmark,Obj,mask
    global bright,contrast,ceb1,ceb2
    global j,l,nx_med,ny_med
    global vBC1,vBC2
    vBC2.set(float(contrast_now))
    contrast_new=float(Tk.Entry.get(ceb2))
    if (contrast != contrast_new):
        contrast=contrast_new
        animate(j,l)



scaleBC1 = Tk.Scale(BCrangeFrame1a, from_=0.0, to=1.0, resolution=0.01, orient=Tk.HORIZONTAL, variable=vBC1, showvalue=0, relief='raised', command=setB_Scale)
scaleBC1.pack(side=Tk.TOP,expand=1,fill=Tk.X)
scaleBC2 = Tk.Scale(BCrangeFrame2a, from_=0.0, to=1.0, resolution=0.01, orient=Tk.HORIZONTAL, variable=vBC2, showvalue=0, relief='raised', command=setC_Scale)
scaleBC2.pack(side=Tk.TOP,expand=1,fill=Tk.X)
CB_setInv = Tk.Checkbutton(BCrangeFrame3, text="Invert", variable=var_invert)
CB_setInv.pack(side=Tk.LEFT)
buttonBC = Tk.Button(BCrangeFrame3, text='SET', command=setBC)
buttonBC.pack(side=Tk.LEFT,expand=1,fill=Tk.X)

nFig3D=1
def saveFig3D():
    global nFig3D
    figName='IFSview3D_'+str(nFig3D)+'.jpg'
    vv.screenshot(figName)
    print ' '+figName+' figure saved'
    nFig3D=nFig3D+1
def saveFig3D_pdf():
    global nFig3D
    figName='IFSview3D_'+str(nFig3D)+'.pdf'
    vv.screenshot(figName)
    print ' '+figName+' figure saved'
    nFig3D=nFig3D+1
startMov=0
def startMovFig3D():
    global rec,startMov
    a = vv.gca()
    f = vv.gcf()
    rec = vv.record(a)
    startMov=1
    print 'Start Recording Movie'
def stopMovFig3D_gif():
    global rec,startMov
    global nFig3D
    figName='IFSview3D_'+str(nFig3D)+'.gif'    
    if (startMov==1):
        print 'Stop Recording Movie'
        rec.Stop()
        rec.Export(figName)
        print ' '+figName+' movie saved'
    startMov=0
    nFig3D=nFig3D+1
def stopMovFig3D_swf():
    global rec,startMov
    global nFig3D
    figName='IFSview3D_'+str(nFig3D)+'.swf'    
    if (startMov==1):
        print 'Stop Recording Movie'
        rec.Stop()
        rec.Export(figName)
        print ' '+figName+' movie saved'
    startMov=0
    nFig3D=nFig3D+1
def addColormapEditor():
    a = vv.gca()
    f = vv.gcf()
    vv.ColormapEditor(a)
        
    
fig3DFrame = Tk.Frame(OptMenu,relief='raised',borderwidth=1)
fig3DFrame.pack(side=Tk.LEFT, expand=1)
label3D = Tk.Label(fig3DFrame, text='Fig3D', width=10)
label3D.pack(side=Tk.LEFT)
buttonFig3D1 = Tk.Button(fig3DFrame, text='Save JPG', command=saveFig3D)
buttonFig3D1.pack(side=Tk.TOP,expand=1,fill=Tk.X)
buttonFig3D2 = Tk.Button(fig3DFrame, text='Save PDF', command=saveFig3D_pdf)
buttonFig3D2.pack(side=Tk.TOP,expand=1,fill=Tk.X)
buttonFig3D3 = Tk.Button(fig3DFrame, text='Start Movie', command=startMovFig3D)
buttonFig3D3.pack(side=Tk.TOP,expand=1,fill=Tk.X)
buttonFig3D4 = Tk.Button(fig3DFrame, text='Save Movie (gif)', command=stopMovFig3D_gif)
buttonFig3D4.pack(side=Tk.TOP,expand=1,fill=Tk.X)
buttonFig3D5 = Tk.Button(fig3DFrame, text='Save Movie (swf)', command=stopMovFig3D_swf)
buttonFig3D5.pack(side=Tk.TOP,expand=1,fill=Tk.X)
#buttonFig3D6 = Tk.Button(fig3DFrame, text='ColorEditor', command=addColormapEditor)
#buttonFig3D6.pack(side=Tk.TOP,expand=1,fill=Tk.X)


colormapFrame = Tk.Frame(OptMenu,relief='raised',borderwidth=1)
colormapFrame.pack(side=Tk.LEFT, expand=1)
label = Tk.Label(colormapFrame, text='Color Map', width=10)
label.pack(side=Tk.TOP)
colormap = Tk.Listbox(colormapFrame, selectmode=Tk.SINGLE,
                      height=6,width=12)
scrollbar = Tk.Scrollbar(colormapFrame,orient=Tk.VERTICAL)
scrollbar['command'] = colormap.yview
colormap.config(yscrollcommand=scrollbar.set)
colormap.pack(side=Tk.LEFT)
scrollbar.pack(side=Tk.LEFT,fill=Tk.Y)

#for text, cont in CONTROLS:
#    colormap.insert(Tk.END, text)
for ll in range(0,l-1):
    m=maps[ll]
    text=(ll,m)
    colormap.insert(Tk.END,text)

colormap['selectmode']=Tk.SINGLE
def colormap_Click(event):
    val = 'none'
    for i in colormap.curselection():
        val = colormap.get(i)
        nval=val[0]
        change_color(nval)
colormap.bind("<ButtonRelease-1>", colormap_Click)

#vv.screenshot('IFSview3D.jpg')
#FIG3D FRAME




#    colormap.bind(m,change_color(ll))
#    colormap.bind(m)


# a tk.DrawingArea


canvas = FigureCanvasTkAgg(fig1, master=plotFrame11)
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
canvasb = FigureCanvasTkAgg(fig2, master=plotFrame12)
canvasb.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg( canvas, MidFrame)
toolbar2 = NavigationToolbar2TkAgg( canvasb, plotFrame2)
#toolbar3d = NavigationToolbar2TkAgg( canvas2, BottomFrame3D)

#canvas2 = FigureCanvasTkAgg(fig1b, master=plotFrame2)
#canvas2.get_tk_widget().pack(side=Tk.LEFT, fill=Tk.BOTH, expand=1)



        
#button2 = Tk.Button(master=root, text='NEXT', command=next)
#button2.pack(side=Tk.RIGHT)
#button2 = Tk.Button(master=root, text='LAST', command=last)
#button2.pack(side=Tk.RIGHT)

###mbs1=  Tk.Menubutton (master=root, text='Slice', relief='raised')
mbs1=  Tk.Menubutton (MenuBar, text='Slice')
mbs1.grid()
mbs1.menu  =  Tk.Menu ( mbs1, tearoff = 0 )
mbs1["menu"]  =  mbs1.menu
mbs1.menu.add_command( label='>>', command=nSlice)
mbs1.menu.add_command( label='<<', command=pSlice)
mbs1.menu.add_command( label='movie', command=movie)
mbs1.menu.add_command( label='specX', command=specX)
mbs1.menu.add_command( label='specY', command=specY)
mbs1.pack(side=Tk.LEFT)

#mbs=  Tk.Menubutton (master=root, text='Change Colormap', relief='raised')
mbs=  Tk.Menubutton (MenuBar, text='Change Colormap')
mbs.grid()
mbs.menu  =  Tk.Menu ( mbs, tearoff = 0 )
mbs["menu"]  =  mbs.menu
mbs.menu.add_command( label='>>', command=next)
mbs.menu.add_command( label='<<', command=previous)
mbs.menu.add_command( label='First', command=first)
mbs.menu.add_command( label='Last', command=last)
mbs.pack(side=Tk.LEFT)

def setSpec_0():
    global type_spectra
    type_spectra=0
    print 'Type ='+str(type_spectra)
def setSpec_1():
    global type_spectra
    type_spectra=1
    print 'Type ='+str(type_spectra)
def setSpec_2():
    global type_spectra
    type_spectra=2
    print 'Type ='+str(type_spectra)


mbs2=  Tk.Menubutton (MenuBar, text='Spectral Representation')
mbs2.grid()
mbs2.menu  =  Tk.Menu ( mbs2, tearoff = 0 )
mbs2["menu"]  =  mbs2.menu
mbs2.menu.add_command( label='Single', command=setSpec_0)
mbs2.menu.add_command( label='RSS image', command=setSpec_1)
mbs2.menu.add_command( label='All Spec', command=setSpec_2)
mbs2.pack(side=Tk.LEFT)



canvas.show()
canvasb.show()
#canvas2.show()

toolbar.update()
toolbar2.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
canvasb._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
#canvas2._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

change_color(j)

#################################
# 3D PLOT
def plot_3d_scatter(wave_1,wave_2,cut,jcm):
    global fitscube,mapFig,ismark,mark
    global specFig,isSpecFig,W1,W2
    global nx,ny,nz
    m=maps[jcm]
    wave_now=0.5*(wave_2+wave_1)
    i_med=int((wave_now-crval)/cdelt)
    i1=int((wave_1-crval)/cdelt)
    i2=int((wave_2-crval)/cdelt)
    if (i1>i2):
        i=i2
        i2=i1
        i1=i

    if ((i1>0) and (i2<nz)):
        nz_med=i_med
#        tmpdata=fitscube[i1:i2,:,:]
        fig = plt.figure()
        #fig.add_subplot(1,2,2)
        ax = Axes3D(fig)
#        ax = fig.gca(projection='3d')
        k=0
        NC=20
        step=int((i2-i1)/NC)
        if (step<3):
            i2=i1+3
            step=1
        for zs in range(i1,i2,step):
            k=0
            xs=[]
            ys=[]
            ZS=[]
            F=[]
            for ii in range(0,nx):
                for jj in range(0,ny):
                    val=fitscube[zs,jj,ii]                    
                    if (val>cut):
                        xs.append(ii)
                        ys.append(jj)

                        if (val>1):
                            val=1
                        if (val<0):
                            val=0
                        F.append(val)
                        k=k+1
                        XS=[]
                        YS=[]
            wave=crval+cdelt*zs
            ax.scatter(xs,ys,wave,alpha=0.1,s=5,c=F)

    ax.set_xlim3d(0,nx)
    ax.set_ylim3d(0,ny)
    ax.set_zlim3d(W1,W2)
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('Z label')
    plt.show()


def plot_3d(wave_1,wave_2,cut,jcm):
    global fitscube,mapFig,ismark,mark
    global specFig,isSpecFig,W1,W2
    global nx,ny,nz
    m=maps[jcm]
    wave_now=0.5*(wave_2+wave_1)
    i_med=int((wave_now-crval)/cdelt)
    i1=int((wave_1-crval)/cdelt)
    i2=int((wave_2-crval)/cdelt)
    if (i1>i2):
        i=i2
        i2=i1
        i1=i

    if ((i1>0) and (i2<nz)):
        nz_med=i_med
        XX=np.zeros((ny,nx))
        YY=np.zeros((ny,nx))
        fig = plt.figure(2)
        ax = Axes3D(fig)
#        ax = fig.gca(projection='3d')
        k=0
        levels=arange(0.05,1.2,0.05)
        for ii in range(0,nx):
            for jj in range(0,ny):                
                YY[jj,ii]=jj
                XX[jj,ii]=ii        
        k=0
        for zs in range(i1,i2):
            tmpdata=fitscube[zs,:,:]
            wave=crval+cdelt*zs
            alpha=0.5+0.5/(i2-i1)
            if (alpha>0.8):
                alpha=0.8
#            cset=ax.contourf(XX,YY,tmpdata,levels,zdir='z',offset=wave,lw=cdelt,alpha=alpha,hold='on',cmap=plt.get_cmap(m,len(levels)-1))
            cset=ax.contour(XX,YY,tmpdata,levels,zdir='z',offset=wave,lw=cdelt,alpha=alpha,hold='on',cmap=plt.get_cmap(m))
#            cset=ax.contourf(XX,YY,tmpdata,levels,zdir='z',offset=k,lw=cdelt,alpha=alpha,hold='on',cmap=plt.get_cmap('jet',len(levels)-1))
            k=k+1

    ax.set_xlim3d(0,nx)
    ax.set_ylim3d(0,ny)
    ax.set_zlim3d(W1,W2)
    ax.set_xlabel('X label')
    ax.set_ylabel('Y label')
    ax.set_zlabel('wavelength label')

    plt.show()

def plot_3d_fill(wave_1,wave_2,jcm):
    global fitscube,mapFig,ismark,mark
    global specFig,isSpecFig,W1,W2
    global nx,ny,nz
    NC=20
    m=maps[jcm]
    wave_now=0.5*(wave_2+wave_1)
    i_med=int((wave_now-crval)/cdelt)
    i1=int((wave_1-crval)/cdelt)
    i2=int((wave_2-crval)/cdelt)
    if (i1>i2):
        i=i2
        i2=i1
        i1=i
    wave_1=crval+cdelt*i1
    wave_2=crval+cdelt*i2

    if ((i1>0) and (i2<nz)):
        nz_med=i_med
        XX=np.zeros((ny,nx))
        YY=np.zeros((ny,nx))
        #fig1b.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.99)
#        ax3d = fig1b.add_subplot(1,1,1,projection='3d')
#        ax3d = fig1b.gca(projection='3d')
        ax3d = Axes3D(fig1b)
        ax3d.clear()
 #       onClick = ax3d.get_figure().canvas.mpl_connect('button_press_event',click)
 #       onMove = ax3d.get_figure().canvas.mpl_connect('motion_notify_event',move)
        k=0
        for ii in range(0,nx):
            for jj in range(0,ny):                
                YY[jj,ii]=jj
                XX[jj,ii]=ii        
        k=0

        i_med=int(0.5*(i1+i2))
        offset=np.ones((ny,nx))
        wave=crval+cdelt*i1
        tmpdata=fitsdata*mask 
        vmax=tmpdata.max()
        vmin=tmpdata.min()        
        tmpdata1=tmpdata+wave*offset 
        levels=arange(wave+0.05*(i2-i1),wave+vmax,0.05*(i2-i1))
        cset=ax3d.contourf(XX,YY,tmpdata1,levels,zdir='z',lw=cdelt,alpha=0.9,cmap=plt.get_cmap(m,len(levels)-1))
        tmpdata=fitscube[i_med,:,:]
        offset=np.ones((ny,nx))
        tmpdata=tmpdata*mask 
        vmax=tmpdata.max()
        vmin=tmpdata.min()        
        step=int((i2-i1)/NC)
        if (step<3):
            i2=i1+3
            step=1


        for zs in range(i1,i2,step):
            tmpdata=fitscube[zs,:,:]
            wave=crval+cdelt*zs
            tmpdata=tmpdata*mask 
            tmpdata1=tmpdata+wave*offset 
#            levels=arange(wave+0.05,wave+vmax,0.05)
            levels=arange(wave+0.05,wave+vmax,(vmax-0.05)/NC)
            #print levels
            #print tmpdata1[35,35]
            alpha=0.5+0.5/(i2-i1)
            if (alpha>0.8):
                alpha=0.8
            cset=ax3d.contourf(XX,YY,tmpdata1,levels,zdir='z',lw=cdelt,alpha=0.5,cmap=plt.get_cmap(m,len(levels)-1))
         #   cset=ax.contour(XX,YY,tmpdata1,levels,zdir='z',lw=cdelt,alpha=0.1,cmap=plt.get_cmap(m,len(levels)-1),extend3d=True)
            k=k+1
    ax3d.set_xlim3d(0,nx)
    ax3d.set_ylim3d(0,ny)
    ax3d.set_zlim3d(wave_1,wave_2)
    ax3d.set_xlabel('X label')
    ax3d.set_ylabel('Y label')
    ax3d.set_zlabel('wavelength label')
    ax3d.get_figure().canvas.draw()                
#            if (zs==i1):
#                plt.show()
#            else:
#        plt.show()

#a = vv.gca()
def plot_3d_nofill(wave_1,wave_2,jcm):
    global fitscube,mapFig,ismark,mark
    global specFig,isSpecFig,W1,W2
    global nx,ny,nz,a,contrast
    global app
    invert=var_invert.get()
    fixY=var_fix.get()
    m=maps[jcm]
    cmap_name=plt.get_cmap(m)    
    my_str=cmap_name.name
    if (invert==1):
        my_str=my_str+'_r'
    cmap=cm.get_cmap(my_str)
    mod_cmap=cmap_center_adjust(cmap, contrast)

    NC=20
    m=maps[jcm]
    wave_now=0.5*(wave_2+wave_1)
    i_med=int((wave_now-crval)/cdelt)
    i1=int((wave_1-crval)/cdelt)
    i2=int((wave_2-crval)/cdelt)
    if (i1>i2):
        i=i2
        i2=i1
        i1=i
    wave_1=crval+cdelt*i1
    wave_2=crval+cdelt*i2

    if ((i1>0) and (i2<nz)):
        nz_med=i_med
        XX=np.zeros((ny,nx))
        YY=np.zeros((ny,nx))
        tmpdata=np.fliplr(fitscube[i1:i2,:,:])
        
#        MyThread().start()        
#matplotlib.use('TkAgg')
#        app=vv.use()
#        a = vv.gca()
        vv.clf()
#vv.xlabel('x axis')
#vv.ylabel('y axis')
#vv.zlabel('z axis')

# show
        t = vv.volshow(tmpdata, renderStyle='mip')
#        t.colormap = vv.CM_JET

        cdict = copy.copy(mod_cmap._segmentdata)    
        a_red=np.asarray(cdict['red'],float)
        a_green=np.asarray(cdict['green'],float)
        a_blue=np.asarray(cdict['blue'],float)
        
        A_red=a_red[:,1]
        A_green=a_green[:,1]
        A_blue=a_blue[:,1]
        n_c=A_red.shape
#        n_cc=n_c[0]
#        A_cmap=np.array([3,n_c])
        A_cmap=np.zeros((n_c[0],3))
        for ii in range(0,n_c[0]):
           A_cmap[ii][0]=A_red[ii]
           A_cmap[ii][1]=A_green[ii]
           A_cmap[ii][2]=A_blue[ii]
            
        #t.colormap=A_cmap
         
        a_cmap=tuple(tuple(x) for x in A_cmap)

        t.colormap=a_cmap
#        print A_cmap,a_cmap,t.colormap
#        print A_red,A_green,A_blue
#        print t.colormap
#        t.colormap = cmap.datad


        #a.camera.fov = 0
#        vv.screenshot('IFSview3D_test.jpg')
        #vv.ColormapEditor(a)
#        app.Run()
#        app.Create()
#        app.Run()
#        app.show()

#        app=vv.use()
#        app.Run()
#        app.Create()
#        app.Run()
#        p=Process(target=app.Run())
#        p.join()
#        print 'Continue the Loop!!!'

def plot_3d_nofill_old(wave_1,wave_2,jcm):
    global fitscube,mapFig,ismark,mark
    global specFig,isSpecFig,W1,W2
    global nx,ny,nz
    NC=20
    m=maps[jcm]
    wave_now=0.5*(wave_2+wave_1)
    i_med=int((wave_now-crval)/cdelt)
    i1=int((wave_1-crval)/cdelt)
    i2=int((wave_2-crval)/cdelt)
    if (i1>i2):
        i=i2
        i2=i1
        i1=i
    wave_1=crval+cdelt*i1
    wave_2=crval+cdelt*i2

    if ((i1>0) and (i2<nz)):
        nz_med=i_med
        XX=np.zeros((ny,nx))
        YY=np.zeros((ny,nx))
        #fig1b.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.99)
        #ax3d = fig1b.add_subplot(1,1,1,projection='3d')
#        ax3d = fig1b.gca(projection='3d')        
        ax3d = Axes3D(fig1b)
        ax3d.clear()
        ax3d.cla()
        ax3d.mouse_init()

        k=0
        for ii in range(0,nx):
            for jj in range(0,ny):                
                YY[jj,ii]=jj
                XX[jj,ii]=ii        
        k=0

        i_med=int(0.5*(i1+i2))
        offset=np.ones((ny,nx))
        wave=crval+cdelt*i1
        tmpdata=fitsdata*mask 
        vmax=tmpdata.max()
        vmin=tmpdata.min()        
        tmpdata1=tmpdata+wave*offset 
        levels=arange(wave+0.05*(i2-i1),wave+vmax,0.05*(i2-i1))
        #cset=ax3d.contour(XX,YY,tmpdata1,levels,zdir='z',lw=cdelt,alpha=0.9,cmap=plt.get_cmap(m,len(levels)-1))
#        cset=ax3d.contour(XX,YY,tmpdata1,levels,zdir='z',lw=cdelt,alpha=0.9,cmap=plt.get_cmap(m,len(levels)-1))
        tmpdata=fitscube[i_med,:,:]
        offset=np.ones((ny,nx))
        tmpdata=tmpdata*mask 
        vmax=tmpdata.max()
        vmin=tmpdata.min()        
        step=int((i2-i1)/NC)
        if (step<1):
            i2=i1+1
            step=1
            wave_2=crval+cdelt*i2

        for zs in range(i1,i2,step):
            tmpdata=fitscube[zs,:,:]
            wave=crval+cdelt*zs
            tmpdata=tmpdata*mask 
            tmpdata1=tmpdata+wave*offset 
            levels=arange(wave+0.05,wave+vmax,(vmax-0.05)/NC)
            #print levels
            #print tmpdata1[35,35]
            alpha=0.5+0.5/(i2-i1)
            if (alpha>0.8):
                alpha=0.8
            cset=ax3d.contour(XX,YY,tmpdata1,levels,zdir='z',lw=cdelt*NC,alpha=0.5,cmap=plt.get_cmap(m,len(levels)-1))
            k=k+1
    ax3d.set_xlim3d(0,nx)
    ax3d.set_ylim3d(0,ny)
    ax3d.set_zlim3d(wave_1,wave_2)
    ax3d.set_xlabel('X label')
    ax3d.set_ylabel('Y label')
    ax3d.set_zlabel('wavelength label')
#    ax3d.legend()
#    ax3d.get_figure().canvas.draw()                
    ax3d.get_figure().canvas.show()                
    #Axes3D.mouse_init()
#            if (zs==i1):
#    plt.show()
#            else:
#        plt.show()
#    ax3d.show()
                        


#################################

onClick_map = fig1.canvas.mpl_connect('button_press_event',click)
onMove_map = fig1.canvas.mpl_connect('motion_notify_event',move)

onClick = fig2.canvas.mpl_connect('button_press_event',click)
onMove = fig2.canvas.mpl_connect('motion_notify_event',move)
plot_3d_nofill(W1,W2,j)


#onClick_3d = fig1b.canvas.mpl_connect('button_press_event',click)
#onMove_3d = fig1b.canvas.mpl_connect('motion_notify_event',move)
#toolbar2 = NavigationToolbar2TkAgg( canvas2, plotFrame3)

#gui3d.show()
#appTk.run
#app.Create()
#def task():
#    global app,root
#    root.after(2000,root.mainloop())
#root.after(2000,task())
#print 'Continue!!!'





#app.Run()

#app.Run()

#       print 'Continue the Loop!!!'

#app.Run()

#root.mainloop()
#gui3d.mainloop()




app.Create()
Tk.mainloop()
app.Run()
