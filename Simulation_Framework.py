## Import Libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import metrics
import nibabel as nib

##Specify arterial (A), arteriolar (a), venule (v), venous (V), and capillary (c) baseline oxygenations
A = 0.98
a = A - 0.05
v = V = 0.65
c = (a+V)/2
##Specify TE
TE = 0.02

##Specify duration of time course
l = 40
X = np.array(range(l))
X2 = np.array(range(2*l))
##delta SvO2 During a Breath Hold (from Sasse et al., 1996 and Jain et al., 2011) for capillary --> can change depending on duration of breath-hold
SvO2 = 0.075
##delta SaO2 During a Breath Hold (from J. P. A. Andersson, L. Evaggelidis, 2009) for artery 
SaO2 = 0.008
##Interpolation of Tissue oxygenation change based on arterial and venous
StO2 = (SaO2 + SvO2)/2

##Defines gamma variate of breath-hold time course (stretch coincides with the peak of the time course following the initiation of signal change)
stretch = 4
Y = SaO2*((X/stretch)**3)*np.exp(3*(1-(X/stretch)))
Y = np.pad(Y, (10, 0), 'constant', constant_values=(0, 0))
Y = Y[0:len(X)]

##Defines delay for tissue and vein
d, d2 = 1, 2
##Defines residue function for tissue and vein
def residue(x, f, T, t):
    return f*np.exp(-T*(x)) + (1-f)*np.exp(-t*(x))
#Not that MTT = f/T + (1-f)/t

##StO2 time-course (roughly f = 0.895 for WM, f = 0.92 for GM) --> Convolves arterial input with tissue residue
y2 = np.convolve(Y, residue(X,0.92,0.68,0.05))
y2 = np.delete(y2, np.s_[len(X):len(X2)])
y2 = np.pad(y2, (d, 0), 'constant', constant_values=(0, 0))[0:len(Y)]
Y1 = (y2/np.max(y2))*StO2
##SvO2 time-course --> Convolves tissue input with venous residue 
#y3 = np.convolve(Y1, residue(X,0.98,0.68,0.05))
#y3 = np.delete(y3, np.s_[len(X):len(X2)])
y3 = np.pad(Y1, (d2, 0), 'constant', constant_values=(0, 0))[0:len(Y)]
Y2 = (y3/np.max(y3))*SvO2

##SaO2 time-course in artery can only increase to a maximum of 100% oxygen saturation --> This is incorporated here
new = []
for i in Y:
    if abs(i) >= (1-A):
        new.append((1-A))
    else: 
        new.append(i)
Y = np.array(new)
##SaO2 time-course in arteriole can also only increase to a maximum of 100% oxygen saturation --> This is incorporated here
new2 = []
for i in (Y+Y1)/2:
    if abs(i) >= (1-a):
        new2.append((1-a))
    else: 
        new2.append(i)
Ya = np.array(new2)

##The following 7T relaxation values and functions are from Uludag 2009 (Ex), extrapolated from Uludag (In), or based on experimental data (CSF)
##NOTE: If conducting a simulation at a field strength other than 7T, MUST CHANGE the following values/functions (can refer to Uludag 2009)
R2_Star_In = 116
R2_Star_Ex = 32.6
R2_Star_CSF = 4.35
#R2*Hb,In (Page 6 (Equation 9b Kamil))
R2_Star_In_c = 549*(np.square(1-(c+Y1))) + R2_Star_In
R2_Star_In_a = 549*(np.square(1-(a+Ya))) + R2_Star_In
R2_Star_In_v = 549*(np.square(1-(v+Y2))) + R2_Star_In
R2_Star_In_A = 549*(np.square(1-(A+Y))) + R2_Star_In
R2_Star_In_V = 549*(np.square(1-(V+Y2))) + R2_Star_In
#Frequency Change based on hematocrit and oxygenation (Page 6 (Equation 10 Kamil))
vs_c = 197.86*np.absolute(1-(c+Y1))*0.69
vs_a = 197.86*np.absolute(1-(a+Ya))*0.69
vs_v = 197.86*np.absolute(1-(v+Y2))*0.69
vs_A = 197.86*np.absolute(1-(A+Y))
vs_V = 197.86*np.absolute(1-(V+Y2))
#R2*Hb,Ex (Table 2 Page 7 Kamil)
R2_Star_Ex_c = 0.03865374*vs_c
R2_Star_Ex_v = 0.04330869*vs_v
R2_Star_Ex_a = 0.04330869*vs_a
R2_Star_Ex_V = 0.07976623*vs_V
R2_Star_Ex_A = 0.07976623*vs_A

##This block incorporates a time course of vasodilation into simulation
##Note that vasodilation (dCBV) values are based on Grubb's Formula (0.18 for vein, 0.38 for artery, in between for tissue; assuming a CBF change of 15%) --> can change values when breath-hold is longer or shorter than 16 seconds
##Shift is the amount of time the vasodilation occurs prior to the oxygenation change
shift = 1
Tiss1 = ((Y/np.max(Y))*0.0545) + 1
Tiss1 = np.pad(Tiss1, (0, shift), 'constant', constant_values=(1, 1))[shift:]
Tiss = ((Y1/np.max(Y1))*0.04) + 1
Tiss = np.pad(Tiss, (0, shift), 'constant', constant_values=(1, 1))[shift:]
Tiss2 = ((Y2/np.max(Y2))*0.025) + 1
Tiss2 = np.pad(Tiss2, (0, shift), 'constant', constant_values=(1, 1))[shift:]
##Comment in if you want to simulate without a CBV change
#Tiss, Tiss1, Tiss2 = 1, 1, 1

##Voxel Properties  (CBV in tissue; arterial blood volume in AIF (ABV); venous blood volume in VOF (VBV); CSF volume in AIF (CSF); CSF colume in VOF (CSFV))
CBV = 0.04
ABV, VBV = 0.5, 0.9
ArtVol = Tiss1*ABV
ArtVolV = Tiss2*VBV
CSF, CSFV = 0.1 - (Tiss1*ABV - ABV), 0.15 - (Tiss2*VBV - VBV)
##Comment in if NO CSF in voxel
#CSF, CSFV = 0, 0
##Alternative AIF Configuration (uncomment next four lines if you want to try the alternate AIF voxel configuration)
#ABV = 1
#ArtVol = Tiss1*ABV
#ArtVol = ArtVol - 1
#CSF = 0.6 - (Tiss1*ABV - ABV)

R2_Star_Micro_Ex = 100*CBV*(Tiss*0.4*R2_Star_Ex_c + Tiss2*0.4*R2_Star_Ex_v + Tiss1*0.2*R2_Star_Ex_a)


##Signal time courses
S_Micro = (1-CBV*Tiss)*np.exp(-(R2_Star_Ex+R2_Star_Micro_Ex)*TE) + (0.4*CBV*Tiss*np.exp(-(R2_Star_In_c)*TE) + 0.4*CBV*Tiss2*np.exp(-(R2_Star_In_v)*TE) + 0.2*CBV*Tiss1*np.exp(-(R2_Star_In_a)*TE))
S_Artery = CSF*np.exp(-(R2_Star_CSF+R2_Star_Ex_A*(ArtVol*100))*TE) + (1-(ArtVol+CSF))*np.exp(-(R2_Star_Ex+R2_Star_Micro_Ex*(1-(ArtVol+CSF))+R2_Star_Ex_A*(ArtVol*100))*TE) + ArtVol*np.exp(-(R2_Star_In_A)*TE) + (1-(ArtVol+CSF))*(0.4*CBV*Tiss*np.exp(-(R2_Star_In_c)*TE) + 0.4*CBV*Tiss2*np.exp(-(R2_Star_In_v)*TE) + 0.2*CBV*Tiss1*np.exp(-(R2_Star_In_a)*TE))
S_Vein = CSFV*np.exp(-(R2_Star_CSF+R2_Star_Ex_V*(ArtVolV*100))*TE) + (1-(ArtVolV+CSFV))*np.exp(-(R2_Star_Ex+R2_Star_Micro_Ex*(1-(ArtVolV+CSFV))+R2_Star_Ex_V*(ArtVolV*100))*TE) + ArtVolV*np.exp(-(R2_Star_In_V)*TE) + (1-(ArtVolV+CSFV))*(0.4*CBV*Tiss*np.exp(-(R2_Star_In_c)*TE) + 0.4*CBV*Tiss2*np.exp(-(R2_Star_In_v)*TE) + 0.2*CBV*Tiss1*np.exp(-(R2_Star_In_a)*TE))
##Relaxation time courses
AIF = (-1/TE)*np.log((S_Artery/S_Artery[1]))
Tissue = (-1/TE)*np.log((S_Micro/S_Micro[1]))
Vein = (-1/TE)*np.log((S_Vein/S_Vein[1]))

##Here, can plot the signal time courses for artery, tissue, and vein
#plt.plot(X, S_Artery, color = 'red', label = 'Artery 7T')
#plt.plot(X, S_Vein, color = 'blue', label = 'Vein 7T')
#plt.plot(X, S_Micro, color = 'green', label = 'Tissue 7T')

##Here, can plot the relaxation time courses for artery, tissue, and vein
#plt.plot(X, AIF, color = 'maroon', label = 'Artery 7T')
#plt.plot(X, Vein, color = 'darkblue', label = 'Vein 7T')
#plt.plot(X, Tissue, color = 'darkgreen', label = 'GM 7T')

plt.legend()
plt.show()
