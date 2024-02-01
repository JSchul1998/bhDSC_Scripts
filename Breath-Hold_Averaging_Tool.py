import numpy as np 
import nibabel as nib
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy.signal import find_peaks

###User must specify the pre-processed signal file, GM mask, and WM mask
data_n = nib.load('Filepath/Signal.nii.gz')
data = data_n.get_fdata()
gm = nib.load('Filepath/GM.nii.gz')
gm = gm.get_fdata()
wm = nib.load('Filepath/WM.nii.gz')
wm = wm.get_fdata()

##File Co-ordinates
x = data.shape[0]
y = data.shape[1]
z = data.shape[2]
t = data.shape[3]
X = np.array(range(t))

##Bound specifies half of the window size that is centred around the bolus peak (i.e., If you want a final signal time-course duration of 20, specify 10 here)
bound = 20
##These are key timepoints of the final averaged bolus (first two: pre-baseline; middle two: Bolus; final two: post-baseline)
##User can play around with time-points to centre around the bolus
t1, t2, t3, t4, t5, t6 = 0,5,10,30,30,35
##Minimum distance between breath-holds (modify if breath-holds are further apart)
dist_param = 20

##Parameters needed for the initial high-pass filter (only modify parameters with associated comments)
##TR and TE used in the scan 
TR, TE = 2, 0.02
T = t*TR
##fs is the sampling frequency in your scan (1/TR)
fs = 0.5
##High-pass filter frequency cut-off (to remove low-frequency signal variations, and in effect, detrend the data)
cutoff1 = 0.01
nyq = 0.5*fs
order = 2
##Defines high pass filter function
def butter_highpass_filter(data, cutoff1, fs, order):
    normal_cutoff = cutoff1 / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='hp', analog=False)
    out = filtfilt(b, a, data)
    return out

##This iterates through the signal time course map and applies the high-pass filter
Y = []
for i in range(x):
    print("Progression of Filtering", (i/data.shape[0])*100,"%")
    for j in range(y):
        for k in range(z):
            Data = data[i, j, k]
            Data_mean = np.mean(Data)
            if Data_mean == 0:
                Y.append(np.zeros(t))
            else:
                yy = butter_highpass_filter(Data, cutoff1, fs, order) + Data_mean
                Y.append(yy)
Y = np.array(Y)
Y = Y.reshape(x, y, z, t)

##Next, this iterates through high-pass filtered map and obtains an average GM time-course for reference (in order to identify breath-hold peaks)
GM = []
for i in range(x):
    print("Progression of Averaging", (i/data.shape[0])*100,"%")
    for j in range(y):
        for k in range(z):
            GMm = gm[i, j, k]
            a = Y[i, j, k]
            a_mean = np.mean(a)
            if GMm > 0 and a_mean != 0:
                GM.append(a)
GM = np.array(GM)
GM_av = GM.mean(0)

##Displays the average GM time-course and the associated peaks using the find_peaks tool
plt.plot(X, GM_av)
plt.hlines(np.mean(GM_av)+0.75*np.std(GM_av), 0, len(GM_av),linestyles='dashed', colors='black')
GM_av_p = find_peaks(GM_av, height=(np.mean(GM_av)+0.75*np.std(GM_av)), distance=dist_param)
GM_av_p = GM_av_p[0]
plt.scatter(GM_av_p, GM_av[GM_av_p], color='red', marker='x')
plt.show()

##Obtains the breath-hold averaged GM time course to determine where bolus and baseline are located
GM_new = []
for l in GM_av_p:
    if bound < l < (t - bound):
        GM_new.append(GM_av[l-bound-1:l+bound-1])
    else:
        pass
GM_new = (np.array(GM_new)).mean(0)

##Displays the breath-hold averaged GM time course and baseline bounds (can be changed at line 27)
plt.plot(np.array(range(len(GM_new))),GM_new)
plt.vlines(t1,np.mean(GM_new)-2*(np.std(GM_new)),np.mean(GM_new)+2*(np.std(GM_new)), linestyles='dashed', colors='black')
plt.vlines(t2,np.mean(GM_new)-2*(np.std(GM_new)),np.mean(GM_new)+2*(np.std(GM_new)), linestyles='dashed', colors='black')
plt.vlines(t5,np.mean(GM_new)-2*(np.std(GM_new)),np.mean(GM_new)+2*(np.std(GM_new)), linestyles='dashed', colors='black')
plt.vlines(t6,np.mean(GM_new)-2*(np.std(GM_new)),np.mean(GM_new)+2*(np.std(GM_new)), linestyles='dashed', colors='black')
plt.show()
print(len(GM_new))
print(np.argmax(GM_new))


##Finally, this iterates through the entire nii file to generate breath-hold averaged maps, CNR, and dS files
S = []
R = []
cnr = []
gmcnr = []
wmcnr = []
ds_gm = []
ds_wm = []
ds = []
for i in range(x):
    print("Progression of Iterating", (i/data.shape[0])*100,"%")
    for j in range(y):
        for k in range(z):
            Data = data[i, j, k]
            Gm = gm[i,j,k]
            Wm = wm[i,j,k]
            data_mean = np.mean(Data)    

            if data_mean == 0:
                S.append(Data[100-bound-1:100+bound-1])
                R.append(Data[100-bound-1:100+bound-1])
                cnr.append(0)
                ds.append(0)
            else:
                data_new = []
                for l in GM_av_p:
                    if (bound+1) < l < (t - (bound+1)):
                        data_new.append(Data[l-bound-1:l+bound-1])
                    else:
                        pass
                data_new = (np.array(data_new)).mean(0)
                S.append(data_new)
                Average_sig = (np.mean(data_new[t1:t2]) + np.mean(data_new[t5:t6]))/2

                ##Calculate CNR for voxel (in GM and WM)
                Peak = -(np.mean(Average_sig) - max(data_new[t3:t4]))
                dS = ((Peak)/np.mean(Average_sig))*100 
                std = (np.std(data_new[t1:t2]) + np.std(data_new[t5:t6]))/2
                CNR = Peak/std
                cnr.append(CNR)
                ds.append(dS)
                if Gm > 0:
                    gmcnr.append(CNR)
                    ds_gm.append(dS)
                if Wm > 0:
                    wmcnr.append(CNR)
                    ds_wm.append(dS)
                r = (-1/TE)*np.log(data_new/Average_sig)
                R.append(r)

S = np.array(S)
R = np.array(R)
cnr = np.array(cnr)
ds = np.array(ds)
GM_CNR = np.array(gmcnr)
WM_CNR = np.array(wmcnr)
dS_GM = np.array(ds_gm)
dS_WM = np.array(ds_wm)


##Outputs the breath-hold averaged time courses, CNR, and dS Maps (user must change filepath as needed)
newarr = S.reshape(x, y, z, len(GM_new))
new_image = nib.Nifti1Image(newarr, data_n.affine)
nib.save(new_image, 'Filepath/Average_Signal.nii.gz')
newarr = R.reshape(x, y, z, len(GM_new))
new_image = nib.Nifti1Image(newarr, data_n.affine)
nib.save(new_image, 'Filepath/Average_Relaxation.nii.gz')
newarr = cnr.reshape(x, y, z)
new_image = nib.Nifti1Image(newarr, data_n.affine)
nib.save(new_image, 'Filepath/CNR.nii.gz')
newarr = ds.reshape(x, y, z)
new_image = nib.Nifti1Image(newarr, data_n.affine)
nib.save(new_image, 'Filepath/dS.nii.gz')


##This final code computes the average GM and WM CNR and delta S values, along with the delta CNR (GM-WM)
dSg_Mean = np.mean(dS_GM)
dSw_Mean = np.mean(dS_WM)
CNRg_Mean = np.mean(GM_CNR)
CNRw_Mean = np.mean(WM_CNR)
CNRw_std = np.std(WM_CNR)
print("GM dS = ", dSg_Mean)
print("WM dS = ", dSw_Mean)
print("GM CNR = ", CNRg_Mean)
print("WM CNR = ", CNRw_Mean)
print("dCNR = ", (CNRg_Mean-CNRw_Mean)/(CNRw_std))
