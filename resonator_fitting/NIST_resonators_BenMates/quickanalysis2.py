import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import fitresonance as fr
import lambdafit as lf

Phi0 = 2.068e-15
Z1 = 50.0
Ls = 22.4e-12

temp = np.load("frsurvey_umux1M_v2_(0,7)W0_Band06_20251201.npz")
f_wide = temp['f_wide']
s21_wide = temp['s21_wide']
nres = 33
fc = np.zeros(nres)
fc[:17] = temp['fc'][:17]
fc[17:] = temp['fc'][16:]
nres = len(fc)
ibias = temp['ibias']
nbias = len(ibias)
fbias = temp['fbias']
npts = 401
s21_fr = np.zeros((nres,nbias,npts),dtype=complex)
s21_fr[:17,:,:] = temp['s21_fr'][:17,:,:]
s21_fr[17:,:,:] = temp['s21_fr'][16:,:,:]
tau = temp['tau']



pf = np.polyfit(f_wide,np.unwrap(np.angle(s21_wide)),deg=1)
# s21_wide_temp = s21_wide * np.exp(-1j*(pf[1] + pf[0]*f_wide))
for n in range(nres):
    for m in range(nbias):
        ftemp = fc[n] + fbias
        s21_fr[n,m,:] = s21_fr[n,m,:] * np.exp(-2j*np.pi*tau*ftemp) * np.exp(-1j*(pf[1] + pf[0]*ftemp))

bad = np.array([16,17])
good = np.setdiff1d(np.arange(nres), bad)

f0fits = np.zeros((nres,len(ibias)))
Qcfits = np.zeros((nres,len(ibias)))
Qifits = np.zeros((nres,len(ibias)))

for n in range(nres):
    if n in bad:
        continue
    ftemp = fc[n] + fbias[100:-100]
    for m in range(nbias):
        # print('Res {:d} bias {:d}'.format(n,m))
        s21temp = s21_fr[n,m,100:-100]
        try:
            if m == 0:
                f0fittemp,Qcfittemp,Qifittemp = fr.fit_resonance(ftemp,s21temp,showplot=False)
            else:
                f0fittemp,Qcfittemp,Qifittemp = fr.fit_resonance(ftemp,s21temp,showplot=False)
            f0fits[n,m] = f0fittemp
            Qcfits[n,m] = Qcfittemp
            Qifits[n,m] = Qifittemp
        except:
            print('Fitting failed on resonator {:d}'.format(n))
            f0fits[n,m] = 0.0
            Qcfits[n,m] = 0.0
            Qifits[n,m] = 0.0

# Fit SQUID parameters
I0fits = np.zeros(nres)
Minfits = np.zeros(nres)
Mcfits = np.zeros(nres)
fbfits = np.zeros(nres)
lambfits = np.zeros(nres)
for n in range(nres):
    if n in bad:
        continue
    if n == 0:
        I0fits[n],Minfits[n],Mcfits[n],fbfits[n],lambfits[n] = lf.fit_lambda(ibias,f0fits[n,:],showplot=False)
    else:
        I0fits[n],Minfits[n],Mcfits[n],fbfits[n],lambfits[n] = lf.fit_lambda(ibias,f0fits[n,:],showplot=False)

dfppfits = ((8*fbfits**2 * Mcfits**2)/(Z1*Ls)) * lambfits/(1-lambfits**2)



plt.figure(1,figsize=(9,5))
plt.plot(f_wide/1e9,20*np.log10(np.abs(s21_wide))+17.5)
plt.xlim(f_wide[0]/1e9,f_wide[-1]/1e9)
plt.xlabel('Frequency (GHz)')
plt.ylabel('|S21| (dB)')

plt.savefig("S21.png")

# Plot frequency distribution
#fmfits = np.mean(f0fits,axis=1)
plt.figure(100,figsize=(6.5,8))
plt.subplot(2,1,1)
plt.plot(np.arange(nres)[good],fbfits[good]/1e9,'o')
plt.xlabel('Resonator #')
plt.ylabel('Frequency (GHz)')
plt.ylim(f_wide[0]/1e9,f_wide[-1]/1e9)
plt.subplot(2,1,2)
plt.hist(np.diff(fbfits[good])/1e6,bins=np.arange(0.0,10.0,0.5))
plt.plot(np.ones(2)*7.0*1.0,np.array([0.0,0.4])*nres,'--k')
plt.xlim(0,10)
plt.xlabel('Frequency spacing (MHz)')
print('Resonator spacing = {:f} +/- {:f} MHz'.format(np.mean(np.diff(np.mean(f0fits[good],axis=1)))/1e6,np.std(np.diff(np.mean(f0fits[good],axis=1)))/1e6))

plt.savefig("spacing.png")
# Bandwidth distribution
Qfits = 1 / (1/np.mean(Qifits,axis=1) + 1/np.mean(Qcfits,axis=1))
BWfits = fbfits / Qfits

plt.figure(200,figsize=(6.5,8))
plt.subplot(2,1,1)
# plt.plot(xpos[good],BW/1e6,'o')
plt.plot(np.arange(nres)[good],BWfits[good]/1e6,'o')
plt.plot(np.array([0,33]),np.ones(2)*1.0,'--k')
plt.ylim(0,2.0)
plt.xlabel('Resonator #')
plt.ylabel('Resonator Bandwidth (MHz)')
plt.subplot(2,1,2)
plt.hist(BWfits[good]/1e6,bins=np.arange(0,2.0,0.1))
plt.plot(np.ones(2)*1.0,np.array([0.0,0.4])*nres,'--k')
plt.xlim(0,2.0)
plt.xlabel('Resonator Bandwidth (MHz)')
print('Resonator bandwidth = {:f} +/- {:f} MHz'.format(np.mean(BWfits[good])/1e6,np.std(BWfits[good])/1e6))
plt.savefig("bw.png")

plt.figure(300,figsize=(6.5,8))
plt.subplot(2,1,1)
plt.plot(np.arange(nres)[good],dfppfits[good]/1e6,'o')
plt.plot(np.array([0,33]),np.ones(2)*1.0,'--k')
plt.ylim(0,2.0)
plt.xlabel('Resonator #')
plt.ylabel('Peak-to-peak shift (MHz)')
plt.subplot(2,1,2)
plt.hist(dfppfits[good]/1e6,bins=np.arange(0,2.0,0.1))
plt.plot(np.ones(2)*1.0,np.array([0.0,0.4])*nres,'--k')
plt.xlim(0,2)
plt.xlabel('Peak-to-peak shift (MHz)')
print('Peak-to-peak shift = {:f} +/- {:f} MHz'.format(np.mean(dfppfits)/1e6,np.std(dfppfits)/1e6))
plt.savefig("shift.png")
Mm = np.mean(Minfits)
plt.figure(401,figsize=(6.5,8))
plt.subplot(2,1,1)
plt.plot(ibias*Mm/2.068e-15,np.transpose(f0fits[good,:]-np.outer(fbfits[good],np.ones(len(ibias))))/1e6)
plt.xlim(0,1.5)
plt.xlabel('Flux (Phi0)')
plt.ylim(-1.0,+0.5)
plt.ylabel('Frequency shift (MHz)')
plt.subplot(2,1,2)
plt.hist(lambfits[good],bins=np.linspace(0,1,30))
plt.plot(np.ones(2)*0.33,np.array([0.0,0.4])*nres,'--k')
plt.xlim(0,1)
plt.xlabel('lambda')
print('lambda = {:f} +/- {:f}'.format(np.mean(lambfits[good]),np.std(lambfits[good])))
plt.savefig("lambda.png")
