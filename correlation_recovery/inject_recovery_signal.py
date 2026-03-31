import numpy as np
import matplotlib.pyplot as plt
import libstempo as T
import libstempo.plot as LP, toasim as LT
from libstempo import spharmORFbasis as anis
import glob
import math
import json
import os
import sys
import scipy.interpolate as interp
import argparse
from enterprise.signals import gp_signals
from enterprise_extensions import model_utils
import enterprise_extensions.blocks as blocks
#from enterprise_extensions import blocks
from enterprise.signals import signal_base
from enterprise.signals import utils
import enterprise.constants as const
from defiant import OptimalStatistic
from defiant import plotting as defplot
from la_forge.core import Core
from enterprise.pulsar import Pulsar, Tempo2Pulsar
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
import enterprise.signals.parameter as parameter
from enterprise.signals import white_signals

import corner
from enterprise_extensions import sampler as sp
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', dest='datadir', type=str, help='Folder with the initial par/tim')
parser.add_argument('--comp', dest='comp', type=int, help='Number of frequency components')
parser.add_argument('--iter_num', dest='iter_num', type=float, help='Number of mcmc iterations')
parser.add_argument('--datadir_out', dest='datadir_out', type=str, help='Output directory for files')
parser.add_argument('--iter_real', dest='iter_real', type=float, help='Number of an iteration (to run the code)')
parser.add_argument('--cust_spec', dest='cust_spec', type=str, help='File with the customised spectrum')
parser.add_argument('--h', dest='h', type=float, default=1e-7, help='Amplitude of the dimensionless spectrum h of the string')
parser.add_argument('--fgw', dest='fgw', type=float, default=1e-8, help='Frequency of GW')
parser.add_argument('--sampler', dest='sampler', type=str, default="True", help='Run sampler')
args = parser.parse_args()

#input data directory
datadir = args.datadir
#number of frequency components
comp = args.comp
#number of iteration in mcmc
iter_num = args.iter_num
#output directory
datadir_out = args.datadir_out
#number of iter
iter_real = args.iter_real
#customised spectrum
add_spec = args.cust_spec
#slope of the red noise in timing residuals
fgw = args.fgw
#amplitude of the 
h = args.h
#run sampler
sampler = args.sampler

if add_spec:
    print("Using customised Omega_gw")
else:
    print("Using fgw={} and h={} for the cosmic string".format(fgw, h))
if sampler == "True":
    print("Sampling is on")
else:
    print("Sampling is not happening")

# fixed: robust directory creation
os.makedirs(os.path.join(datadir_out, "par"), exist_ok=True)
os.makedirs(os.path.join(datadir_out, "tim"), exist_ok=True)
    
parfiles = sorted(glob.glob(datadir + '*.par'))
Npsr = len(parfiles)
print(parfiles)

psrs = []

for ii in range(0,Npsr):

    # years of observations>
    psr = LT.fakepulsar(parfile=parfiles[ii],
            obstimes=np.arange(53000,53000+10*365.25,28.), toaerr=0.01)

    # We now remove the computed residuals from the TOAs, obtaining (in effect) a perfect realization of the deterministic timing model. The pulsar parameters will have changed somewhat, so `make_ideal` calls `fit()` on the pulsar object.
    LT.make_ideal(psr)

    #Generate white noise
    LT.add_efac(psr,efac=1.0)
    
#    add_rednoise(psr, 2e-14, 2.1)

    # add to list
    psrs.append(psr)
    
gwtheta = np.pi / 2
gwphi = np.pi / 2
h=h
fgw = fgw
phase0 = np.random.uniform(0, 2*np.pi)
psi = np.random.uniform(0, 2*np.pi)
inc = np.random.uniform(0, 2*np.pi) #np.pi / 4
pdist = 0.3*np.random.randn(len(psrs)) + 2

for ii in range(len(psrs)):
    LT.add_cstring(
        psrs[ii],
        gwtheta,
        gwphi,
        h,
        fgw,
        phase0,
        psi,
        pdist=pdist[ii],
        psrTerm=False,
        tref=0
        )

Amp = 40e-15
gamma = 13./3.
inj_params = {"gw_log10_A": np.log10(Amp), "gw_gamma": gamma}

#LT.createGWB(psrs, Amp=Amp, gam=gamma)

#Psrs = []
#for ii in psrs:
#   psr = Tempo2Pulsar(ii)
#   Psrs.append(psr)
    
os.system("mkdir " + datadir_out)

for Psr in psrs:
    Psr.savepar(datadir_out + "/par/" + Psr.name + ".par")
    Psr.savetim(datadir_out + "/tim/" + Psr.name + ".tim")
    T.purgetim(datadir_out + "/tim/" + Psr.name + ".tim")
    

#initialising enterprise
parfiles = sorted(glob.glob(datadir_out + '/par/*.par'))
timfiles = sorted(glob.glob(datadir_out + '/tim/*.tim'))


Psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t)
    Psrs.append(psr)

# find the maximum time span to set GW frequency sampling
Tspan = model_utils.get_tspan(Psrs)
#start = np.min([p.toas().min() * 86400 for p in Psrs])# - 86400
#stop = np.max([p.toas().max() * 86400 for p in Psrs])# + 86400

# duration of the signal
#Tspan = stop - start

# Here we build the signal model
# First we add the timing model
s = gp_signals.TimingModel()

# Then we add the white noise
# We use different white noise parameters for every backend/receiver combination
# The white noise parameters are held constant
efac = parameter.Constant(1.0)
log10_A = parameter.Uniform(-18, -13)('gw_log10_A')
gamma = parameter.Constant(0.001)('gw_gamma')

pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
s += white_signals.MeasurementNoise(efac=efac)

# Finally, we add the common red noise, which is modeled as a Fourier series with 30 frequency components
# The common red noise has a power-law PSD with spectral index of 4.33
#s += gp_signals.FourierBasisGP(pl, fmin=fgw, fmax=fgw+0.5/Tspan, components=1, name="gw", Tspan=Tspan)
s += gp_signals.FourierBasisGP(pl, fmin=fgw, fmax=fgw+0.5/Tspan, components=1, name="gw", Tspan=Tspan)

#s += blocks.red_noise_block(psd='spectrum', prior='log-uniform', components=30)

# We set up the PTA object using the signal we defined above and the pulsars
pta = signal_base.PTA([s(p) for p in Psrs])

def run_sampler(pta, iter_num, outdir = ''):

    N = int(iter_num)                                    # number of samples
    x0 = np.hstack([p.sample() for p in pta.params])
    ndim = len(x0)                                  # number of dimensions
    print('x0 =', x0)

    # initial jump covariance matrix
    cov = np.diag(np.ones(ndim) * 0.01**2)
    
    #initialize the sampler object
    sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, outDir=outdir, resume=False)
    
    # additional jumps
    jp = sp.JumpProposal(pta)
    sampler.addProposalToCycle(jp.draw_from_prior, 5)
    
    sel_sig = ["rn", "red_noise", "dm_gp", "fcn", "chrom-rn", "srn", "dm_srn", "freechrom-srn", "chrom-srn",
                        "dm-expd", "freechrom-expd", "chrom-expd",
                        "dm-y", "freechrom-y", "chrom-y",
                        "gw"]
    for s in sel_sig:
        if any([s in p for p in pta.param_names]):
            #pnames = [p.name for p in pta.params if s in p.name]
            #print('Adding %s prior draws with parameters :'%s, pnames, '\n')
            print('Adding %s prior draws.'%s)
            sampler.addProposalToCycle(jp.draw_from_par_prior(s), 10)

        
    sampler.sample(x0, N, SCAMweight=40, AMweight=25, DEweight=55) # these weights relate to frequency of jumps
    # write a list of the parameters to a text file
    # and a list of the parameter groupings used
    #filename = outdir + '/params.txt'
    #np.savetxt(filename,list(map(str, pta.param_names)), fmt='%s')
    
    lfcore = Core(chaindir=datadir_out, params=pta.param_names)
    lfcore.save(datadir_out + 'chain.core')
    
    return None

print(pta.params)

if sampler == "True":
    run_sampler(pta, iter_num, datadir_out)


chainname = 'chain_1'
chain = np.loadtxt(datadir_out + chainname + '.txt')

burn = int(0.3*chain.shape[0])

corner.corner(chain[burn:,-5],
                      bins =30,
                      plot_datapoints=False, plot_density=True, 
                      plot_contours=False,fill_contours=False,
                      show_titles = True, use_math_text=True, verbose=True)
#plt.ylim(-11, -3)
#plt.show()
plt.savefig(datadir_out + "corner.png", dpi=300)
#plt.clf()


#calculating os
lfcore = Core(corepath=datadir_out + 'chain.core')

os_obj = OptimalStatistic(Psrs, pta=pta, gwb_name='gw', core=lfcore, orfs=['hd'])


# If params=None, then DEFIANT will use maximum likelihood values from os_obj.lfcore
out = os_obj.compute_OS(return_pair_vals=True)

plt.clf()

plt.plot(out['xi'], out['rho']/out["A2"], ".")
plt.ylim(-1, 2)
plt.savefig(datadir_out + "indiv_correlation.png", dpi=300)

plt.clf()

out = os_obj.compute_OS()
xi,rho,sig,C,A2,A2s,idx = (out[k] for k in ['xi','rho','sig','C','A2','A2s','idx'])
defplot.create_correlation_plot(xi,rho,sig,C,A2,A2s, bins=10,)
#xi_range = np.linspace(0,np.pi,100)[1:] # Define our range of pulsar separations
#plt.plot(xi_range,10**(2*inj_params['gw_log10_A'])*hd_mod,'--k',label='Injected')

plt.savefig(datadir_out + "aver_correlation.png", dpi=300)

