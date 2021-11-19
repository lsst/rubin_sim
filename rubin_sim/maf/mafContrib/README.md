# Metrics File Name: 211116_yso_3D_90b.py

# Script description 
This metrics computes the number of young stellar objects (YSO) with age t<10 Myrs and mass M>0.3 Msun.
Four Opsim surveys are considered. The total counts of YSOs are estimated for each OpSim, assuming to observe only with gri filters. 


## How to use


### 1. Getting the extinction map and the code for querying it.

- Download the map file from 
 [this link](http://www-personal.umd.umich.edu/~wiclarks/rubin/merged_ebv3d_nside64_defaults.fits.gz) 
 http://www-personal.umd.umich.edu/~wiclarks/rubin/merged_ebv3d_nside64_defaults.fits.gz 
 provided by Will Clarkson (in datalab, you can use `wget` to get the file). Extract the map (you can use `gunzip`) and be sure to put it in the same folder as this script`.

- Get the code to read the map with git: in the terminal, navigate to `root` and type


```
git clone https://github.com/willclarkson/rubinCadenceScratchWIC.git
```


** WARNING ** Until the relevant changes are implemented into Will Clarkson's repository, the version with the necessary code comes from Alessandro Mazzi's fork, so please run also

```
git clone https://github.com/Thalos12/rubinCadenceScratchWIC.git rubinCadenceScratchWIC_fork
```

then get into the newly created folder and type

```
git checkout distmag_pixels_subset_2
```

to get the branch that has the working code. The notebook will use the code from there. When the code is ready, I will update the README and the notebook, and the rubinCadenceScratchWIC_fork will have to be deleted.

### 2. Using the notebook.

Run the script

## Metrics results and Recommedation for Observing Strategy to SCOC

The number of YSOs detected with vary_gp_gpfrac1.00_v2.0_10yrs.db  is 10% larger than that obtained using the baseline_v2.0_10yrs.db. 
We strongly recommend to remove the constraint of the E(B-V) upper limit in the non-bulge GP and  to 
adopt the Opsim /sims_maf/fbs_2.0/vary_gp/vary_gp_gpfrac1.00_v2.0_10yrs.db to homogeneously map the Galactic Plane as well the Bulge.  

This is an important metric and we want to use the results to change our view of survey strategy
---

Many thanks to Will Clarkson (@willclarkson) for helping implementing the required functions.
Code written by Peter Yoachim, Loredana Prisinzano and Alessandro Mazzi

