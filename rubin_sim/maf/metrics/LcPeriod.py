#This metric use Gatspy (https://arxiv.org/abs/1502.01344)) to recover the period of the simulated temporal series. Produce a multipanel plot and return:
#best period, period_model-best_period/best_period and period_model-best_period/best_period*1/number of cycles 
import matplotlib.pyplot as plt 
from gatspy import periodic
import numpy as np
import os

def main(mv,LcTeoLSST,LcTeoLSST_noised,index_notsaturated,label,path3):
    #########################################################################
    #period range in the periodogram for the plot of the periodogram
    period_model=LcTeoLSST['period']    
    minper_plot=period_model-0.9*period_model
    maxper_plot=period_model+ 0.9*period_model
    #Step's choice
    periods = np.linspace(minper_plot, maxper_plot,10000)
    #period range for the optimization of the period's search with LombScargleMultiband
    minper_opt=period_model- 0.5*period_model
    maxper_opt=period_model+0.5*period_model
    #########################################################################

    colors = {'u': 'b','g': 'g','r': 'r',
              'i': 'purple',"z": 'y',"y": 'magenta'}

    #Figure with multiband periodograms
    fig = plt.figure(figsize=(10, 4))
    gs = plt.GridSpec(5, 2, left=0.07, right=0.95, bottom=0.15,
                          wspace=0.1, hspace=0.6)
    ax = [  fig.add_subplot(gs[:, 0]),
              fig.add_subplot(gs[:-2, 1]),
              fig.add_subplot(gs[-2:, 1])]
    ax[0].set_xlabel('Phase')
    ax[0].set_ylabel('Mag')
    ax[0].invert_yaxis()

    for filterName in mv['filter']:
        ax[0].scatter(LcTeoLSST['phase'+filterName][index_notsaturated['ind_notsaturated_'+filterName]],
                      LcTeoLSST_noised['mag'+filterName][index_notsaturated['ind_notsaturated_'+filterName]],
                      c=colors[filterName], label=filterName)
        
    model = periodic.NaiveMultiband(BaseModel=periodic.LombScargleFast)
    model.fit(LcTeoLSST_noised['time_all'][index_notsaturated['ind_notsaturated_all']],
              LcTeoLSST_noised['mag_all'][index_notsaturated['ind_notsaturated_all']],
              LcTeoLSST_noised['dmag_all'][index_notsaturated['ind_notsaturated_all']],
              np.asarray(mv['filter'][index_notsaturated['ind_notsaturated_all']]))
    P = model.scores(periods)
    ax[1].set_xlim(minper_plot, maxper_plot)
    ax[1].set_title('Standard Periodogram in Each Band', fontsize=12)
    ax[1].yaxis.set_major_formatter(plt.NullFormatter())
    ax[1].xaxis.set_major_formatter(plt.NullFormatter())
    ax[1].set_ylabel('power + offset')

    for i, band in enumerate('ugrizy'):
        n_temp=len((np.where(mv['filter'][index_notsaturated['ind_notsaturated_all']]== band))[0])
        if n_temp >= 1:
            offset = 5 - i
            ax[1].plot(periods, P[band] + offset, lw=1, c=colors[band])
            ax[1].text(0.89, 1 + offset, band, fontsize=10, ha='right', va='top')

    LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=0)
    LS_multi.fit(LcTeoLSST_noised['time_all'],LcTeoLSST_noised['mag_all'],LcTeoLSST_noised['dmag_all'], mv['filter'])
    P_multi = LS_multi.periodogram(periods)


    periodogram_noise=np.median(P_multi)
    periodogram_noise_mean=np.mean(P_multi)

    print('Noise level (median vs mean)')
    print(periodogram_noise,periodogram_noise_mean)

    fitLS_multi= periodic.LombScargleMultiband(fit_period=True)
    fitLS_multi.optimizer.period_range=(minper_opt, maxper_opt)
    fitLS_multi.fit(LcTeoLSST_noised['time_all'],LcTeoLSST_noised['mag_all'],LcTeoLSST_noised['dmag_all'], mv['filter'])
    best_per_temp=fitLS_multi.best_period


    tmin=min(LcTeoLSST_noised['time_all'])
    tmax=max(LcTeoLSST_noised['time_all']) 
    cicli=(tmax-tmin)/period_model

    diffper=best_per_temp-period_model
    diffper_abs=abs(best_per_temp-period_model)/period_model*100
    diffcicli=abs(best_per_temp-period_model)/period_model*1/cicli
    print(' Period of the model:')
    print(period_model)
    print(' Period found by Gatpy:')
    print(best_per_temp)
    print(' DeltaP/P (in perc):')
    print(diffper)
    print(' DeltaP/P*1/number of cycle:')
    print(diffcicli)

    ax[2].plot(periods, P_multi, lw=1, color='gray')

    ax[2].set_title('Multiband Periodogram', fontsize=12)
    ax[2].set_yticks([0, 0.5, 1.9])
    ax[2].set_ylim(0, 1.0)
    ax[2].set_xlim(minper_plot, maxper_plot)
        #ax[2].set_xlim(period_model-.01, period_model+.01)
        #ax[2].axvline(fitLS_multi.best_period,color='r');
    ax[2].axhline(periodogram_noise,color='r');
    ax[2].yaxis.set_major_formatter(plt.NullFormatter())
    ax[2].text((minper_plot+maxper_plot)/2.,0.83,'Best Period = %.5f days' % fitLS_multi.best_period,color='r')
    ax[2].text((minper_plot+maxper_plot)/2.,0.43,'Noise = %.5f' % periodogram_noise,color='b')
    ax[2].text((minper_plot+maxper_plot)/2.,0.63,'|$\Delta$P| = %.12f days' % diffper_abs,color='r')
    ax[2].set_xlabel('Period (days)')
    ax[2].set_ylabel('power')
    
    if os.path.exists(path3):
        plt.savefig(str(path3)+'/Period_'+label+'.pdf')
    else:
        os.makedirs(path)
        plt.savefig(str(path3)+'/Period_'+label+'.pdf')
        
    plt.close
    
    return best_per_temp,diffper,diffper_abs
