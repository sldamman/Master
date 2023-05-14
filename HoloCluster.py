import numpy as np, matplotlib.pyplot as plt, xarray as xr, netCDF4 as nc
from sklearn import cluster
import sklearn as sk
import PIL, tqdm, os
from scipy import stats
#from matplotlib.cm import get_cmap
#from cartopy.feature import NaturalEarthFeature
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from joblib.externals.loky import set_loky_pickler
import datetime
from pylab import cm
from numpy import matlib
import hdbscan
import dill


class HoloCluster:
    '''
    This class initiates and runs the full KS-clustering algorithm as described in
    the thesis -On the Homogeneity of Arctic Mixed-Phase Clouds- 
    '''
    def __init__(self, data, droplet_cutoff_pct=0.7, n_jobs=10, merge=None, cutoff=None):
        if type(data) == str:
            self.data = xr.open_dataset(data)
        else:
            self.data = data
        
        try:
            self._datenum_to_datetime()
        except:
            print('Dataset already on datetime format')
    
        if merge is not None:
            self.data = self._merge(self.data, merge)

        self.meanN = float(self.data.Water_totalCount.mean())
        self.dcp = droplet_cutoff_pct
        self.n_jobs = n_jobs
        self.n_droplets = [cutoff if cutoff is not None else np.ceil(self.dcp * self.meanN)][0]

        self._cutoff(cutoff=cutoff)

        self.n_holo = len(self.data.time.to_numpy())
        self.diameter_bins = data.diameter.to_numpy() * 1000000
        self.number_per_bin = (self.data.Water_PSDnoNorm * self.data.Total_volume * 100**3).to_numpy().T.astype(int)
        if self.number_per_bin.shape[1] == self.n_holo:
            self.number_per_bin = self.number_per_bin.T
        self.diameter_observations = self._calc_diameter_observations()
        self.KS_matrix = None
        self.current_fit_method = None
        self.algorithm = None
    
    def compute_KS(self, ensemble_members=10, n_jobs=None, fit_method='each', 
                    binary=False, gamma_samples=None, seed=None):
        '''Create KS matrix'''
        print('Generating ensemble members')
        
        method = 'exact'
        if self.n_droplets > 1000:
            method = 'asymp'
        if gamma_samples is not None:
            if gamma_samples > 1000:
                method = 'asymp'

        self.ensemble = self._create_ensemble(
            ensemble_members=ensemble_members, 
            fit_method=fit_method, 
            gamma_samples=gamma_samples, 
            seed=seed)

        if n_jobs is None:
            n_jobs = self.n_jobs

            
        print('Computing KS matrix')
        if n_jobs > 0:
            with tqdm_joblib('Calculating KS matrix', total=self.n_holo) as progress_bar:
                self.KS_matrix = np.array(Parallel(n_jobs=n_jobs, backend='loky') \
                            (delayed(self._KS_row)(row_number, self.ensemble, binary=binary, method=method) for row_number in range(self.n_holo)))
        else:
            self.KS_matrix = np.array([self._KS_row(row_number, self.ensemble, binary=binary, method=method) for row_number in tqdm.tqdm(range(self.n_holo))])
        self.KS_matrix += self.KS_matrix.T - np.diag(np.diag(self.KS_matrix))
        self.current_fit_method = fit_method
        self.ensemble_members = ensemble_members
    
    def plot_KS(self, cmap=None, ax=None, add_colorbar=True, alpha=1):
        return self.plot_matrix(self.KS_matrix, cmap=cmap, ax=ax, 
                                add_colorbar=True, title='KS-matrix', alpha=alpha)
    
    def set_KS(self, KS):
        self.KS_matrix = KS

    def cluster(self, distmat=None, eps=0, min_samples=None, verbose=True, algorithm='dbscan',
                min_cluster_size=50, cluster_selection_method='leaf', sortby='size'):
        '''Find clusters with DBSCAN'''
        if distmat is None:
            if self.KS_matrix is None:
                distmat = self.compute_KS()
            else:
                distmat = self.KS_matrix

        if eps != 0: # eps is specified by user
            if type(eps) == float:
                eps_to_test = [eps]
            else:
                eps_to_test = eps
        elif algorithm == 'dbscan': # test different eps by default
            eps_to_test = np.arange(0.01, 0.5, 0.001)

        if min_samples is not None: # min_samples is specified by user
            if type(min_samples) == int:
                min_samples_to_test = [min_samples]
            else:
                min_samples_to_test = min_samples
        else: # test different min_samples by default
            min_samples_to_test = np.arange(10, 30, 1)
        
        if algorithm == 'dbscan':
            n_classes = -1; best_eps = -1; best_min_samples = -1; keep = None; noise = -1
            for eps in eps_to_test:
                for ms in min_samples_to_test:
                    results = cluster.DBSCAN(
                        eps=eps, 
                        min_samples=ms, 
                        metric='precomputed'
                    ).fit_predict(distmat)
                        
                    if self._count_classes(results) > n_classes:
                        n_classes = self._count_classes(results)
                        best_result = results
                        best_eps = eps
                        best_min_samples = ms
                        noise = len(np.where(results < 0)[0])
                        
                    elif self._count_classes(results) == n_classes and len(np.where(results < 0)[0]) < noise:
                        n_classes = self._count_classes(results)
                        best_result = results
                        best_eps = eps
                        best_min_samples = ms
                        noise = len(np.where(results < 0)[0])
            self.classes = best_result
            if sortby == 'size':
                self.sort_classes()
            elif sortby == 'ice':
                self.sort_classes('ice')
            if verbose: 
                if len(np.where(self.classes < 0)[0]) / len(self.classes) == 1:
                    print(f'100% Noise with eps={best_eps:.4f} and min_samples={best_min_samples}')
                else:
                    print(f'Found {n_classes} classes with eps={best_eps:.4f} and min_samples={best_min_samples}')
                    string = ''
                    for i in range(n_classes):
                        string += f'Class {i}: {len(np.where(self.classes==i)[0])} | '
                    print(string[:-2])
                    print(f'Noise occurence: {len(np.where(self.classes < 0)[0]) / len(self.classes) * 100:.2f}%')
        else:
            best_result = hdbscan.HDBSCAN(
                cluster_selection_epsilon=eps,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples, 
                cluster_selection_method=cluster_selection_method,
                metric='precomputed',
                allow_single_cluster=True
            ).fit_predict(distmat)

            self.classes = best_result
            if sortby == 'size':
                self.sort_classes()
            elif sortby == 'ice':
                self.sort_classes('ice')

            if verbose: 
                if len(np.where(self.classes < 0)[0]) / len(self.classes) == 1:
                    print(f'100% Noise with min_cluster_size={min_cluster_size} and min_samples={min_samples}')
                else:
                    n_classes = self._count_classes(self.classes, exclude_noise=True)
                    print(f'Found {n_classes} classes with min_cluster_size={min_cluster_size} and min_samples={min_samples}')
                    string = ''
                    for i in range(n_classes):
                        string += f'Class {i}: {len(np.where(self.classes==i)[0])} | '
                    print(string[:-2])
                    print(f'Noise occurence: {len(np.where(self.classes < 0)[0]) / len(self.classes) * 100:.2f}%')
        
        self.algorithm = algorithm
        #self.class_colors = cm.get_cmap('jet', len(np.unique(np.where(self.classes, self.classes, 0))))
        self.class_colors = cm.get_cmap('jet', self._count_classes(self.classes, exclude_noise=True))
        #self.class_colors_with_noise = cm.get_cmap('jet', len(np.unique(self.classes)))
        self.class_colors_with_noise = cm.get_cmap('jet', self._count_classes(self.classes, exclude_noise=False))
        self.clustered_holograms = self.get_classed_holos()
    
    def plot_clusters(self, cmap='jet', ax=None, exclude_noise=True, title=None, 
                      add_colorbar=True):
        try:
            rowwise = np.array([self.classes for _ in range(self.n_holo)])
        except AttributeError:
            print('Classifying KS matrix with default DBSCAN. \
                  For custom input to DBSCAN run HoloCluster.cluster().')
            self.cluster()
            rowwise = np.array([self.classes for _ in range(self.n_holo)])
        
        columnwise = rowwise.T
        classes_matrix = np.where(rowwise==columnwise, rowwise, np.nan)
        classes_matrix = [np.where(classes_matrix != -1, classes_matrix, np.nan) \
                                   if exclude_noise else classes_matrix][0]
        
        #cmap = cm.get_cmap(cmap, len(np.unique(self.classes)))
        
        if title is None:
            title = f'{self.algorithm}'


        vmin = [-0.5 if exclude_noise else -1.5][0]
        vmax = self._count_classes(self.classes, exclude_noise=True) - 0.5

        if vmax >= 0:
            img = self.plot_matrix(
                classes_matrix, vmin=vmin, vmax=vmax, add_colorbar=False, title='Detected clusters',
                cmap=[self.class_colors if exclude_noise else self.class_colors_with_noise][0], 
                ax=ax)
            
            if add_colorbar:
                cbar = plt.colorbar(img , 
                                    ticks=np.arange(int(vmin + 0.5), int(vmax - 0.5) + 1), 
                                    shrink=0.8)
                cbar.ax.set_yticklabels(np.arange(int(vmin + 0.5), int(vmax - 0.5) + 1) + 1)
            return img
        else:
            return self.plot_matrix(classes_matrix, add_colorbar=False, title='100\% Noise', ax=ax)
    
    def _merge(self, data, resolution='1s'):
        vars = ['Total_volume', 'Water_totalCount', 'Water_PSDnoNorm', 
                'Ice_totalCount', 'Ice_PSDnoNorm', 'instData_height_above_ground']
        grouped = data[vars]
        #grouped['time'] = self._datenum_to_datetime(data.time.to_numpy())
        grouped['instData_height_above_ground'] /= (float(resolution[0]) * 6)
        return grouped.resample(time=resolution).sum()

    def _cutoff(self, cutoff=None):
        if cutoff is None:
            cutoff = np.ceil(self.dcp * self.meanN)
        keep_idx = [i for i, number in enumerate(self.data.Water_totalCount.to_numpy()) \
                        if number > cutoff]
        self.data = self.data.isel(time=keep_idx)

    def _number_per_bin(self, phase='water'):
        '''Calculate the number of particles in each diameter bin'''
        if phase == 'water':
            return (self.data.Water_PSDnoNorm * self.data.Total_volume * 100**3).astype(int)
        elif phase == 'ice':
            return (self.data.Ice_PSDnoNorm * self.data.Total_volume * 100**3).astype(int)
        elif phase == 'both':
            return (self.data.Total_PSDnoNorm * self.data.Total_volume * 100**3).astype(int)
        else:
            raise NameError('Phase must be one of water, ice or both')
    
    def _repmat(self, t):
        '''Returns shape (n_droplets) for hologram number t'''
        dropletD = matlib.repmat(self.diameter_bins[0], self.number_per_bin[t][0], 1)
        for i in range(1, len(self.diameter_bins)):
            dropletD = np.concatenate([dropletD, matlib.repmat(self.diameter_bins[i], self.number_per_bin[t][i], 1)])
        return dropletD.flatten()
    
    def _calc_diameter_observations(self):
        return [self._repmat(t) for t in range(self.n_holo)]
    

    def _create_ensemble(self, ensemble_members=10, fit_method='each', gamma_samples='fixed', seed=None):
        '''Returns shape (n_holograms, n_ensemble_members, n_droplets)'''
        n_droplets = int(np.floor(self.n_droplets)) - 1
        ensemble = np.zeros((self.n_holo, ensemble_members, n_droplets))
        diameter_obs = self.diameter_observations
        if gamma_samples is None:
            gamma_samples = n_droplets

        seeds = [None for _ in range(self.n_holo * ensemble_members)]
        if seed is not None:
            seeds = np.arange(self.n_holo * ensemble_members)
        #if fit_method == 'mean':
        #    ensemble = np.zeros((self.n_holo, int(self.meanN * self.dcp)))
        def _resample(diameter_observation, n_samples, seed=None):
            '''Returns shape (n_droplets)'''
            return sk.utils.resample(diameter_observation, n_samples=n_samples, replace=False, random_state=seed)

        def _fit_gamma(obs, n_samples):
            '''Returns shape (n_droplets) of randomly drawn gamma fitted samples.
                Very strange behaviour in scipy.gamma.fit(). Fixing location parameter 
                with specifying floc=0, but known why. See https://github.com/scipy/scipy/issues/10212'''
            fit_alpha, fit_loc, fit_beta = stats.gamma.fit(obs, floc=5.99/1000000)
            return stats.gamma.rvs(fit_alpha, loc=fit_loc, scale=fit_beta, size=n_samples)
        
        def f(t):
            if fit_method == 'each':
                return np.array(
                    [_fit_gamma(
                        _resample(
                            diameter_obs[t], 
                            n_samples=n_droplets, 
                            seed=seeds[t*i]
                        ), 
                        gamma_samples
                    ) 
                for i in range(ensemble_members)]
            )
                
            elif fit_method == 'mean':
                return _fit_gamma(np.array([_resample(diameter_obs[t], n_samples=n_droplets, seed=seeds[t*i]) 
                                for i in range(ensemble_members)]).flatten(), n_samples=gamma_samples)
            elif fit_method == 'none':
                return np.array([_resample(diameter_obs[t], n_samples=n_droplets, seed=seeds[t*i]) 
                                for i in range(ensemble_members)])
            else:
                raise ValueError('Not a valid fit method. Must be one of [each, mean, none].')
        
        if self.n_jobs > 0:
            with tqdm_joblib('Calculating KS matrix', total=self.n_holo) as progress_bar:
                return np.array(Parallel(n_jobs=self.n_jobs, backend='loky')(delayed(f)(t) for t in range(self.n_holo)))
        else:
            return np.array([f(t) for t in tqdm.tqdm(range(self.n_holo))])
    
    @staticmethod
    def _KS_row(row_number, ensemble, binary=False, method=None):
        '''Has to be static in order for job_lib to work'''
        row = np.zeros(len(ensemble))

        if len(ensemble.shape) == 2:
            primary_member = ensemble[row_number]
            for col_number, secondary_member in enumerate(ensemble[:row_number]):
                ks_result = stats.ks_2samp(primary_member, secondary_member)
                if binary:
                    row[col_number] = [0 if ks_result.pvalue > 0.05 else 1][0]
                else:
                    row[col_number] = [0 if ks_result.pvalue > 0.05 else ks_result.statistic][0]
        else:
            for ip, primary_member in enumerate(ensemble[row_number]):
                for col_number in range(row_number + 1):
                    for secondary_member in ensemble[col_number]:
                        ks_result = stats.ks_2samp(primary_member, secondary_member, method=method)
                        if binary:
                            row[col_number] += [0 if ks_result.pvalue > 0.05 else 1][0]
                        else:
                            row[col_number] += [0 if ks_result.pvalue > 0.05 else ks_result.statistic][0]
            row /= len(ensemble[row_number])**2
        return row
    
    def _datenum_to_datetime(self):
        """
        Convert Matlab datenum into Python datetime.
        :param datenum: Date in datenum format
        :return:        Datetime object corresponding to datenum.
        """
        datenum = self.data.time.to_numpy()
        if len(datenum.shape) == 0:
            days = datenum % 1
            self.data['time'] =  datetime.datetime.fromordinal(int(datenum)) \
                + datetime.timedelta(days=days) \
                - datetime.timedelta(days=366)
        else:
            dates = []
            for dn in datenum:
                days = dn % 1
                dates.append(datetime.datetime.fromordinal(int(dn)) \
                            + datetime.timedelta(days=days) \
                            - datetime.timedelta(days=366))
            self.data['time'] = np.array(dates)

    def _count_classes(self, cluster_results, exclude_noise=True):
        if exclude_noise:
            return len(np.unique(cluster_results)) - [1 if -1 in cluster_results else 0][0]
        else:
            return len(np.unique(cluster_results))
    
    def sort_classes(self, by='size', order='decreasing'):
        old_classes = self.classes
        if by == 'size':
            sortby = [len(np.where(old_classes == i)[0]) for i in range(self._count_classes(old_classes))]
        if by == 'ice':
            sortby = [float(c.Ice_totalCount.mean()) for c in self.get_classed_holos(exclude_noise=True)]

        sortby = np.argsort(sortby)
        if order == 'decreasing':
            sortby = np.flip(sortby)

        new_classes = np.zeros(len(old_classes)) - 1

        for new_i, old_i in enumerate(sortby):
            new_classes[np.where(old_classes == old_i)] = new_i

        self.classes = new_classes


    def get_classed_holos(self, exclude_noise=True):
        '''Return a list of xr.dataset objects grouped by DBSCAN result'''
        classed_holos = []
        start = [0 if exclude_noise else -1][0]
        for i in range(start, self._count_classes(self.classes)):
            classed_holos.append(self.data.isel(time=np.argwhere(self.classes == i).flatten()))
        return classed_holos
    
    def _fit_gamma(self, obs, n_samples):
        '''Returns shape (n_droplets) of randomly drawn gamma fitted samples'''
        fit_alpha, fit_loc, fit_beta = stats.gamma.fit(obs, floc=5.99/1000000)
        return stats.gamma.rvs(fit_alpha, loc=fit_loc, scale=fit_beta, size=n_samples)

    @staticmethod
    def plot_matrix(matrix, cmap=None, ax=None, title=None, vmin=None, vmax=None, 
                    show=False, add_colorbar=True, alpha=1):
        if cmap is None:
            cmap = cm.get_cmap('inferno', 20)
        if ax is None:
            plt.close('all')
            img = plt.imshow(matrix, cmap=cmap, interpolation='none', alpha=alpha,
                             vmin=vmin, vmax=vmax, filternorm=False, resample=False)#(X, Y, c=KS_matrix)
            plt.title = title
            if add_colorbar:
                plt.colorbar(img, shrink=0.8)
            if show:
                plt.show()
            return img
        else:
            img = ax.imshow(matrix, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax,
                            filternorm=False, resample=False, alpha=alpha)
            ax.set_title(title)
            if add_colorbar:
                plt.colorbar(img, ax=ax, shrink=0.8)
            return img
        
    def plot_class_distributions(self, disttype='cdf', fit=True, ax=None, cmap='jet', exclude_noise=False):
        if ax is None:
            plt.close('all')
            fig, ax = plt.subplots(1, 1)
        #cmap = cm.get_cmap(cmap, len(np.unique(self.classes)) +1)
        cmap=[self.class_colors if exclude_noise else self.class_colors_with_noise][0]
        obs = self.diameter_observations
        alpha_noise = [0 if exclude_noise else 0.1][0]
        if disttype == 'cdf':
            if fit:
                [ax.plot(np.sort(self._fit_gamma(obs[i], len(obs[i]))), 
                        np.arange(len(obs[i])) / (len(obs[i]) - 1), 
                        color=cmap(self.classes[i]), 
                        label=f'Class {self.classes[i]}', 
                        alpha=[0.5 if self.classes[i] >= 0 else alpha_noise][0]) 
                    for i in np.arange(len(self.classes))]

            else:
                [ax.plot(np.sort(obs[i]), np.arange(len(obs[i])) / (len(obs[i]) - 1), 
                          color=cmap(self.classes[i]), 
                          label=f'Class {self.classes[i]}', 
                          alpha=[0.5 if self.classes[i] >= 0 else alpha_noise][0]) 
                    for i in np.arange(len(self.classes))]
        else:
            if fit:
                [ax.hist(self._fit_gamma(obs[i], len(obs[i])), 
                        color=cmap(self.classes[i]), histtype='step', density=True) 
                    for i in range(len(self.classes))]
            else:
                [ax.hist(obs[i], color=cmap(self.classes[i]), histtype='step', density=True) 
                    for i in range(len(self.classes))]
            #[plt.plot(np.sort(obs[i]), color=cmap(self.classes[i] / 15), label=f'Class {self.classes[i]}', 
            #          alpha=[0.5 if self.classes[i] >= 0 else 0.3][0]) for i in np.arange(len(self.classes))]
        #plt.plot(np.sort(ensem_fit[2]), np.arange(len(ensem_fit[1])) / (len(ensem_fit[1]) - 1))
        ax.set_xlim(5, 45)
    
    
    def plot_mean_distributions(self, cmap='jet', ax=None, pdf=False):
        axs = ax
        if ax is None:
            plt.close('all')
            fig, axs = plt.subplots(1, 1)
        classed = self.get_classed_holos()
        cmap = self.class_colors

        if not pdf:
            [axs.plot(
                self.data.diameter.to_numpy()*1000000, 
                classed[i].Water_PSDnoNorm.mean(dim='time').to_numpy(), 
                color=cmap(i),
                label=f'Class {i + 1}'
            ) for i in range(len(classed))]

            axs.plot(self.data.diameter.to_numpy() * 1000000, 
                    self.data.Water_PSDnoNorm.mean(dim='time').to_numpy(),
                    color='k', label='Total', linewidth=4)
        
        else:
            [axs.plot(
                self.data.diameter.to_numpy()*1000000, 
                classed[i].Water_PSDnoNorm.mean(dim='time').to_numpy() / sum(classed[i].Water_PSDnoNorm.mean(dim='time').to_numpy()), 
                color=cmap(i),
                label=f'Class {i + 1}'
            ) for i in range(len(classed))]

            axs.plot(self.data.diameter.to_numpy() * 1000000, 
                    self.data.Water_PSDnoNorm.mean(dim='time').to_numpy() / sum(self.data.Water_PSDnoNorm.mean(dim='time').to_numpy()),
                    color='k', label='Total', linewidth=4)
        


        axs.legend(fontsize=12, frameon=False)
        axs.set_xlabel('Diameter [$\mu m$]', size=12)
        if not pdf:
            axs.set_ylabel('Concentration [cm$^{-3}$]', size=12)
        else:
            axs.set_ylabel('Probability density', size=12)
        axs.set_title('Mean distributions', size=20)
        axs.set_xlim(6, 50)
        if ax is None:
            plt.legend(fontsize=12, frameon=False)
            plt.xlabel('Diameter [$\mu$m]', size=12)
            plt.ylabel('Concentration [cm$^{-3}$]', size=12)
            plt.show()


    def _repmat2(self, t, N_per_bin, diameter_bins):
        '''Designed to work for classed holograms'''
        dropletD = matlib.repmat(diameter_bins[0], int(N_per_bin.isel(time=t, diameter=0)), 1)

        for i in range(1, len(diameter_bins)):
            dropletD = np.concatenate([dropletD, matlib.repmat(diameter_bins[i], 
                                                            int(N_per_bin.isel(time=t, diameter=i)), 1)])
        return dropletD.flatten()

    def _calc_diameter_observations2(self, data):
        '''Designed to work for classed holograms specifically for plot_gamma_parameters()'''
        N_per_bin = (data.Water_PSDnoNorm * data.Total_volume * 100**3).astype(int)
        diameter_bins = data.diameter.to_numpy()
        n_holo = len(data.time)
        return [self._repmat2(t, N_per_bin, diameter_bins) for t in range(n_holo)]

    def plot_gamma_parameters(self, ax=None, exclude_noise=False):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        for i, cluster in enumerate(self.get_classed_holos(exclude_noise=True)):
            
            diameters = self._calc_diameter_observations2(cluster)
            color = self.class_colors(i)
            
            
            shape, loc, scale = stats.gamma.fit(diameters[0], floc=5.99/1000000)
            ax.scatter(scale * 1000000, shape, color=color, label=f'Class {i+1}')
            for dat in diameters[1:]:
                shape, loc, scale = stats.gamma.fit(dat, floc=5.99/1000000)
                ax.scatter(scale * 1000000, shape, color=color)
        
        if not exclude_noise:
            diameters = self._calc_diameter_observations2(self.get_classed_holos(exclude_noise=False)[0])
            color = 'grey'
            
            shape, loc, scale = stats.gamma.fit(diameters[0], floc=5.99/1000000)
            ax.scatter(scale * 1000000, shape, color=color, alpha=0.3, label=f'Noise')
            for dat in diameters[1:]:
                shape, loc, scale = stats.gamma.fit(dat, floc=5.99/1000000)
                ax.scatter(scale * 1000000, shape, color=color, alpha=0.3)


        #total = np.concatenate(self._calc_diameter_observations2(self.data)).ravel()
        #shape, loc, scale = stats.gamma.fit(total, floc=0)

        #ax.scatter(scale * 1000000, shape, s=200, c='k', label='total')
        ax.legend(fontsize=12, frameon=False)
        ax.set_xlabel('Scale [$\mu m$]', size=12)
        ax.set_ylabel('Shape', size=12)
        #ax.set_title('Gamma parameters', size=20)


    def plot_summary(self, xlim=(5, 50), exclude_noise=True, suptitle=None, pdf=False, alpha=1,
                     cmap=None):
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        self.plot_KS(ax=ax[0, 0], add_colorbar=True, alpha=alpha, cmap=cmap)
        self.plot_clusters(ax=ax[0, 1], add_colorbar=True)
        self.plot_mean_distributions(ax=ax[1, 0], pdf=pdf)
        self.plot_gamma_parameters(ax=ax[1, 1], exclude_noise=exclude_noise)

        ax[0, 0].set_title('KS-Matrix', size=18)
        ax[0, 1].set_title(self.algorithm.upper() + ' Clusters', size=18)
        ax[1, 0].set_title('Mean Size Distributions', size=18)
        ax[1, 1].set_title('Gamma Parameters', size=18)
        ax[1, 0].set_xlim(xlim[0], xlim[1])

        ax[0, 0].set_ylabel('Hologram \#', size=12)
        ax[0, 0].set_xlabel('Hologram \#', size=12)
        ax[0, 1].set_ylabel('Hologram \#', size=12)
        ax[0, 1].set_xlabel('Hologram \#', size=12)

        subplot_labels = ['a)', 'b)', 'c)', 'd)']
        for i, x in enumerate(ax.ravel()):
            x.text(x=0, y=1.04, s=subplot_labels[i], transform=x.transAxes,
                fontsize=16)
        plt.subplots_adjust(hspace=0.15, wspace=0.15)
        fig.suptitle(suptitle, size=22, y=0.94, x=0.45)

    def store_KS(self, filename=None):
        if filename is None:
            filename = datetime.datetime.now().strftime('%y%m%d_%H%M%S') + '_KS_'
            filename += str(self.ensemble_members)
        with open(filename, 'wb') as file:
            dill.dump(self.KS_matrix, file, protocol=-1)


