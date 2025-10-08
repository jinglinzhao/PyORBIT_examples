"""
Combined RV and Activity Plotting Notebook
Easily customizable for different datasets
"""

# ============================================================================
# CONFIGURATION SECTION - CUSTOMIZE HERE
# ============================================================================

# Configuration presets - change this to switch between different setups
# CONFIG_PRESET = 'DS4_2p_4activity_indi'  # Options: 'iaras_DS1', 'ESSP_multi', 'HD189567'
# CONFIG_PRESET = 'DS4_1p_4activity_indi'
# CONFIG_PRESET = 'DS2_1p_4activity_indi'
CONFIG_PRESET = 'DS1_1p_4activity_indi'

# Planet configuration
# Specify which planets to plot (e.g., ['b'], ['b', 'c'], ['c'])
PLANETS_TO_PLOT = ['b']
# PLANETS_TO_PLOT = ['b', 'c']


# Configuration dictionary containing all preset options
CONFIGS = {



    'DS1_1p_4activity_indi': {
        'dir_base': '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_jz/single/DS1/DS1_1p/DS1_1p_4activity_indi/',
        'dir_mods': 'DS1_1p_4activity_indi/',
        'dir_plot': 'emcee_plot/model_files/',
        'filename': 'DS1_1p_4activity_indi',
        'datasets_list': ['RVdata'],
        'datasets_labels': {'RVdata': 'RV'},
        'activity_model': 'gp_multidimensional',
        'activity_list': ['BISdata', 'FWHMdata', 'CaIIdata', 'Halphadata'],
        'activity_labels': {'BISdata': 'BIS', 'FWHMdata': 'FWHM', 'CaIIdata': 'CaII', 'Halphadata': 'Halpha'},
        'activity_dict': {
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [59329, 59433],
        },
        'full_dict': {
            'reference_planet': 'b',
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [59329, 59433],
        }
    },

    'DS2_1p_4activity_indi': {
        'dir_base': '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_jz/single/DS2/DS2_1p/DS2_1p_4activity_indi/',
        'dir_mods': 'DS2_1p_4activity_indi/',
        'dir_plot': 'emcee_plot/model_files/',
        'filename': 'DS2_1p_4activity_indi',
        'datasets_list': ['RVdata'],
        'datasets_labels': {'RVdata': 'RV'},
        'activity_model': 'gp_multidimensional',
        'activity_list': ['BISdata', 'FWHMdata', 'CaIIdata', 'Halphadata'],
        'activity_labels': {'BISdata': 'BIS', 'FWHMdata': 'FWHM', 'CaIIdata': 'CaII', 'Halphadata': 'Halpha'},
        'activity_dict': {
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [59329, 59433],
        },
        'full_dict': {
            'reference_planet': 'b',
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [59329, 59433],
        }
    },

    'DS4_1p_4activity_indi': {
        'dir_base': '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_jz/single/DS4/DS4_1p/DS4_1p_4activity_indi/',
        'dir_mods': 'DS4_1p_4activity_indi/',
        'dir_plot': 'emcee_plot/model_files/',
        'filename': 'DS4_1p_4activity_indi',
        'datasets_list': ['RVdata'],
        'datasets_labels': {'RVdata': 'RV'},
        'activity_model': 'gp_multidimensional',
        'activity_list': ['BISdata', 'FWHMdata', 'CaIIdata', 'Halphadata'],
        'activity_labels': {'BISdata': 'BIS', 'FWHMdata': 'FWHM', 'CaIIdata': 'CaII', 'Halphadata': 'Halpha'},
        'activity_dict': {
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [59329, 59433],
        },
        'full_dict': {
            'reference_planet': 'b',
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [59329, 59433],
        }
    },



    'DS4_2p_4activity_indi': {
        'dir_base': '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/results_jz/single/DS4/DS4_2p/DS4_2p_4activity_indi/',
        'dir_mods': 'DS4_2p_4activity_indi/',
        'dir_plot': 'emcee_plot/model_files/',
        'filename': 'DS4_2p_4activity_indi',
        'datasets_list': ['RVdata'],
        'datasets_labels': {'RVdata': 'RV'},
        'activity_model': 'gp_multidimensional',
        'activity_list': ['BISdata', 'FWHMdata', 'CaIIdata', 'Halphadata'],
        'activity_labels': {'BISdata': 'BIS', 'FWHMdata': 'FWHM', 'CaIIdata': 'CaII', 'Halphadata': 'Halpha'},
        'activity_dict': {
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [59329, 59433],
        },
        'full_dict': {
            'reference_planet': 'b',
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [59329, 59433],
        }
    },






    'iaras_DS1': {
        'dir_base': '/work2/lbuc/jzhao/PyORBIT_ESSP/ESSP/iaras/DS1/DS1_3p/DS1_3p_2activity_indi/',
        'dir_mods': 'DS1_3p_2activity_indi/',
        'dir_plot': 'emcee_plot/model_files/',
        'filename': 'iaras_DS1_3p_2activity_indi',
        'datasets_list': ['RVdata'],
        'datasets_labels': {'RVdata': 'RV'},
        'activity_model': 'gp_multidimensional',
        'activity_list': ['BISdata', 'FWHMdata'],
        'activity_labels': {'BISdata': 'BIS', 'FWHMdata': 'FWHM'},
        'activity_dict': {
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [59329, 59433],
        },
        'full_dict': {
            'reference_planet': 'b',
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [59329, 59433],
        }
    },
    
    'ESSP_multi': {
        'dir_base': './',
        'dir_mods': 'ESSP_gp_HARPSN_EXPRES_NEID_HARPS_poly_cpu/',
        'dir_plot': 'emcee_plot/model_files/',
        'filename': 'ESSP_gp_HARPSN_EXPRES_NEID_HARPS_poly_cpu',
        'datasets_list': ['ESSP_HARPSN', 'ESSP_EXPRES', 'ESSP_NEID', 'ESSP_HARPS'],
        'datasets_labels': {
            'ESSP_HARPSN': 'HARPSN', 
            'ESSP_EXPRES': 'EXPRES', 
            'ESSP_NEID': 'NEID', 
            'ESSP_HARPS': 'HARPS'
        },
        'activity_model': 'gp_multidimensional',
        'activity_list': ['ESSP_BIS_HARPSN', 'ESSP_BIS_EXPRES', 'ESSP_BIS_NEID', 'ESSP_BIS_HARPS'],
        'activity_labels': {
            'ESSP_BIS_HARPSN': 'BIS_HARPSN', 
            'ESSP_BIS_EXPRES': 'BIS_EXPRES', 
            'ESSP_BIS_NEID': 'BIS_NEID', 
            'ESSP_BIS_HARPS': 'BIS_HARPS'
        },
        'activity_dict': {
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [59332, 59360],
        },
        'full_dict': {
            'reference_planet': 'b',
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [59332, 59360],
        }
    },
    
    'HD189567': {
        'dir_base': './',
        'dir_mods': 'HD189567_3p_run7/',
        'dir_plot': 'emcee_plot/model_files/',
        'filename': 'HD189567_3p_run7',
        'datasets_list': ['RVdata'],
        'datasets_labels': {'RVdata': 'RV'},
        'activity_model': 'gp_multidimensional',
        'activity_list': ['BISdata', 'FWHMdata'],
        'activity_labels': {
            'BISdata': 'BIS',
            'FWHMdata': 'FWHM',
        },
        'activity_dict': {
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [2455480., 2460211.522268],
        },
        'full_dict': {
            'reference_planet': 'b',
            'limits_full_x': [-0.25, 1.25],
            'limits_bjd': [2455480., 2460211.522268],
        }
    }
}

# Load the selected configuration
config = CONFIGS[CONFIG_PRESET]

# Extract configuration variables
dir_base = config['dir_base']
dir_mods = config['dir_mods']
dir_plot = config['dir_plot']
filename = config['filename']
datasets_list = config['datasets_list']
datasets_labels = config['datasets_labels']
activity_model = config['activity_model']
activity_list = config['activity_list']
activity_labels = config['activity_labels']
activity_dict = config['activity_dict']
full_dict = config['full_dict']

# Plotting parameters
font_label = 12
dot_size = 18
figsize = (10, 7)

# ============================================================================
# END OF CONFIGURATION SECTION
# ============================================================================


# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
import collections
import matplotlib.gridspec as gridspec
import matplotlib
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import itertools


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def plots_in_grid():
    """
    Create a 2-panel grid layout for main plot and residuals
    """
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.0, 1.0])
    gs.update(left=0.2, right=0.95, bottom=0.08, top=0.93, wspace=0.02, hspace=0.03)
    
    ax_0 = plt.subplot(gs[0])
    ax_1 = plt.subplot(gs[1])

    # Adding minor ticks only to x axis
    minorLocator = AutoMinorLocator()
    ax_0.xaxis.set_minor_locator(minorLocator)
    ax_1.xaxis.set_minor_locator(minorLocator)

    # Disabling the offset on top of the plot
    ax_0.ticklabel_format(useOffset=False, style='plain')
    ax_1.ticklabel_format(useOffset=False, style='plain')
    
    return ax_0, ax_1


# ============================================================================
# LOAD DATA AND SETUP PLANET DICTIONARY
# ============================================================================
summary_percentiles_parameters = pickle.load(
    open(dir_base + dir_mods + 'emcee_plot/dictionaries/summary_percentiles_parameters.p', 'rb')
)
summary_percentiles_derived = pickle.load(
    open(dir_base + dir_mods + 'emcee_plot/dictionaries/summary_percentiles_derived.p', 'rb')
)

planet_dict = collections.OrderedDict()

# Configure planets based on PLANETS_TO_PLOT list
for planet_name in PLANETS_TO_PLOT:
    if planet_name in summary_percentiles_parameters:
        planet_dict[planet_name] = {
            'P': summary_percentiles_parameters[planet_name]['P'][3],
            'limits_folded_x': [-0.25, 1.25],
            'transit_folded': False,
            'K_error_1sigma': (summary_percentiles_parameters[planet_name]['K'][4] - 
                               summary_percentiles_parameters[planet_name]['K'][2]) / 2,
            'K_error_2sigma': (summary_percentiles_parameters[planet_name]['K'][5] - 
                               summary_percentiles_parameters[planet_name]['K'][1]) / 2,
            'K_error_3sigma': (summary_percentiles_parameters[planet_name]['K'][6] - 
                               summary_percentiles_parameters[planet_name]['K'][0]) / 2,
        }
        print(f"Planet {planet_name} configured successfully")
    else:
        print(f"Warning: Planet {planet_name} not found in data files")

print("Planet dictionary loaded:")
print(planet_dict)


# ============================================================================
# PLOT 1: FOLDED RV PLOTS FOR EACH PLANET
# ============================================================================
print("\n" + "="*60)
print("GENERATING FOLDED RV PLOTS")
print("="*60)

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams.update({'font.size': font_label})

# Assign colors to datasets
dataset_colors = {}
for i, dataset in enumerate(datasets_list):
    dataset_colors[dataset] = 'C{}'.format(i)

for key_name, key_val in planet_dict.items():
    print(f"\nProcessing planet {key_name}...")
    
    RV_kep = np.genfromtxt(
        dir_base + dir_mods + dir_plot + 'RV_planet_' + key_name + '_kep.dat', 
        skip_header=1
    )

    if key_val.get('transit_folded', True):
        RV_pha = np.genfromtxt(
            dir_base + dir_mods + dir_plot + 'RV_planet_' + key_name + '_Tcf.dat', 
            skip_header=1
        )
    else:
        RV_pha = np.genfromtxt(
            dir_base + dir_mods + dir_plot + 'RV_planet_' + key_name + '_pha.dat', 
            skip_header=1
        )

    fig = plt.figure(figsize=figsize)
    ax_0, ax_1 = plots_in_grid()
    
    # Error bands
    K_error_1sigma = key_val['K_error_1sigma']
    K_error_2sigma = key_val['K_error_2sigma']
    K_error_3sigma = key_val['K_error_3sigma']
    K_rvs = np.amax(RV_pha[:, 1])
    RV_unitary = RV_pha[:, 1] / K_rvs
    
    ax_0.fill_between(RV_pha[:, 0], RV_unitary*(K_rvs-K_error_1sigma), 
                      y2=RV_unitary*(K_rvs+K_error_1sigma), 
                      alpha=0.10, color='black', zorder=0)
    ax_0.fill_between(RV_pha[:, 0], RV_unitary*(K_rvs-K_error_2sigma), 
                      y2=RV_unitary*(K_rvs+K_error_2sigma), 
                      alpha=0.10, color='black', zorder=0)
    ax_0.fill_between(RV_pha[:, 0], RV_unitary*(K_rvs-K_error_3sigma), 
                      y2=RV_unitary*(K_rvs+K_error_3sigma), 
                      alpha=0.10, color='black', zorder=0)
    
    # Plot model
    ax_0.plot(RV_pha[:, 0]-1, RV_pha[:, 1], color='k', linestyle='-', 
              zorder=2, label='RV model')
    ax_0.plot(RV_pha[:, 0]+1, RV_pha[:, 1], color='k', linestyle='-', zorder=2)

    # Plot data for each dataset
    for n_dataset, dataset in enumerate(datasets_list):
        color = dataset_colors[dataset]

        RV_mod = np.genfromtxt(
            dir_base + dir_mods + dir_plot + dataset + '_radial_velocities_' + key_name + '.dat', 
            skip_header=1
        )
        error = np.sqrt(RV_mod[:, 9]**2 + RV_mod[:, 12]**2)
    
        if key_val.get('transit_folded', False):
            rv_phase = RV_mod[:, 1] / planet_dict[key_name]['P']
        else:
            rv_phase = RV_mod[:, 2]
        
        # Main points
        ax_0.errorbar(rv_phase, RV_mod[:, 8], yerr=error, color='black', 
                      markersize=0, alpha=0.25, fmt='o', zorder=0)
        ax_0.scatter(rv_phase, RV_mod[:, 8], c=color, s=dot_size, 
                    zorder=20-n_dataset, alpha=1.0, label=datasets_labels[dataset])

        # Points at phase-1
        ax_0.errorbar(rv_phase-1, RV_mod[:, 8], yerr=error, color='black', 
                      markersize=0, alpha=0.25, fmt='o', zorder=0)
        ax_0.scatter(rv_phase-1, RV_mod[:, 8], c='gray', s=dot_size, zorder=20, alpha=1.0)

        # Points at phase+1
        ax_0.errorbar(rv_phase+1, RV_mod[:, 8], yerr=error, color='black', 
                      markersize=0, alpha=0.25, fmt='o', zorder=0)
        ax_0.scatter(rv_phase+1, RV_mod[:, 8], c='gray', s=dot_size, zorder=20, alpha=1.0)

        # Residuals
        ax_1.errorbar(rv_phase, RV_mod[:, 10], yerr=error, color='black', 
                      markersize=0, alpha=0.25, fmt='o', zorder=1)
        ax_1.scatter(rv_phase, RV_mod[:, 10], c=color, s=dot_size, 
                    zorder=20-n_dataset, alpha=1.0)

        # Residuals at phase-1
        ax_1.errorbar(rv_phase-1, RV_mod[:, 10], yerr=error, color='black', 
                      markersize=0, alpha=0.25, fmt='o', zorder=1)
        ax_1.scatter(rv_phase-1, RV_mod[:, 10], c='gray', s=dot_size, zorder=20, alpha=1.0)

        # Residuals at phase+1
        ax_1.errorbar(rv_phase+1, RV_mod[:, 10], yerr=error, color='black', 
                      markersize=0, alpha=0.25, fmt='o', zorder=1)
        ax_1.scatter(rv_phase+1, RV_mod[:, 10], c='gray', s=dot_size, zorder=20, alpha=1.0)

    # Residuals zero line and error bands
    ax_1.axhline(0.000, c='k', zorder=3)
    ax_1.fill_between(RV_pha[:, 0], -K_error_1sigma, K_error_1sigma, 
                      alpha=0.10, color='black', zorder=0)
    ax_1.fill_between(RV_pha[:, 0], -K_error_2sigma, K_error_2sigma, 
                      alpha=0.10, color='black', zorder=0)
    ax_1.fill_between(RV_pha[:, 0], -K_error_3sigma, K_error_3sigma, 
                      alpha=0.10, color='black', zorder=0)

    # Set limits
    if key_val.get('limits_folded_x', False):
        ax_0.set_xlim(key_val['limits_folded_x'][0], key_val['limits_folded_x'][1])
        ax_1.set_xlim(key_val['limits_folded_x'][0], key_val['limits_folded_x'][1])
    if key_val.get('limits_folded_y', False):
        ax_0.set_ylim(key_val['limits_folded_y'][0], key_val['limits_folded_y'][1])
    if key_val.get('limits_residuals_y', False):
        ax_1.set_ylim(key_val['limits_residuals_y'][0], key_val['limits_residuals_y'][1])

    # Vertical lines
    if key_val.get('transit_folded', False):
        print(f'Planet {key_name}, RV curve folded around the transit time')
        ax_0.axvline(-0.500, c='k', zorder=3, alpha=0.5, linestyle='--')
        ax_0.axvline(0.500, c='k', zorder=3, alpha=0.5, linestyle='--')
        ax_1.axvline(-0.500, c='k', zorder=3, alpha=0.5, linestyle='--')
        ax_1.axvline(0.500, c='k', zorder=3, alpha=0.5, linestyle='--')
    else:
        print(f'Planet {key_name}, RV curve folded around the reference time')
        ax_0.axvline(0.00, c='k', zorder=3, alpha=0.5, linestyle='--')
        ax_0.axvline(1.00, c='k', zorder=3, alpha=0.5, linestyle='--')
        ax_1.axvline(0.00, c='k', zorder=3, alpha=0.5, linestyle='--')
        ax_1.axvline(1.00, c='k', zorder=3, alpha=0.5, linestyle='--')

    # Formatting
    ax_0.axes.get_xaxis().set_ticks([])
    ax_0.yaxis.set_major_locator(MultipleLocator(5))
    ax_0.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax_0.yaxis.set_minor_locator(MultipleLocator(1))
    ax_1.yaxis.set_major_locator(MultipleLocator(5))
    ax_1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax_1.yaxis.set_minor_locator(MultipleLocator(1))
    
    ax_0.set_ylabel('RV [m/s]')
    ax_1.set_xlabel('Orbital Phase')
    ax_1.set_ylabel('Residuals [m/s]')
    ax_0.legend(framealpha=1.0, loc='lower left')
    
    plot_filename = filename + '_' + key_name + '_folded.png'
    print(f'Folded plot for planet {key_name} saved to: {plot_filename}')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')


# ============================================================================
# PLOT 2: FULL RV TIME SERIES
# ============================================================================
print("\n" + "="*60)
print("GENERATING FULL RV TIME SERIES PLOT")
print("="*60)

key_name = full_dict['reference_planet']

fig = plt.figure(figsize=figsize)
ax_0, ax_1 = plots_in_grid()

for n_dataset, dataset in enumerate(datasets_list):
    print(f"Processing dataset: {dataset}")
    default_color = 'C' + repr(n_dataset)

    RV_full = np.genfromtxt(
        dir_base + dir_mods + dir_plot + dataset + '_full.dat', 
        skip_header=1
    )
    ax_0.plot(RV_full[:, 0]-2450000, RV_full[:, 1], color='k', 
              linestyle='-', zorder=2, lw=0.1)

    RV_mod = np.genfromtxt(
        dir_base + dir_mods + dir_plot + dataset + '_radial_velocities_' + key_name + '.dat', 
        skip_header=1
    )
    error = np.sqrt(RV_mod[:, 9]**2 + RV_mod[:, 12]**2)

    ax_0.errorbar(RV_mod[:, 0]-2450000, RV_mod[:, 7]-RV_mod[:, 5], yerr=error, 
                  color='black', markersize=0, alpha=0.25, fmt='o', zorder=0)
    ax_0.scatter(RV_mod[:, 0]-2450000, RV_mod[:, 7]-RV_mod[:, 5], c=default_color, 
                s=dot_size, zorder=20-n_dataset, alpha=1.0, label=datasets_labels[dataset])

    # Residuals
    ax_1.errorbar(RV_mod[:, 0]-2450000, RV_mod[:, 10], yerr=error, color='black', 
                  markersize=0, alpha=0.25, fmt='o', zorder=1)
    ax_1.scatter(RV_mod[:, 0]-2450000, RV_mod[:, 10], c=default_color, 
                s=dot_size, zorder=20-n_dataset, alpha=1.0)

ax_1.axhline(0.000, c='k', zorder=3)

# Set limits
if full_dict.get('limits_full_y', False):
    ax_0.set_ylim(full_dict['limits_full_y'])
if full_dict.get('limits_residuals_y', False):
    ax_1.set_ylim(full_dict['limits_residuals_y'])

ax_0.set_xlim(full_dict['limits_bjd'][0]-2450000, full_dict['limits_bjd'][1]-2450000)
ax_1.set_xlim(full_dict['limits_bjd'][0]-2450000, full_dict['limits_bjd'][1]-2450000)

# Formatting
ax_0.axes.get_xaxis().set_ticks([])
ax_0.yaxis.set_major_locator(MultipleLocator(5))
ax_0.yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax_0.yaxis.set_minor_locator(MultipleLocator(1))
ax_1.yaxis.set_major_locator(MultipleLocator(5))
ax_1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax_1.yaxis.set_minor_locator(MultipleLocator(1))

ax_0.set_ylabel('RV [m/s]')
ax_1.set_xlabel('Time [BJD-2450000]')
ax_1.set_ylabel('Residuals [m/s]')
ax_0.legend(framealpha=1.0, loc='lower left')

plot_filename = filename + '_full_model.png'
print(f'Full RV plot saved to: {plot_filename}')
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')


# ============================================================================
# PLOT 3: ACTIVITY INDICATORS (COMBINED)
# ============================================================================
print("\n" + "="*60)
print("GENERATING COMBINED ACTIVITY PLOT")
print("="*60)

fig = plt.figure(figsize=figsize)
ax_0, ax_1 = plots_in_grid()

for n_dataset, dataset in enumerate(datasets_list):
    print(f"Processing dataset: {dataset}")
    default_color = 'C' + repr(n_dataset)

    # Plot model
    activity_full = np.genfromtxt(
        dir_base + dir_mods + dir_plot + dataset + '_' + activity_model + '_full.dat', 
        skip_header=1
    )
    ax_0.plot(activity_full[:, 0]-2450000.0, activity_full[:, 3], color='k', 
              linestyle='-', zorder=2, label='Activity model', lw=1)

    activity_mod = np.genfromtxt(
        dir_base + dir_mods + dir_plot + dataset + '_' + activity_model + '.dat', 
        skip_header=1
    )
    error = np.sqrt(activity_mod[:, 9]**2 + activity_mod[:, 12]**2)

    ax_0.errorbar(activity_mod[:, 0]-2450000.0, activity_mod[:, 8], yerr=error, 
                  color='black', markersize=0, alpha=0.25, fmt='o', zorder=0)
    ax_0.scatter(activity_mod[:, 0]-2450000.0, activity_mod[:, 8], c=default_color, 
                s=dot_size, zorder=20-n_dataset, alpha=1.0, label=datasets_labels[dataset])

    # Residuals
    ax_1.errorbar(activity_mod[:, 0]-2450000.0, activity_mod[:, 10], yerr=error, 
                  color='black', markersize=0, alpha=0.25, fmt='o', zorder=1)
    ax_1.scatter(activity_mod[:, 0]-2450000.0, activity_mod[:, 10], c=default_color, 
                s=dot_size, zorder=20-n_dataset, alpha=1.0)

ax_1.axhline(0.000, c='k', zorder=3)

# Set limits
if activity_dict.get('limits_full_y', False):
    ax_0.set_ylim(activity_dict['limits_full_y'])
if activity_dict.get('limits_residuals_y', False):
    ax_1.set_ylim(activity_dict['limits_residuals_y'])

ax_0.set_xlim(activity_dict['limits_bjd'][0]-2450000, activity_dict['limits_bjd'][1]-2450000)
ax_1.set_xlim(activity_dict['limits_bjd'][0]-2450000, activity_dict['limits_bjd'][1]-2450000)

# Formatting
ax_0.axes.get_xaxis().set_ticks([])
ax_0.yaxis.set_major_locator(MultipleLocator(5))
ax_0.yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax_0.yaxis.set_minor_locator(MultipleLocator(1))
ax_1.yaxis.set_major_locator(MultipleLocator(5))
ax_1.yaxis.set_major_formatter(FormatStrFormatter('%d'))
ax_1.yaxis.set_minor_locator(MultipleLocator(1))

ax_0.set_ylabel('RV [m/s]')
ax_1.set_xlabel('Time [BJD-2450000]')
ax_1.set_ylabel('Residuals [m/s]')
ax_0.legend(framealpha=1.0, loc='lower left')

plot_filename = filename + '_activity_RV_model.png'
print(f'Combined activity plot saved to: {plot_filename}')
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')


# ============================================================================
# PLOT 4: INDIVIDUAL ACTIVITY INDICATORS
# ============================================================================
print("\n" + "="*60)
print("GENERATING INDIVIDUAL ACTIVITY PLOTS")
print("="*60)

for n_dataset, dataset in enumerate(activity_list):
    print(f"\nProcessing activity indicator: {dataset}")
    
    fig = plt.figure(figsize=figsize)
    ax_0, ax_1 = plots_in_grid()

    # Plot model
    activity_full = np.genfromtxt(
        dir_base + dir_mods + dir_plot + dataset + '_' + activity_model + '_full.dat', 
        skip_header=1
    )
    ax_0.plot(activity_full[:, 0], activity_full[:, 3], color='k', 
              linestyle='-', zorder=2, label='Activity model', lw=0.1)

    default_color = 'C' + repr(n_dataset)

    activity_mod = np.genfromtxt(
        dir_base + dir_mods + dir_plot + dataset + '_' + activity_model + '.dat', 
        skip_header=1
    )
    error = np.sqrt(activity_mod[:, 9]**2 + activity_mod[:, 12]**2)

    ax_0.errorbar(activity_mod[:, 0], activity_mod[:, 8], yerr=error, 
                  color='black', markersize=0, alpha=0.25, fmt='o', zorder=0)
    ax_0.scatter(activity_mod[:, 0], activity_mod[:, 8], c=default_color, 
                s=dot_size, zorder=20-n_dataset, alpha=1.0, label=activity_labels[dataset])

    # Residuals
    ax_1.errorbar(activity_mod[:, 0], activity_mod[:, 10], yerr=error, 
                  color='black', markersize=0, alpha=0.25, fmt='o', zorder=1)
    ax_1.scatter(activity_mod[:, 0], activity_mod[:, 10], c=default_color, 
                s=dot_size, zorder=20-n_dataset, alpha=1.0)

    ax_1.axhline(0.000, c='k', zorder=3)

    # Auto-scale limits
    val_min = np.amin(activity_mod[:, 8] - error)
    val_max = np.amax(activity_mod[:, 8] + error)
    res_min = np.amin(activity_mod[:, 10] - error)
    res_max = np.amax(activity_mod[:, 10] + error)
    
    ax_0.set_xlim(activity_dict['limits_bjd'])
    ax_0.set_ylim(val_min - np.abs(val_min)*0.05, val_max + np.abs(val_max)*0.05)
    ax_1.set_xlim(activity_dict['limits_bjd'])
    ax_1.set_ylim(res_min - np.abs(res_min)*0.05, res_max + np.abs(res_max)*0.05)

    # Formatting
    ax_0.axes.get_xaxis().set_ticks([])
    ax_0.set_ylabel('Activity index')
    ax_1.set_xlabel('Time [BJD-2450000]')
    ax_1.set_ylabel('Residuals [m/s]')
    ax_0.legend(framealpha=1.0, loc='lower left')
    
    plot_filename = filename + '_activity_' + dataset + '.png'
    print(f'Activity plot for {dataset} saved to: {plot_filename}')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

print("\n" + "="*60)
print("ALL PLOTS GENERATED SUCCESSFULLY!")
print("="*60)