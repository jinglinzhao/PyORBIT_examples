import numpy as np
import matplotlib.pyplot as plt
from finufft import nufft1d3


def wrap(ϕ):
    '''	
    An individual phase ranges within (-np.pi, np.pi)
    The difference of two phases ranges within (-2*np.pi, 2*np.pi)
    Adding multiples of 2*np.pi to the phase is effectively the same phase
    This function wraps Δϕ such that it lies within (-np.pi, np.pi)
    '''		
    for i in np.arange(len(ϕ)):
        ϕ[i] = ϕ[i] - int(ϕ[i]/np.pi) * 2 * np.pi
    return ϕ


def FIESTA_III(log_wav, spec_tpl, spec, 
                               freq_max=10000, n_freq=10001,
                               plot_template=False, plot_comparison=False):
    """
    Compute spectral analysis using NUFFT for template and target spectrum/spectra.
    
    Parameters
    ----------
    log_wav : array-like
        Log wavelength array (will be converted to float64)
    spec_tpl : array-like
        Template spectrum (1-D, will be converted to complex128)
    spec : array-like
        Target spectrum - can be 1-D or 2-D array of multiple 1-D spectra
        (will be converted to complex128)
    freq_max : float, optional
        Maximum frequency for uniform grid (default: 10000)
        Frequency grid will span from -freq_max to freq_max
    n_freq : int, optional
        Number of frequency points in the uniform grid (default: 10001)
    plot_template : bool, optional
        Whether to plot the template spectrum analysis (default: False)
    plot_comparison : bool, optional
        Whether to plot comparison between template and target.
        Only applicable if spec is 1-D (default: False)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'frequencies': uniform frequency grid
        - 'spectrum_tpl': NUFFT of template
        - 'magnitude_tpl': magnitude of template spectrum
        - 'phase_tpl': phase of template spectrum
        - 'power_tpl': power of template spectrum
        - 'spectrum': NUFFT of target (1-D or 2-D array)
        - 'magnitude': magnitude of target spectrum
        - 'phase': phase of target spectrum
        - 'power': power of target spectrum
        - 'is_1d': boolean indicating if spec is 1-D
        - 'phase_diff': wrapped phase difference (spec - template)
        - 'shift_spectrum': phase shift over -2pi (same shape as phase_diff)
        - 'rv_shift': weighted average of shift_spectrum using power as weights
    """
    
    # Convert to appropriate dtypes
    log_wav = log_wav.astype(np.float64)
    spec_tpl = spec_tpl.astype(np.complex128)
    spec = spec.astype(np.complex128)
    
    # Determine if spec is 1-D or 2-D
    is_1d = (spec.ndim == 1)
    
    # Define uniform frequency grid (symmetric from -freq_max to freq_max)
    frequencies = np.linspace(120, freq_max, n_freq).astype(np.float64)
    
    # Compute NUFFT for template
    spectrum_tpl = nufft1d3(log_wav, spec_tpl, frequencies)
    magnitude_tpl = np.abs(spectrum_tpl)
    phase_tpl = np.angle(spectrum_tpl)
    power_tpl = np.abs(spectrum_tpl)**2
    
    # Compute NUFFT for target spectrum/spectra
    if is_1d:
        # Single spectrum
        spectrum = nufft1d3(log_wav, spec, frequencies)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)
        power = np.abs(spectrum)**2
        # Calculate phase difference and RV shift for 1D
        phase_diff = wrap(phase - phase_tpl)
        shift_spectrum = phase_diff / (-2 * np.pi)
        # Weighted average for rv_shift
        if np.sum(power) > 0:
            rv_shift = np.sum(shift_spectrum * power) / np.sum(power)
        else:
            rv_shift = np.nan
    else:
        # Multiple spectra - process each row
        n_spectra = spec.shape[0]
        spectrum = np.zeros((n_spectra, len(frequencies)), dtype=np.complex128)
        magnitude = np.zeros_like(spectrum, dtype=np.float64)
        phase = np.zeros_like(spectrum, dtype=np.float64)
        power = np.zeros_like(spectrum, dtype=np.float64)
        phase_diff = np.zeros_like(spectrum, dtype=np.float64)
        shift_spectrum = np.zeros_like(spectrum, dtype=np.float64)
        rv_shift = np.zeros(n_spectra, dtype=np.float64)
        for i in range(n_spectra):
            spectrum[i] = nufft1d3(log_wav, spec[i], frequencies)
            magnitude[i] = np.abs(spectrum[i])
            phase[i] = np.angle(spectrum[i])
            power[i] = np.abs(spectrum[i])**2
            phase_diff[i] = wrap(phase[i] - phase_tpl)
            shift_spectrum[i] = phase_diff[i] / (-2 * np.pi)
            # Weighted average for rv_shift for each spectrum
            pwr = power[i]
            if np.sum(pwr) > 0:
                rv_shift[i] = np.sum(shift_spectrum[i] * pwr) / np.sum(pwr)
            else:
                rv_shift[i] = np.nan
    
    # Plot template if requested
    if plot_template:
        # Get first spectrum from spec (if 2D, use first row; if 1D, use as is)
        spec_first = spec[0] if not is_1d else spec
        _plot_template_spectrum(log_wav, spec_tpl, spec_first)
    
    # Plot comparison if requested and spec is 1-D
    if plot_comparison and is_1d:
        _plot_comparison(frequencies, 
                        phase_tpl, power_tpl,
                        phase, power)
    elif plot_comparison and not is_1d:
        print("Warning: plot_comparison is only available for 1-D spec. Skipping plot.")
    
    # Return results
    results = {
        'frequencies': frequencies,
        'spectrum_tpl': spectrum_tpl,
        'magnitude_tpl': magnitude_tpl,
        'phase_tpl': phase_tpl,
        'power_tpl': power_tpl,
        'spectrum': spectrum,
        'magnitude': magnitude,
        'phase': phase,
        'power': power,
        'is_1d': is_1d,
        'phase_diff': phase_diff,
        'shift_spectrum': shift_spectrum,
        'rv_shift': rv_shift
    }
    
    return results


def _plot_template_spectrum(log_wav, spec_tpl, spec_first):
    """Helper function to plot the template spectrum and first target spectrum in log wavelength space."""
    fig = plt.figure(figsize=(16, 4))
    plt.plot(log_wav, np.real(spec_tpl), '.', label='spec_tpl', alpha=0.6)
    plt.plot(log_wav, np.real(spec_first), '.', label='spec', alpha=0.6)
    plt.xlabel('Log Wavelength')
    plt.ylabel('Spectrum')
    plt.title('Template and Target Spectra')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_comparison(freqs, phase_tpl, power_tpl, phase, power):
    """Helper function to plot comparison between template and target spectra."""
        
    fig, (ax_top, ax_mid, ax_phase, ax_bot, ax_rv) = plt.subplots(
        5, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 1, 1]}
    )
    
    # First panel: log power spectra of both
    ax_top.semilogy(freqs, power_tpl, '.', label='spec_tpl', alpha=0.6)
    ax_top.semilogy(freqs, power, '.', label='spec', alpha=0.6)
    ax_top.set_ylabel('Power (log scale)')
    ax_top.set_title('Power Spectrum of spec_tpl and spec')
    ax_top.legend()
    ax_top.grid(True, alpha=0.3)
    
    # Second panel: ratio of the power spectra
    power_diff =  power / power_tpl
    ax_mid.plot(freqs, power_diff, '.', label='power / power_tpl', alpha=0.6)
    ax_mid.set_ylabel('Power Ratio')
    ax_mid.set_title('Power Spectrum Ratio')
    ax_mid.legend()
    ax_mid.grid(True, alpha=0.3)
    
    # Third panel: phase spectra of both
    ax_phase.plot(freqs, phase_tpl, '.', label='phase_tpl', alpha=0.6)
    ax_phase.plot(freqs, phase, '.', label='phase', alpha=0.6)
    ax_phase.set_ylabel('Phase [rad]')
    ax_phase.set_title('Phase Spectra')
    ax_phase.legend()
    ax_phase.grid(True, alpha=0.3)
    
    # Fourth panel: phase difference
    phase_diff = wrap(phase - phase_tpl)
    ax_bot.plot(freqs, phase_diff, '.', label='phase - phase_tpl', alpha=0.6)
    ax_bot.set_ylabel('Phase Diff [rad]')
    ax_bot.set_title('Phase Spectrum Difference')
    ax_bot.legend()
    ax_bot.grid(True, alpha=0.3)

    # Fifth panel: RV shift
    shift_spectrum = phase_diff / (-2 * np.pi)
    ax_rv.plot(freqs, shift_spectrum, '.', label='RV shift (phase_diff / -2π)', alpha=0.6)
    ax_rv.set_xlabel('Frequency')
    ax_rv.set_ylabel('Shift')
    ax_rv.set_title('Shift')
    ax_rv.legend()
    ax_rv.grid(True, alpha=0.3)

    
    plt.tight_layout()
    plt.show()
