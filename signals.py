import numpy as np

def generate_continuous_signal(t, A=1, phi=2, offset=0,signal_type='square'):
        """
        Generate different types of input signals
        
        Parameters:
        -----------
        t : array
            Time array
        A : float
            Amplitude of the signal (default 1)
        signal_type : str, optional
            Type of input signal (default 'square')
            Options: 'square', 'sine', 'triangle'
        
        Returns:
        --------
        numpy.ndarray
            Input signal
        """
        wave = np.sin(phi * np.pi * t) + offset/A

        if signal_type == 'square':
            return A * np.sign(wave)
        elif signal_type == 'sine':
            return A * wave


def generate_impulse_signal(t, rise_time=5, growth_rate=1, peak_amplitude=1, offset=0, plateau=False, lamb=0.1):
    signal = np.zeros_like(t)
    
    for i in range(min(rise_time, len(t))):
        normalized_value = (np.exp(growth_rate * i) - 1) / (np.exp(growth_rate * (rise_time - 1)) - 1)
        signal[i] = normalized_value * peak_amplitude
    
    if rise_time > 0:
        peak_index = min(rise_time-1, len(t)-1)
        signal[peak_index] = peak_amplitude

        if plateau:
            if lamb == 0:
                signal[peak_index:] = peak_amplitude
            else:
                for i in range(peak_index, len(t)):
                    decay_value = peak_amplitude * np.exp(-lamb * (t[i] - t[peak_index]))
                    signal[i] = max(decay_value, 0)
    
    return signal + offset


def generate_trapezoidal_signal(
    time,
    amplitude=0.68,
    rise_time=0.1,
    flat_top_time=0.2,
    fall_time=0.2,
    baseline=0.0,
    exponential_decay=False):
    """
    Generate a trapezoidal signal similar to CC4 Pulser Receiver with optional exponential decay.
    
    Parameters:
    -----------
    time : numpy.ndarray
        Time array for the signal
    amplitude : float
        Peak amplitude of the signal in volts
    rise_time : float
        Time taken for signal to rise from baseline to peak
    flat_top_time : float
        Time taken for signal to stay at peak amplitude
    fall_time : float
        Time taken for signal to fall from peak to baseline
    baseline : float
        Baseline voltage level
    exponential_decay : bool
        If True, uses exponential decay for falling edge instead of linear
        
    Returns:
    --------
    numpy.ndarray
        Signal array corresponding to the input time array
    """
    
    # Calculate sampling rate from time array
    sampling_rate = 1 / (time[1] - time[0])
    num_points = len(time)
    
    # Initialize signal array
    signal = np.zeros(num_points) + baseline
    
    # Convert times to number of samples
    rise_samples = int(rise_time * sampling_rate)
    flat_top_samples = int(flat_top_time * sampling_rate)
    fall_samples = int(fall_time * sampling_rate)
    
    # Calculate start indices
    start_idx = int(0.2 * num_points)  # Start at 20% of duration
    
    # Ensure we don't exceed array bounds
    flat_top_start = min(start_idx + rise_samples, num_points)
    fall_start = min(flat_top_start + flat_top_samples, num_points)
    end_idx = min(fall_start + fall_samples, num_points)
    
    # Generate rising edge
    if start_idx < flat_top_start:
        rise_points = flat_top_start - start_idx
        signal[start_idx:flat_top_start] = np.linspace(baseline, amplitude, rise_points)
    
    # Generate flat top
    if flat_top_start < fall_start:
        signal[flat_top_start:fall_start] = amplitude
    
    # Generate falling edge
    if fall_start < end_idx:
        fall_points = end_idx - fall_start
        if exponential_decay:
            # Calculate time constant tau based on fall_time
            # Using tau = fall_time/5 ensures signal reaches ~1% of initial value
            tau = fall_time / 5
            t = np.linspace(0, fall_time, fall_points)
            decay = amplitude * np.exp(-t/tau) + baseline
            signal[fall_start:end_idx] = decay
        else:
            # Original linear decay
            signal[fall_start:end_idx] = np.linspace(amplitude, baseline, fall_points)
    
    return signal

def generate_step_impulse_signal(time, amplitude=1.0, step_time=0.2, duration=0.05, baseline=0.0):
    """
    Generate a step impulse signal of a given amplitude and duration.
    
    Parameters:
    -----------
    time : numpy.ndarray
        Time array for the signal
    amplitude : float
        Amplitude of the step impulse signal
    step_time : float
        Time at which the step occurs
    duration : float
        Duration of the impulse
    baseline : float
        Baseline voltage level
    
    Returns:
    --------
    numpy.ndarray
        Step impulse signal array corresponding to the input time array
    """
    signal = np.full_like(time, baseline)
    step_idx = np.searchsorted(time, step_time)
    end_idx = np.searchsorted(time, step_time + duration)
    signal[step_idx:end_idx] = amplitude
    return signal

