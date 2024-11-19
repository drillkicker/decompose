import librosa
import soundfile as sf
import numpy as np
from kymatio import Scattering1D
from scipy.signal import butter, filtfilt
from sklearn.cluster import KMeans

def debug_save_band(filtered_signal, band_idx, sr, chunk_idx):
    """
    Save filtered signal for a band for debugging purposes.
    Parameters:
        filtered_signal (1D array): Signal after bandpass filtering.
        band_idx (int): Band index.
        sr (int): Sampling rate.
        chunk_idx (int): Chunk index.
    """
    sf.write(f"debug_band_{band_idx}_chunk_{chunk_idx}.wav", filtered_signal, sr)

def remove_dc_offset(signal):
    """Removes the DC offset from the signal."""
    mean_value = np.mean(signal)
    return signal - mean_value

# Bandpass filtering for a specific frequency band
def apply_filter(signal, sr, low_freq, high_freq):
    """
    Apply bandpass filter to a signal for a specific scale.
    Parameters:
        signal (1D array): Input signal.
        sr (int): Sampling rate.
        low_freq (float): Lower cutoff frequency.
        high_freq (float): Upper cutoff frequency.
    Returns:
        filtered_signal (1D array): Bandpass-filtered signal.
    """
    nyquist = sr / 2

    # Apply DC offset removal
    signal_no_dc = remove_dc_offset(signal)
    
    if low_freq >= nyquist or high_freq >= nyquist:
        raise ValueError("Cutoff frequencies must be below the Nyquist frequency.")
    b, a = butter(4, [low_freq / nyquist, high_freq / nyquist], btype="band")
    return filtfilt(b, a, signal)

def apply_lowpass(signal, sr, high_freq):
    nyquist = sr / 2

    signal_no_dc = remove_dc_offset(signal)

    b, a = butter(4, [high_freq / nyquist], btype="low")
    
    return filtfilt(b, a, signal)

# Wavelet Scattering Transform
def compute_scattering(signal, sr, J=6, Q=12):
    scattering = Scattering1D(J=J, shape=len(signal), Q=Q)
    return scattering(signal)

# Identify wavetables from zero-crossings
def identify_wavetables(signal, num_zero_crossings):
    """
    Identify wavetables by grouping multiple zero-crossings.
    Parameters:
        signal (1D array): Input signal.
        num_zero_crossings (int): Number of zero-crossings per wavetable.
    Returns:
        wavetables (list of arrays): Segments of the signal between grouped zero-crossings.
        positions (list of tuples): Start and end positions of each wavetable.
    """
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    wavetables = []
    positions = []

    for i in range(0, len(zero_crossings) - num_zero_crossings, num_zero_crossings):
        start = zero_crossings[i]
        end = zero_crossings[i + num_zero_crossings]
        wavetable = signal[start:end]
        wavetables.append(wavetable)
        positions.append((start, end))

    # Handle the remaining segment
    if len(zero_crossings) >= num_zero_crossings:
        start = zero_crossings[-num_zero_crossings]
        end = len(signal)
        wavetable = signal[start:end]
        wavetables.append(wavetable)
        positions.append((start, end))

    print(f"DEBUG: Found {len(wavetables)} wavetables, combined length: {sum(len(w) for w in wavetables)}")
    return wavetables, positions

def rearrange_wavetables(wavetables, features, method="nearest", max_clusters=200):
    """
    Rearrange wavetables based on clustering to reduce memory usage.
    Parameters:
        wavetables (list of arrays): List of identified wavetables.
        features (array): Features to guide rearrangement.
        method (str): Rearrangement method ("nearest", "random", etc.).
        max_clusters (int): Maximum number of clusters for grouping features.
    Returns:
        list of arrays: Rearranged wavetables.
    """
    num_wavetables = len(wavetables)
    num_features = len(features)

    if num_wavetables == 0 or num_features == 0:
        print("DEBUG: No wavetables or features available for rearrangement.")
        return wavetables

    # Interpolate features to match the number of wavetables
    if num_wavetables != num_features:
        print(f"DEBUG: Mismatch in number of wavetables ({num_wavetables}) and features ({num_features}). Interpolating features.")
        feature_indices = np.linspace(0, num_features - 1, num=num_wavetables, dtype=np.float64)
        features = np.array([np.interp(feature_indices, np.arange(num_features), features[:, i]) for i in range(features.shape[1])]).T

    # Check for NaN values in features
    if np.isnan(features).any():
        print("DEBUG: NaN values detected in features. Replacing NaNs with zeros.")
        features = np.nan_to_num(features, nan=0.0)

    if len(features) != len(wavetables):
        print(f"DEBUG: Adjusting features ({len(features)}) to match wavetables ({len(wavetables)}).")
        if len(features) < len(wavetables):
            # Repeat features if fewer than wavetables
            features = np.tile(features, (len(wavetables) // len(features) + 1, 1))[:len(wavetables)]
        else:
            # Truncate features if more than wavetables
            features = features[:len(wavetables)]

    if method == "nearest":
        print(f"DEBUG: Using k-means clustering for memory-efficient rearrangement.")

        # Step 1: Cluster features
        kmeans = KMeans(n_clusters=min(max_clusters, len(features)), random_state=0)
        cluster_labels = kmeans.fit_predict(features)

        # Step 2: Sort features by cluster labels
        sorted_indices = np.argsort(cluster_labels)

        # Rearrange wavetables
        rearranged = [wavetables[i] for i in sorted_indices]
    elif method == "random":
        rearranged = np.random.permutation(wavetables).tolist()
    else:
        raise ValueError(f"Unknown rearrangement method: {method}")

    print(f"DEBUG: Rearranged {num_wavetables} wavetables using method '{method}'.")
    return rearranged

# Reconstruct signal from rearranged wavetables
def reconstruct_signal_no_spacing(wavetables, length):
    """
    Reconstructs a signal by concatenating rearranged wavetables sequentially,
    ensuring no overlap or gaps.

    Parameters:
        wavetables (list of arrays): List of rearranged wavetables.
        length (int): Total length of the reconstructed signal.
    Returns:
        reconstructed_signal (1D array): Reconstructed signal.
    """
    reconstructed_signal = np.zeros(length)
    current_position = 0

    for idx, wavetable in enumerate(wavetables):
        wavetable_length = len(wavetable)

        # Skip empty or invalid wavetables
        if wavetable_length == 0:
            continue

        # Truncate wavetable if it exceeds remaining signal length
        if current_position + wavetable_length > length:
            wavetable = wavetable[:length - current_position]

        # Add wavetable to the reconstructed signal
        reconstructed_signal[current_position:current_position + wavetable_length] = wavetable
        current_position += wavetable_length

        # Stop if the reconstructed signal is fully filled
        if current_position >= length:
            break

    print(f"DEBUG: Used {idx + 1}/{len(wavetables)} wavetables to reconstruct signal.")
    print(f"DEBUG: Reconstructed signal filled up to position {current_position} of {length}.")
    return reconstructed_signal

def validate_and_normalize_signal(signal):
    if np.any(np.isnan(signal)) or len(signal) == 0:
        print("DEBUG: Combined signal contains NaN or is empty. Replacing with silence.")
        signal = np.zeros_like(signal)
    else:
        # Normalize signal to range [-1, 1]
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val
    return signal

# Process signal independently for each frequency band
def process_signal_by_band(signal, sr, band_info, grain_size, J=6, Q=12, method="nearest", chunk_idx=0):
    combined_signal = np.zeros_like(signal)

    for band_idx, (low_freq, high_freq) in enumerate(band_info):
        print(f"Processing band {band_idx}: {low_freq}-{high_freq} Hz")

        try:
            # Step 1: Bandpass filter the signal
            if low_freq < 40:
                filtered_signal = apply_lowpass(signal, sr, high_freq)
            else:
                filtered_signal = apply_filter(signal, sr, low_freq, high_freq)

            ## Save filtered signal for inspection
            #sf.write(f"debug_filtered_band_{band_idx}_chunk_{chunk_idx}.wav", filtered_signal, sr)
            #print(f"DEBUG: Filtered signal for band {band_idx} - Min: {np.min(filtered_signal)}, Max: {np.max(filtered_signal)}, Mean: {np.mean(filtered_signal)}, Std: {np.std(filtered_signal)}")

            ## Abort band processing if signal appears invalid
            #if np.all(filtered_signal == 0) or np.isnan(filtered_signal).any():
            #    print(f"ERROR: Filtered signal for band {band_idx} is silent or contains NaN. Skipping.")
            #    continue

            # Debug: Check filtered signal
            if np.isnan(filtered_signal).any() or np.isinf(filtered_signal).any():
                print(f"DEBUG: Filtered signal for band {band_idx} contains NaN or Inf values. Replacing with zeros.")
                filtered_signal = np.nan_to_num(filtered_signal, nan=0.0, posinf=0.0, neginf=0.0)
            
            normalized_signal = filtered_signal / (np.max(np.abs(filtered_signal)) + 1e-8)
            if np.max(np.abs(filtered_signal)) == 0:
                print(f"DEBUG: Filtered signal for band {band_idx} is silent.")
                continue
        except ValueError as e:
            print(f"Skipping band {band_idx} due to filter error: {e}")
            continue

        # Step 2: Compute scattering coefficients
        if low_freq < 40:
            J_band0 = 2
            Q_band0 = 32  # Reduce Q but ensure it’s at least 1
            scattering_features = compute_scattering(filtered_signal, sr, J_band0, Q_band0)
        else:
            scattering_features = compute_scattering(filtered_signal, sr, J, Q)

        # Handle NaN values in scattering features
        if np.isnan(scattering_features).any() or np.isinf(scattering_features).any():
            print(f"DEBUG: Invalid scattering features detected in band {band_idx}. Replacing with zeros.")
            scattering_features = np.nan_to_num(scattering_features, nan=0.0, posinf=0.0, neginf=0.0)

        # Step 3: Identify and rearrange wavetables
        wavetables, _ = identify_wavetables(filtered_signal, grain_size)
        if not wavetables:
            print(f"No valid wavetables found for band {band_idx}. Skipping.")
            continue

        print(f"DEBUG: Band {band_idx} - Found {len(wavetables)} wavetables, combined length: {sum(len(w) for w in wavetables)}")

        rearranged_wavetables = rearrange_wavetables(wavetables, scattering_features, method)

        # Step 4: Reconstruct the band-specific signal
        band_signal = reconstruct_signal_no_spacing(rearranged_wavetables, len(signal))

        ## Debugging: Save band signal
        #debug_band_file = f"debug_band_{band_idx}_chunk_{chunk_idx}.wav"
        #sf.write(debug_band_file, band_signal, sr)
        #print(f"DEBUG: Saved reconstructed band {band_idx} to {debug_band_file}")

        # Validate band signal
        if np.isnan(band_signal).any():
            print(f"WARNING: Band {band_idx} contains NaN values. Skipping this band.")
            continue

        # Add band signal to combined signal
        combined_signal += band_signal

    # Validate and normalize combined signal
    #if np.isnan(combined_signal).any() or np.max(np.abs(combined_signal)) == 0:
    #    print(f"DEBUG: Combined signal contains NaN or is empty. Replacing with silence.")
    #    combined_signal = np.zeros_like(signal)
    #else:
    #    combined_signal = validate_and_normalize_signal(combined_signal)

    ## Debugging: Save combined signal
    #debug_combined_file = f"debug_combined_chunk_{chunk_idx}.wav"
    #sf.write(debug_combined_file, combined_signal, sr)
    #print(f"DEBUG: Saved debug combined output to {debug_combined_file}")

    return combined_signal

# Main function
def main(input_wav, output_wav, J=6, Q=12, chunk_duration=10, method="nearest", grain_size=16, num_bands=5): 
    """
    Main function to process the input stereo .wav file and save the rearranged output.
    Parameters:
        input_wav (str): Path to the input .wav file.
        output_wav (str): Path to save the output .wav file.
        band_info (list of tuples): List of (low_freq, high_freq) ranges for each scale.
        J (int): Maximum scale of scattering.
        Q (int): Number of wavelets per octave.
        chunk_duration (float): Duration of each chunk in seconds.
        method (str): Rearrangement criterion.
    """
    # Define frequency bands for each scale
    log_space = (np.logspace(np.log2(20), np.log2(20000), num_bands + 1, base=2))
    band_info = [(log_space[i], log_space[i + 1]) for i in range(len(log_space) - 1)]
    for i, band in enumerate(band_info, 1):
        print(f"Band {i}: {band[0]:.2f} - {band[1]:.2f}")

    # Load the input stereo .wav file
    signal, sr = sf.read(input_wav)
    signal = signal.T  # Ensure signal is shape (channels, samples)
    print(f"Loaded input .wav file with shape {signal.shape} and sample rate {sr}")

    # Determine chunk size
    chunk_size = int(chunk_duration * sr)
    num_chunks = int(np.ceil(signal.shape[1] / chunk_size))

    # Split the signal into chunks
    chunks = [
        signal[:, i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)
    ]
    print(f"Split signal into {len(chunks)} chunks of approximately {chunk_duration} seconds each.")

    processed_left = []
    processed_right = []

    for chunk_idx, chunk in enumerate(chunks):
        print(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")

        left_channel = chunk[0, :]
        right_channel = chunk[1, :]

        # Process the left channel
        try:
            left_processed = process_signal_by_band(left_channel, sr, band_info, grain_size, J, Q, method, chunk_idx=chunk_idx)
            if left_processed is None:
                raise ValueError(f"Invalid processed left channel for chunk {chunk_idx}")
        except Exception as e:
            print(f"ERROR: Failed to process left channel of chunk {chunk_idx}: {e}")
            left_processed = np.zeros_like(left_channel)  # Fallback to silence for invalid chunks

        # Process the right channel
        try:
            right_processed = process_signal_by_band(right_channel, sr, band_info, grain_size, J, Q, method, chunk_idx=chunk_idx)
            if right_processed is None:
                raise ValueError(f"Invalid processed right channel for chunk {chunk_idx}")
        except Exception as e:
            print(f"ERROR: Failed to process right channel of chunk {chunk_idx}: {e}")
            right_processed = np.zeros_like(right_channel)  # Fallback to silence for invalid chunks

        # Append processed channels
        processed_left.append(left_processed)
        processed_right.append(right_processed)

    # Ensure valid chunks before concatenating
    if not processed_left or not processed_right:
        raise ValueError("No valid chunks were processed. Cannot produce output file.")

    # Concatenate all chunks for the final left and right channels
    try:
        final_left = np.concatenate(processed_left)
        final_right = np.concatenate(processed_right)
    except Exception as e:
        raise ValueError(f"ERROR: Failed to concatenate chunks: {e}")

    # Save the final stereo .wav file
    try:
        sf.write(output_wav, np.stack([final_left, final_right], axis=1), sr)
        print(f"Saved processed output to {output_wav}")
    except Exception as e:
        raise ValueError(f"ERROR: Failed to save output file {output_wav}: {e}")

if __name__ == "__main__":
    main("input.wav", "output.wav", J=6, Q=12, chunk_duration=10, method="nearest", grain_size=32, num_bands=5)
