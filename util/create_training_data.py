"""
Amazon Software License

1. Definitions
“Licensor” means any person or entity that distributes its Work.

“Software” means the original work of authorship made available under this License.

“Work” means the Software and any additions to or derivative works of the Software that are made available
under this License.

The terms “reproduce,” “reproduction,” “derivative works,” and “distribution” have the meaning as provided
under U.S. copyright law; provided, however, that for the purposes of this License, derivative works shall
not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work.

Works, including the Software, are “made available” under this License by including in or with the Work either
(a) a copyright notice referencing the applicability of this License to the Work, or (b) a copy of this License.

2. License Grants
2.1 Copyright Grant. Subject to the terms and conditions of this License, each Licensor grants to you a perpetual,
worldwide, non-exclusive, royalty-free, copyright license to reproduce, prepare derivative works of, publicly
display, publicly perform, sublicense and distribute its Work and any resulting derivative works in any form.
2.2 Patent Grant. Subject to the terms and conditions of this License, each Licensor grants to you a perpetual,
worldwide, non-exclusive, royalty-free patent license to make, have made, use, sell, offer for sale, import,
and otherwise transfer its Work, in whole or in part. The foregoing license applies only to the patent claims
licensable by Licensor that would be infringed by Licensor’s Work (or portion thereof) individually and excluding
any combinations with any other materials or technology.

3. Limitations
3.1 Redistribution. You may reproduce or distribute the Work only if (a) you do so under this License, (b) you include
a complete copy of this License with your distribution, and (c) you retain without modification any copyright, patent,
trademark, or attribution notices that are present in the Work.
3.2 Derivative Works. You may specify that additional or different terms apply to the use, reproduction, and distribution
of your derivative works of the Work (“Your Terms”) only if (a) Your Terms provide that the use limitation in Section 3.3
applies to your derivative works, and (b) you identify the specific derivative works that are subject to Your Terms.
Notwithstanding Your Terms, this License (including the redistribution requirements in Section 3.1) will continue to
apply to the Work itself.
3.3 Use Limitation. The Work and any derivative works thereof only may be used or intended for use with the web services,
computing platforms or applications provided by Amazon.com, Inc. or its affiliates, including Amazon Web Services, Inc.
3.4 Patent Claims. If you bring or threaten to bring a patent claim against any Licensor (including any claim,
cross-claim or counterclaim in a lawsuit) to enforce any patents that you allege are infringed by any Work, then your
rights under this License from such Licensor (including the grants in Sections 2.1 and 2.2) will terminate immediately.
3.5 Trademarks. This License does not grant any rights to use any Licensor’s or its affiliates’ names, logos, or
trademarks, except as necessary to reproduce the notices described in this License.
3.6 Termination. If you violate any term of this License, then your rights under this License (including the grants
in Sections 2.1 and 2.2) will terminate immediately.

4. Disclaimer of Warranty.
THE WORK IS PROVIDED “AS IS” WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES
OR CONDITIONS OF M ERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR NON-INFRINGEMENT. YOU BEAR THE RISK OF
UNDERTAKING ANY ACTIVITIES UNDER THIS LICENSE. SOME STATES’ CONSUMER LAWS DO NOT ALLOW EXCLUSION OF AN IMPLIED WARRANTY,
SO THIS DISCLAIMER MAY NOT APPLY TO YOU.

5. Limitation of Liability.
EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE),
CONTRACT, OR OTHERWISE SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK
(INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION, LOST PROFITS OR DATA, COMPUTER FAILURE OR
MALFUNCTION, OR ANY OTHER COMM ERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY
OF SUCH DAMAGES.

Effective Date – April 18, 2008 © 2008 Amazon.com, Inc. or its affiliates. All rights reserved.
"""

from spectrogram_plotter import plot_spectrogram
import os
import os.path as path
import glob
import random as rnd
import librosa as libr
import numpy as np


# paths for the input audio files
ALARM_AUDIO_SOURCE = path.join('..', 'audio', 'alarms')
BACKGROUND_AUDIO_SOURCE = path.join('..', 'audio', 'background')

# how to create the sound samples
MAX_BACKGROUND_LAYERS = 4  # when creating a new sample sound
NUM_IMAGES_TO_GENERATE_PER_CLASS = 10

# Spectrogram Types to create
SPECTROGRAM_TYPES = ['Std', 'Mel', 'QPlot-freq', 'reassigned', 'harmonic']

# Tuned for Alarm use case
FREQ_LIMIT = [1000, 4000]

# 3 seconds @ 48000
SAMPLE_DURATION = 3
SAMPLE_RATE = 48000
SAMPLE_LEN = SAMPLE_RATE * SAMPLE_DURATION

# generated folders
TOP_FOLDER = '../training-data'
DIR_PREFIX_WITH = 'alarm'
DIR_PREFIX_WITHOUT = 'no_alarm'

sample_cache = dict()

def get_wav_file_list(a_path):
    return glob.glob(path.join(a_path, '*.wav'))


def load_wav_file(a_path):
    if a_path in sample_cache:
        return sample_cache[a_path]
    samples, _ = libr.load(a_path, sr=SAMPLE_RATE, mono=True)
    # store the entire sample.  We'll take a random subset (or randomly pad it)
    # each time we reference it
    sample_cache[a_path] = samples
    return samples


def normalize_length(samples):
    num_samples = len(samples)
    if num_samples > SAMPLE_LEN:
        # too big?  truncate
        offset = rnd.randint(0, num_samples - SAMPLE_LEN)
        return np.copy(samples[offset:offset + SAMPLE_LEN])
    elif num_samples < SAMPLE_LEN:
        # too small?  pad
        pad_amount = SAMPLE_LEN - num_samples
        split = rnd.randint(0, pad_amount)
        pre = np.zeros(split, dtype='float32') if split > 0 else np.array([], dtype='float32')
        post = np.zeros(pad_amount - split, dtype='float32') if (pad_amount - split) > 0 else np.array([], dtype='float32')
        return np.concatenate((pre, samples, post))
    else:
        # perfect length, no adjustments needed
        return samples


def get_random_subset_of_waveform(wav_data):
    max_len = len(wav_data)
    # just for the alarm sounds, take from 50% to 100% of the sample data
    new_len = rnd.randrange(max_len // 2, max_len)
    offset = rnd.randint(0, SAMPLE_LEN - new_len)
    return np.copy(wav_data[offset: offset + new_len])


def layer_sounds(alarm, backgrounds):
    if alarm:
        wav_data = alarm
    else:
        wav_data = np.zeros(SAMPLE_LEN)
    for background in backgrounds:
        wav_data += background
    return wav_data


def get_random_mixed_audio(with_alarm):
    if with_alarm:
        fname = rnd.choice(alarm_file_list)
        wav_data = load_wav_file(fname)
    else:
        wav_data = np.zeros(SAMPLE_LEN)
    wav_data = get_random_subset_of_waveform(wav_data)
    wav_data = normalize_length(wav_data)

    num_background_layers = rnd.randint(1, MAX_BACKGROUND_LAYERS)
    for _ in range(num_background_layers):
        background_fname = rnd.choice(background_file_list)
        background_wav = load_wav_file(background_fname)
        background_wav = normalize_length(background_wav)
        wav_data += background_wav
    return wav_data


def save_spectrograms_for(spectrogram_type, folder, wav_data, img_num):
    fname = f'{folder}/img_{img_num}.png'
    plot_spectrogram(wav_data, SAMPLE_RATE, fileName=fname, spect_type=spectrogram_type, freq_range=FREQ_LIMIT)


def get_output_folder_for(spectrogram_type, with_alarm, output_subfolder):
    leaf_folder = DIR_PREFIX_WITH if with_alarm else DIR_PREFIX_WITHOUT
    folder_name = f'{TOP_FOLDER}/{spectrogram_type}/{output_subfolder}/{leaf_folder}'
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


# for debugging
os.chdir('clean_copy/util')

# get a list of all file names for alarm + background wav files
alarm_file_list = get_wav_file_list(ALARM_AUDIO_SOURCE)
background_file_list = get_wav_file_list(BACKGROUND_AUDIO_SOURCE)
MAX_BACKGROUND_LAYERS = min(MAX_BACKGROUND_LAYERS, len(background_file_list))


def generate_images():
    num_images = 0
    for spectrogram_type in SPECTROGRAM_TYPES:
        for with_alarm in [True, False]:
            for output_subfolder in ['train', 'test', 'validate']:
                folder = get_output_folder_for(spectrogram_type, with_alarm, output_subfolder)
                print(f'Creating images for {folder}')
                for _ in range(NUM_IMAGES_TO_GENERATE_PER_CLASS):
                    wav_data = get_random_mixed_audio(with_alarm)
                    save_spectrograms_for(spectrogram_type, folder, wav_data, num_images)
                    num_images += 1
    print(f'Created {num_images} images')


generate_images()
