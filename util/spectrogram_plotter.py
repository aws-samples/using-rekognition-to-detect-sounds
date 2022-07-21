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
import matplotlib
import matplotlib.pyplot as plt
from librosa.feature import melspectrogram
from librosa.display import specshow
from librosa import stft
from librosa import decompose
from librosa import amplitude_to_db
from librosa import power_to_db
from librosa import feature
from librosa import cqt
from librosa import reassigned_spectrogram
import numpy as np


def plot_spectrogram(wavdata, frequency, freq_range=[1, 8000], fig=None, fileName=None, showaxis='off',
                     fig_height=8, fig_width=16, dpi=120, spect_type='Std', image_buffer=None):
    """
    Render supplied wav data as a spectrogram of the selected type

    Args:
        wavdata (NumPy.array): Sampled wav audio data
        frequency (int): Sample Frequency
        freq_range (list, 2 dimensional array (min .. max frequency): Y Axis filter. Defaults to [0, 8000].
        fig (Pyplot.Figure, optional): Existing pyplot figure. Defaults to None.
        fileName (str, optional): Filename target to save spectrogram (overwrites exiting) Defaults to None.
        showaxis (str, optional 'off' or 'on'): Draw axis on spectrogram. Defaults to 'off'.
        fig_height (int, optional): Height of image in centimeters. Defaults to 8.
        fig_width (int, optional): Width of image in centimeters. Defaults to 16.
        dpi (int, optional): Dots per inch. Defaults to 120.
        spect_type (str, optional): type of spectrogram, acceptable values:
            None = Standard (matplotlib)
            Std = Standard (matplotlib)
            Mel = Melodic (librosa)
            QPlot-freq = Q Plot based on Frequency (librosa)
            QPlot-axis = Q Plot based on decibels (librosa)
            Chroma = Chromagram Plot (librosa) Note: does not apply freq_range to output
            mfcc = Standard MFCC (librosa) Note: does not apply freq_range to output
            mfcc-rast = MFCC RASTAMAT (librosa) Note: does not apply freq_range to output
            mfcc-htk = MFCC HTK Representation (librosa) Note: does not apply freq_range to output
            reassigned = Reassigned spectrogram
            harmonic = Harmonic-Percussive Source Separation
            percussive = Harmonic-Percussive Source Separation
            wave = wave plot (matplotlib)
            Defaults to 'Std'.
        image_buffer ([type], optional): Image buffer to render image into. Defaults to None.
    """

    matplotlib.use('Agg')
    plt.ioff()

    if fig is None:
        fig = plt.figure()

    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)
    fig.set_dpi(dpi)
    ax1 = fig.add_subplot()

    if spect_type is None:
        spect_type = 'Std'

    if spect_type == 'Std':
        D = stft(wavdata)
        S_db = amplitude_to_db(np.abs(D), ref=np.max)
        img = specshow(S_db, x_axis='time', y_axis='log', ax=ax1)
        ax1.axis(showaxis)
        ax1.set_ylim(freq_range)
    elif spect_type == 'Mel':
        S = melspectrogram(y=wavdata, sr=frequency)
        S_db = power_to_db(S, ref=np.max)
        img = specshow(S_db, x_axis='time', y_axis='mel', ax=ax1)
        ax1.set_ylim(freq_range)
        ax1.axis(showaxis)
    elif spect_type == 'QPlot-freq':
        C = cqt(y=wavdata, sr=frequency)
        C_db = amplitude_to_db(np.abs(C), ref=np.max)
        img = specshow(C_db, x_axis='time', y_axis='cqt_hz', ax=ax1)
        ax1.axis(showaxis)
        ax1.set_ylim(freq_range)
    elif spect_type == 'QPlot-axis':
        C = cqt(y=wavdata, sr=frequency)
        C_db = amplitude_to_db(np.abs(C), ref=np.max)
        img = specshow(C_db, x_axis='time', y_axis='cqt_note', ax=ax1)
        ax1.set_ylim(freq_range)
        ax1.axis(showaxis)
    elif spect_type == 'Chroma':
        chroma = feature.chroma_cqt(y=wavdata, sr=frequency)
        img = specshow(chroma, x_axis='time', y_axis='chroma', ax=ax1)
        # ax1.set_ylim(freq_range)
        ax1.axis(showaxis)
    elif spect_type == 'mfcc':
        mfccs = feature.mfcc(y=wavdata, sr=frequency, n_mfcc=40)
        img = specshow(mfccs, x_axis='time', ax=ax1)
        # ax1.set_ylim(freq_range)
        ax1.axis(showaxis)
    elif spect_type == 'mfcc-rast':
        m_slaney = feature.mfcc(y=wavdata, sr=frequency, dct_type=2)
        img = specshow(m_slaney, x_axis='time', ax=ax1)
        # ax1.set_ylim(freq_range)
        ax1.axis(showaxis)
    elif spect_type == 'mfcc-htk':
        m_htk = feature.mfcc(y=wavdata, sr=frequency, dct_type=3)
        img = specshow(m_htk, x_axis='time', ax=ax1)
        # ax1.set_ylim(freq_range)
        ax1.axis(showaxis)
    elif spect_type == 'reassigned':
        n_fft = 64
        m_htk = feature.mfcc(y=wavdata, sr=frequency, dct_type=3)
        freqs, times, mags = reassigned_spectrogram(y=wavdata, sr=frequency,
                                                    n_fft=n_fft)
        mags_db = power_to_db(mags, ref=np.max)
        ax1.scatter(times, freqs, c=mags_db, cmap="magma", alpha=0.1, s=5)
        ax1.axis(showaxis)
        ax1.set_ylim(freq_range)
        # ax1.set(title='Reassigned spectrogram')
    elif spect_type == 'harmonic':
        D = stft(y=wavdata)
        D_harmonic, D_percussive = decompose.hpss(D)
        rp = np.max(np.abs(D))

        specshow(amplitude_to_db(np.abs(D_harmonic), ref=rp),
                 y_axis='log', x_axis='time', ax=ax1)
        # specshow(np.abs(D_harmonic), y_axis='linear', x_axis='time', ax=ax1)
        # ax1.set(title='Harmonic spectrogram')
        ax1.set_ylim(freq_range)
        ax1.axis(showaxis)
    elif spect_type == 'percussive':
        D = stft(y=wavdata)
        D_harmonic, D_percussive = decompose.hpss(D)
        rp = np.max(np.abs(D))

        specshow(amplitude_to_db(np.abs(D_percussive), ref=rp),
                 y_axis='log', x_axis='time', ax=ax1)
        ax1.set_ylim(freq_range)
        ax1.axis(showaxis)
    elif spect_type == 'wave':
        plt.plot(np.linspace(0, len(wavdata) /
                 frequency, num=len(wavdata)), wavdata)
        plt.grid(True)
        plt.title('Normalized Waveform')
        plt.xlabel("Time")
        plt.ylabel("Amplitude")

    # render to disk
    if fileName is not None:
        plt.savefig(fileName, bbox_inches='tight',
                    pad_inches=0, format='png')
        plt.cla()
        plt.clf()
        plt.close()
        plt.close('all')

    # render to memory
    if image_buffer is not None:
        plt.savefig(image_buffer, bbox_inches='tight',
                    pad_inches=0, format='png')
        plt.cla()
        plt.clf()
        plt.close()
        plt.close('all')

    return
