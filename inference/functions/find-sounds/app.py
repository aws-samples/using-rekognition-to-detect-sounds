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

import librosa as libr
from urllib.parse import unquote_plus
import os
import io
import datetime
import boto3
from aws_lambda_powertools import Tracer
from spectrogram_plotter import plot_spectrogram
from rekognition_wrapper import show_custom_labels
from sns_wrapper import publish_message

tracer = Tracer()

# Value must be between 0 and 1.  0 <= value < 1
OVERLAP = float(os.getenv("SAMPLE_OVERLAP", '0.25'))
# Lenghth of the audio clip in seconds
CLIP_LENGTH = int(os.getenv("SAMPLE_LENGTH", '3'))
# Minimum acceptable confildence. 0 <= value <= 1
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", '0.90'))
# (Probable) S3 Bucket name
S3_BUCKET = os.getenv("AUDIO_BUCKET")

# Min to max frequencies to plot spectrogram
FREQ_LIMIT = [1000, 4000]
# Samples per second
SAMPLE_RATE = 48000
# length of sample in terms of samples
SAMPLE_LEN = SAMPLE_RATE * CLIP_LENGTH
# How much to advance when sampling for tne next clip
CLIP_OFFSET = int(SAMPLE_LEN*(1.0-OVERLAP))
# Resample the audio to mono
IS_MONO = True


def build_clipset(raw_audio):
    """
    Raw S3 data is resampled to the target rate and converted to 
    mono audio if in stereo

    Args:
        raw_audio (numpy.Array): full resampled audio data

    Returns:
        Array[numpy.Array]: The full set of clips to be tested
    """
    sample_data, _ = libr.load(raw_audio, sr=SAMPLE_RATE, mono=IS_MONO)

    clip_len = sample_data.shape[0]
    clip_position = 0

    clipset = []

    if clip_len < SAMPLE_LEN:
        clipset.append(sample_data)
    else:
        while clip_position < clip_len:
            window_sample = sample_data[clip_position: min(clip_position+SAMPLE_LEN, clip_len)]
            clipset.append(window_sample)
            clip_position = clip_position + CLIP_OFFSET

    return clipset


def download_to_memory_file_object(bucketname, key):
    """
    Load S3 audio object into memory (retaining the file structure)

    Args:
        bucketname (str): S3 bucket name
        key (str): S3 object key

    Returns:
        io.BytesIO: Memory structure containing file contents
    """
    # download from S3 to memory
    session = boto3.Session()
    session_s3 = session.resource('s3')
    my_bucket = session_s3.Bucket(bucketname)
    s3_object = my_bucket.Object(key)
    byte_data = io.BytesIO()
    s3_object.download_fileobj(byte_data)
    byte_data.seek(0)
    return byte_data


def send_event(confidence, time_delta_start, time_delta_end, file):
    """
    For detected alarms, send a message through SNS to subscrivers

    Formats message and attaches meta-data

    Args:
        confidence (float): Rekognition generated confidence that the clip described contains an alarm sound
        time_delta_start (time.timedelta): Time from start of clip to the start of the clip (including ms) that has the alarm
        time_delta_end ([type]): Time from start of clip to the end of the clip (including ms) that has the alarm
        file (str): S3 object key for audio file that conained alarm
    """
    attributes = {'Confidence': f'{confidence:.2f}',
                  'Start_Time': str(time_delta_start),
                  'End_Time': str(time_delta_end)}
    message = f'Found alarm in {file}\n{attributes}'

    publish_message(message, attributes)


def check_audio_for_event(bucketname, key):
    """
    Main controller for alarm detection:
      Downloads file from S3
      Break it down into clips
      Create spectrogram for each clip
      Use rekognition to classify
      Send notification if clip contains alarm sound with 
        confidence > MIN_CONFIDENCE

    Args:
        bucketname (str): S3 bucket name
        key (str): S3 object key
    """
    # download the file
    data = download_to_memory_file_object(bucketname, key)

    # build clip set
    clipset = build_clipset(data)

    # convert to spectrogram
    index = 0
    for clip in clipset:
        image_buffer = io.BytesIO()
        plot_spectrogram(clip, SAMPLE_RATE, spect_type='Mel', freq_range=FREQ_LIMIT, image_buffer=image_buffer)

        labels = show_custom_labels(image_buff=image_buffer, min_conf=0)

        label = next(item for item in labels if item['Name'] == 'alarm')
        confidence = float(label['Confidence']) / 100

        offset = (index * CLIP_OFFSET) / SAMPLE_RATE
        time_delta_start = datetime.timedelta(milliseconds=offset*1000)
        time_delta_end = datetime.timedelta(milliseconds=(offset+CLIP_LENGTH)*1000)

        if confidence >= MIN_CONFIDENCE:
            print(f'Sending a message for index: {index}')
            send_event(confidence, time_delta_start, time_delta_end, key)
        index += 1


@tracer.capture_lambda_handler
def lambda_handler(event, context):
    """
    Lambda kicked off by S3 object upload

    Args:
        event (dictionary): S3 event information
        context (dictionary): Lambda execution context
    """
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = unquote_plus(record['s3']['object']['key'])
        print(f'Checking file {key} in bucket {bucket} for audio event')
        check_audio_for_event(bucket, key)


if __name__ == "__main__":
    check_audio_for_event(S3_BUCKET, 'Crowded.wav')
