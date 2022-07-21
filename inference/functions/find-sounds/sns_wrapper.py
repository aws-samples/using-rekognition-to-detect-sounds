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
import logging
import boto3
from botocore.exceptions import ClientError
import os

# Variables supplied by the environment
SNS_CLIENT = boto3.client('sns')
TOPIC_ARN = os.getenv("SNS_TOPIC_ARN")
LOGGER = logging.getLogger(__name__)


def publish_message(message, attributes):
    """
    Send a message to a configured SNS Topic

    Args:
        message (str): Text to put in the message
        attributes (str): dictionary of tags

    Returns:
        str: SNS Supplied message ID
    """
    try:
        att_dict = {}
        for key, value in attributes.items():
            if isinstance(value, str):
                att_dict[key] = {'DataType': 'String', 'StringValue': value}
            elif isinstance(value, bytes):
                att_dict[key] = {'DataType': 'Binary', 'BinaryValue': value}
        response = SNS_CLIENT.publish(Message=message, MessageAttributes=att_dict, TopicArn=TOPIC_ARN)
        message_id = response['MessageId']
        LOGGER.info(
            "Published message with attributes %s to topic %s.", attributes,
            TOPIC_ARN)
    except ClientError:
        LOGGER.exception("Couldn't publish message to topic %s.", TOPIC_ARN)
        raise
    else:
        return message_id
