# Copyright (c) 2015, Oleg Puzanov
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
Simple demo to show the machine learning in action for iBeacon-based advertising.

Supplied CSV contains the collected data from a shopping mall, where iBeacons are installed.

Every record defines the parameters for a successful case - visitor has entered a store or clicked
on a mobile app's banner / button.

This code is using TinyLearn module, which simplifies the classification tasks with Scikit-Learn and Pandas.
"""

import pandas as pd
import numpy as np
from tinylearn import CommonClassifier
from sklearn.preprocessing import LabelEncoder

some_data = pd.read_csv("data/beacon_data.csv", header=0, index_col=None)

# Encode strings from CSV into numeric values
enc = LabelEncoder()

for col_name in some_data:
    some_data[col_name] = enc.fit_transform(some_data[col_name])

# Split the data into training and test sets (the last 5 items)
train_features, train_labels = some_data.iloc[:-5, :-1], some_data.iloc[:-5, -1]

# Create an instance of CommonClassifier, which will use the default list of estimators.
# Removing the features with a weight smaller than 0.1.
wrk = CommonClassifier(default=True, cv=3, reduce_func=lambda x: x < 0.1)
wrk.fit(train_features, train_labels)
wrk.print_fit_summary()

# Predicting and decoding the labels back to strings
print("\nPredicted data:")
predicted = wrk.predict(some_data.iloc[-5:, :-1])
print(enc.inverse_transform(predicted))

print("\nActual accuracy: " +
      str(np.sum(predicted == some_data.iloc[-5:, -1])/predicted.size*100) + '%')
