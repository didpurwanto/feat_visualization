# feat_visualization
Feature visualization for I3D Network <br />


This code demonstrates how to visualize the features from 3D Convolutional network. We use vanilla gradients and guided back propagation to compute these saliency representation. Note, since the I3D is 3D convolution, hence, the input is a set of frames. To visualize the features, we need to modify the axis of visualization.py inside the folder /usr/local/lib/python2.7/dist-packages/saliency. <br /><br />

Step by step: <br />
1. install saliency by <br />
pip install saliency <br />

2. edit the axis of visualization into 3, which originally set as 2. <br />
Go to folder /usr/local/lib/python2.7/dist-packages/saliency <br />
Line 23: axis = 3, instead of axis = 2. <br />

3. put this folder project inside the I3D/experiments/dataset/ along with the train.py and test.py. <br />

4. the results are stored in .visualization/results/. <br /><br />





This code is inspired by PAIRML Saliency library <a href="https://github.com/PAIR-code/saliency/blob/master/Examples.ipynb">link</a> <br />

Thanks to 吳侒融.