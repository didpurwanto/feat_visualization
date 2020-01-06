# I3D Feature Visualization
Feature visualization for I3D Network <br />


This code demonstrates how to visualize the features from 3D Convolutional network. We use vanilla gradients and guided back propagation to compute these saliency representation. Note, since the I3D is 3D convolution, hence, the input is a set of frames. To visualize the features, we need to modify the axis of <code>visualization.py </code> in the folder <code>/usr/local/lib/python2.7/dist-packages/saliency.</code> <br /><br />

<h2>Running the code</h2> 
<h3>Setup </h3><br />

1. Clone the project <br />
<code> $ git clone https://github.com/didpurwanto/feat_visualization</code><br />

2. install saliency by <br />
<code >$ pip install saliency </code><br />

3. edit the axis of visualization into 3, which originally set as 2. <br />
Go to folder <code >/usr/local/lib/python2.7/dist-packages/saliency </code><br />
<code >line 23: axis = 3, instead of axis = 2. </code><br />

4. put this folder project inside the I3D/experiments/dataset/ along with the train.py and test.py. <br />

<h3>Run the code</h3> 
<code >$ python feat_visualization.py</code> <br />
The results are stored in <code >.visualization/results/. </code><br /><br />

Input image: <br />
<img width="200px"  src="./visualization/results/TVs_Best_Kisses_Top_50_52_to_41_kiss_h_nm_np2_le_goo_1/orig_0001.jpg"><br />
<img width="200px"  src="./visualization/results/TVs_Best_Kisses_Top_50_52_to_41_kiss_h_nm_np2_le_goo_1/orig_0003.jpg"><br />
<img width="200px"  src="./visualization/results/TVs_Best_Kisses_Top_50_52_to_41_kiss_h_nm_np2_le_goo_1/orig_0005.jpg"><br />
<img width="200px"  src="./visualization/results/TVs_Best_Kisses_Top_50_52_to_41_kiss_h_nm_np2_le_goo_1/orig_0025.jpg"><br />



Output image:<br />
<img width="200px"  src="./visualization/results/TVs_Best_Kisses_Top_50_52_to_41_kiss_h_nm_np2_le_goo_1/maps_0001.jpg"><br />
<img width="200px"  src="./visualization/results/TVs_Best_Kisses_Top_50_52_to_41_kiss_h_nm_np2_le_goo_1/maps_0003.jpg"><br />
<img width="200px"  src="./visualization/results/TVs_Best_Kisses_Top_50_52_to_41_kiss_h_nm_np2_le_goo_1/maps_0005.jpg"><br />
<img width="200px"  src="./visualization/results/TVs_Best_Kisses_Top_50_52_to_41_kiss_h_nm_np2_le_goo_1/maps_0025.jpg"><br />




This code is inspired by PAIRML Saliency library <a href="https://github.com/PAIR-code/saliency/blob/master/Examples.ipynb">link</a> <br />

Thanks to 吳侒融.