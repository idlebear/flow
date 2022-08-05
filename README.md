# Flow

## Using a convolutional Neural Net to predict future positions

The code in this repository is working example of a CNN trained to predict the future motion of other vehicles based on 1 second of LIDAR input data.  The backbone neural net is based on the work of [Casas et al.](http://openaccess.thecvf.com/content/CVPR2021/html/Casas_MP3_A_Unified_Model_To_Map_Perceive_Predict_and_Plan_CVPR_2021_paper.html)[^1]

The methods to generate the dataset from [Carla](https://carla.org/) LIDAR sensors are not currently included.

[^1]:Casas, Sergio, Abbas Sadat, and Raquel Urtasun. "Mp3: A unified model to map, perceive, predict and plan." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021.
