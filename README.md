# Kaggle/Santander Customer Satisfaction

<div align="center">
  <img src="./input/front.png"><br><br>
</div>

## Abstract
[Santander Customer Satisfaction Competition](https://www.kaggle.com/c/santander-customer-satisfaction)

- Host : [**Santander Bank**](https://www.santanderbank.com/us/personal), a British bank, wholly owned by the Spanish Santander Group.
- Prize : $ 60,000
- Problem : Binary Classification
- Evaluation : [AUC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- Period : Mar 2 2016 ~ May 2 2016 (61 days)

Santander Bank is asking Kagglers to help them identify dissatisfied customers early in their relationship. Doing so would allow Santander to take proactive steps to improve a customer's happiness before it's too late.

In this competition, you'll work with hundreds of anonymized features to predict if a customer is satisfied or dissatisfied with their banking experience.

## Result
| Submission | CV LogLoss | Public LB | Rank | Private LB | Rank |
|:----------:|:----------:|:---------:|:----:|:----------:|:----:|
| bare_minimum | 0.800430 | | | 0.785805 | **3958** |
| kweonwooj | 

## How to Run

**[Data]** 

Place data in ```input``` directory. You can download data from [here](https://www.kaggle.com/c/santander-customer-satisfaction/data).

**[Code]**

Above results can be replicated by runinng

```
python code/main.py
```
for each of the directories.

Make sure you are on Python 3.5.2 with library versions same as specified in requirements.txt

## Add requirements.txt

**[Submit]**

Submit the resulting csv file [here](https://www.kaggle.com/c/santander-customer-satisfaction/submissions/attach) and verify the score.

## Expected Result

for bare minimum
<div align="center">
  <img src="./input/bare_minimum.png"><br><br>
</div>

for reduced version of kweonwooj
<div align="center">
  <img src="./input/kweonwooj.png"><br><br>
</div>

## Objective
- modularize the codes following the best practices
- be able to reproduce, re-use the modules developed here
- use annotation + object detection method, it is key methodology in image classification

## Winning Methods
# update this section
* 1st by jacobkie [Link](https://www.kaggle.com/c/state-farm-distracted-driver-detection/forums/t/22906/a-brief-summary/131467#post131467)
    - Pre-trained VGG16, modified VGG16_3 (Single Model got LB score around 0.3)
    - VGG16_3 trained with two selected regions of interests(head and radio area) together with original image
    - K-Nearest Neighbor Average: Uses last Maxpool layer(pool5) of VGG16 to map test image to 512*7*7 coordinate, use distances in this space to define similarity, weighted average of predictions together with 10-NN improves single model score by 0.10~0.12
    - Ensemble average for each category separately. Models with top 10% cross-entropy loss associated with category are chosen. Outperforms simple arithmetic/geometric average.
    - Segment Average: Divide test images into group using pool5-feature space, if one group displayes consistency and confidence, renormalize all the images in that group to share predictions.
* 3rd by BRAZIL_POWER (0.08877 > 0.09058) [Link](https://www.kaggle.com/c/state-farm-distracted-driver-detection/forums/t/22631/3-br-power-solution)
    - Ensemble of 4 models - ResNet152, VGG16
    - Use synthetic test image = image + nearest neighbor images
   
* 5th by DZS Team (0.10252 > 0.12144) [Link](https://www.kaggle.com/c/state-farm-distracted-driver-detection/forums/t/22627/share-your-best-single-model-score-on-public-lb)
    - Synthetic Train Images = Half + Half of train image. 5 Million synthetic image to train GoogleNet
   
* 10th by toshi-k (0.14354 > 0.14911) [Link](https://github.com/toshi-k/kaggle-distracted-driver-detection)
    - 20 Models for Ensembling
    - CNN to detect driver body pixel (Semantic Segmentation)
    - Crop driver region(by bounding box) and use another classifier on this region
