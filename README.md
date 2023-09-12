# Learning Visual Representation Clusters for Cross-View Geo-Location
Cross-view geo-location is crucial for drone navigation in GNSS-denied environments, aerial surveillance, and environmental monitoring. Current work mainly frames the geo-location task as a classification task by treating images from the same locations as the same category, and focus on pushing the representation distances of different categories, neglecting the similarity of intra-category samples. To solve this problem, we propose to learn representations invariant to views and platforms, so that cross-view images can cluster together.
## Dataset & Preparation
### Dataset
To reproduce the results, please download [University-1652](https://github.com/layumi/University1652-Baseline) dataset.
### Preparation
Our methodology is based on a single-branch network, employing [OSnet](https://github.com/KaiyangZhou/deep-person-reid) pre-trained on ImageNet to extract the feature map. 
If you want to use the pre-trained model, you can also download it from OSnet's [Model Zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html).
## Train & Evaluation
### Train
```  
main.py
```
### Test
```  
script/anntest.py
```
### Evaluation
```
script/evaluate_gpu.py
```
## Cite:
If you use this code for your research, please cite:
```

```
