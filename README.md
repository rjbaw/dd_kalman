# Kalman Examples
https://arxiv.org/abs/2210.14878

[Colab Demo](https://colab.research.google.com/drive/1KPn_slEGinYSy-zqQ662MZrHwI7StI4N?usp=sharing)

## Setup
```
pip install .
```

## Examples
```
python3 examples/example1.py
```

Mass = 20 kg  
Damping Coefficient = 4 N/(ms^-1)  
Spring Stiffness = 2 N/m  
Force = 10 N  
Number of Mass-Spring-Damper System in Series = 3  

### Response
#### Response y
![Response](img/y.png)
![Response](img/scatter_loss.png)
![Response](img/scatter_axis.png)
![Response](img/y2.png)
![Response](img/scatter_loss2.png)
![Response](img/scatter_axis2.png)

### Training loss
![Error](img/trainloss.png)

### Kalman Gain vs ARE solution
![Gain](img/gain.png)
![Gain](img/msegain.png)

### MSE Kalman Gain for different n_series
![nseries_Error](img/gain_mse_nseries_time.png)
![nseries_Error](img/gain_mse_nseries.png)

### Filters for Audio file
![audio](img/audio_y.png)
![audio](img/audio_y2.png)

### Comparison
![compare](img/mse_prediction.png)
![compare](img/dspo_vs_regret.png)



