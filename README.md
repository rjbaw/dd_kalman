# Kalman Examples
https://arxiv.org/abs/2210.14878

## Setup
```
pip install .
```

## Examples
```
python3 examples/example1.py
```
### Response
#### State x
Mass = 20 kg
![Response](img/x_1.png)
Mass = 5 kg
![Response](img/x_2.png)

#### State y
Mass = 20 kg
![Response](img/y_2.png)
Mass = 5 kg
![Response](img/y_2.png)

### Error
Mass = 20 kg
![Error](img/gradient_1.png)
Mass = 5 kg
![Error](img/gradient_2.png)

### Kalman Gain
Mass = 20 kg
![Gain](img/gain_1.png)
![Gain](img/msegain_1.png)
![Gain](img/gain2_1.png)
![Gain](img/msegain2_1.png)
Mass = 5 kg
![Gain](img/gain_2.png)
![Gain](img/msegain_2.png)
![Gain](img/gain2_2.png)
![Gain](img/msegain2_2.png)


