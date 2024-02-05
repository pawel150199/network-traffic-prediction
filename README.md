# Network Traffic Prediction

## Dataset
The data was captured in Europe - dataset Euro28 and in USA - dataset US26 in optical network infrastracture and in the next step was generated using original one.

## Euro28

<img src="img/euro28.png" width="500px" height="450px"><img>

## US26

<img src="img/us26.png" width="500px" height="450px"><img>

## Experiments

### Checks if preprocssing can improve
This experiment checks if preprocessing can give us better results using simple models.

### Compare many models both deep and simple models
This experiment checks which models is the best for that purpose.


## How to run experiment?
To run first experiment run `python3 experiment_1.py` after that in `results` directory you will have a few npy files: `experiment1_<dataset>-mean.npy`, `experiment1_<dataset>-std.npy`, `experiment1_<dataset>.npy`. First file has mean data of experiment, second has std of the experiment and the last one has all data of each iteration.
* To run second experiment run `python3 experiment_2.py`
If you would like to run 


