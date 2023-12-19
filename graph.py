from utils import *

# Example usage
plot_named_tuples([(8.13, 39.6, "hidT=256"), (7.83, 37.96, "hidS=32"), (11.80, 42.06, "hidS=32,hidT=256"), (17.38, 47.10, "Simplest"),
                   (8.9, 40.66, "NT=4"), (5.82, 36.35, "baseline"), ])
models = ['baseline','hid256', 'hidS32', 'NT4', 'Simplest' ]
values = [36.35, 36.70, 35.77,37.09,37.38]
plot_bar_chart(models, values, 'Model Performance Comparison', 'Models', 'MSE Loss')