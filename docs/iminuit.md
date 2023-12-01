## Iminuit Docs 

| Parameter | Description |
| --- | --- |
| FCN | The minimum value of the chi-squared function. |
| Nfcn | The number of function calls made during the minimization process. |
| EDM | The estimated distance to the minimum, which should be below the goal (0.0002 in this case). |
| Valid Minimum, Valid Parameters, etc. | Flags indicating the quality of the fit and whether the fit has converged. |

The table also shows the fitted parameter values, their Hesse errors, and the limits for each parameter:

| Parameter | Description |
| --- | --- |
| n_s | The normalization factor for the Gaussian signal. |
| n_b | The normalization factor for the linear background. |
| mu | The mean of the Gaussian signal. |
| sigma | The standard deviation of the Gaussian signal. |
| a | The slope of the linear background. |
| b | The intercept of the linear background. |
