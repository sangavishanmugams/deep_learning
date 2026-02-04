# Advanced Time Series Forecasting with Neural ODEs and Uncertainty Quantification

This project explores time series forecasting using Neural Ordinary Differential Equations (Neural ODEs) and compares their performance with a traditional LSTM model. The dataset used is a synthetically generated non-linear time series containing multiple seasonal patterns, trend shifts, noise, and occasional shocks to mimic real-world complexity.

The LSTM model acts as a baseline and learns step-by-step temporal dependencies in discrete time. In contrast, the Neural ODE models the hidden state as a continuous-time dynamic system by learning the rate of change of the latent representation and solving it using a numerical ODE solver.

To make the predictions more informative, Monte Carlo Dropout is used during inference to estimate uncertainty by running multiple stochastic forward passes and constructing prediction intervals.

In terms of performance, the Neural ODE achieved around 7% improvement in RMSE and about 9% improvement in MAE compared to the LSTM baseline. The results suggest that continuous-time modeling can better capture smooth, structured non-linear dynamics and provide more stable forecasts, along with uncertainty estimates.

Overall, the project demonstrates the practical implementation of Neural ODEs for time series forecasting, highlights their advantages over discrete-time models in certain scenarios, and integrates uncertainty quantification in a clean, production-style workflow.
