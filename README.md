# stocksGAN

The original code for the project can be found here: https://github.com/rshea33/FA-691/tree/main/Final%20Project

## Basic Usage

The main file is `main.py`. It takes in the following arguments:

- `-t` or `--ticker` - The ticker symbol of the stock to use.

- `-sd` or `--start_date` - The start date of the data to use.

- `-ed` or `--end_date` - The end date of the data to use.

- `-n` or `--n_iter` - The number of iterations of the Random Search.

- `-v` or `--verbose` - The verbosity level of the output (0, 1, or 2).

- `-d` or `--default` - Whether to use the default hyperparameters or not.

- `-s` or `--save` - The name of the file to save the model to.

- `-ns` or `--n_samples` - The number of samples to generate.

For an example, run the following command:

`> python3 main.py -t TSLA -sd 2019-01-01 -ed 2020-01-01 -n 100 -v 2 -d True -s model.pt -ns 100`

## About

The goal of this project is to test the ability of GANs to generate realistic financial time-series data, in an effort to replace other synthetic implementations, such as geometric Brownian motions. Why? Because GBMs are unrealistic due to the assumptions about the underlying returns, mainly that they are i.i.d. Synthetic data that mimics the asset's underlying structure is crucial for more accurate approximations of risk metrics, specifically Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR). This is because non-i.i.d. events, such as volatility clustering, can lead to significant deviations from historical metrics, making accurate risk assessments difficult. The ability for a neural network to capture these structures could lead to more accurate approximations.

The framework of the project is as follows:

1. Gather stock data and convert to log returns

2. Segment the data into $t$ consecutive days (for the original project, $t = 5$ as the goal was to find a structure in a short time frame, and then if that works, scale it up)

3. Train the GAN on the segmented data

4. Validate the data using a separate model (LightGBM). I did it this way as I could train it with GBM paths and have a baseline to compare it to.

## TODO

- Increase regularization efforts

    - Increase Dropout
    
    - Implement Early Stopping
    
    - Change the model's underlying structure
    
- Use one stock, not all in S&P 500

- Experimant with different neural nets

    - RNNs / LSTMs
    
    - CNNs

- Add more inputs rather than just log returns

    - Signals for vol, the model did not do a good job of modeling noise
    
        - Standardized High - Close for each day
        
        - Volume
        
- Make hyperparameter tuning more efficient (Bayesian Optimization)

    - Currently using Random Search of hyperparams
    
- Run on a `while True` a.k.a. create a ton of models and hope one does well

    - Was only able to create 13 due to time constraints, issues
    
- Once it works on the small time frame, increase the horizon

## Most Recent Results (updated 2023-03-03)

- The GBM test accuracy is at 0.779. The main goal is to beat this using a GAN.

- The best FFNN GAN accuracy was 0.92 (20-Fold CV, $\pm 0.03$). This is **not** better as the LightGBM model is able to relatively easily tell the difference between what is real and fake.
