# Timegpt

This is a model designed by **Nixtla** for time series prediction. This document is intended for personal learning and exploration of the reasoning behind TimeGPT. Most of the content on this document is from '[Garza A, Challu C, Mergenthaler-Canseco M. TimeGPT-1[J]. arXiv preprint arXiv:2310.03589, 2023.](https://arxiv.org/abs/2310.03589)'.

Website of Nixtla: https://www.nixtla.io/docs/forecasting/timegpt_quickstart

Github of Nixtla: https://github.com/Nixtla

===============================================================================
Update of Forecasting using Timegpt:

I tried to [forecast the daily 10yr and 2yr treasury yield] using Timegpt. But the result seems not that good. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/e1ed4ffc-81c3-418a-9942-f8eee9c4c661" width="100%" />
  <br>
  <b>Figure 1:</b> 10-Year Treasury Yield Forecast (TimeGPT)
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/cb78ce08-9aad-4f4d-8671-f1f678079468" width="100%" />
  <br>
  <b>Figure 1:</b> 2-Year Treasury Yield Forecast (TimeGPT)
</p>

To inspect the results, I choose to compare with number from [econforecasting](https://econforecasting.com/forecast/t10y). The methodoly can be found in https://econforecasting.com/static/research/rate-forecasts-draft.pdf









===============================================================================





## Summary:

Time series can differ in:

* Frequency / Sparsity / Trend / Seasonality / Stationarity / Heteroscedasticity

These characteristics introduce different levels of complexity for both local and global models. The goal of timegpt is to build a foundation-level forecasting model, which can process time series with various frequencies and characteristics, and support a range of input sizes and forecasting horizons.

This adaptability is largely thanks to the flexible and powerful Transformer architecture it is built upon. -- Not based on existing LLM, TimeGPT follows the same principle of training a large transformer model on a vast dataset.

## Dataset:

Scale:

* Over 100 billion time series data points.

Diversity of Domains:

* Finance / Economics / Demographics / Healthcare / Weather / IoT sensor data / Energy / Web traffic / Sales / Transportation / Banking

Missing values:

* TimeGPT requires time series data without missing values. So for some dates with no value, you need to [fill them](https://www.nixtla.io/docs/data_requirements/missing_values)

## Training process

TimeGPT underwent a **multi-day training period** on a cluster of NVIDIA A10G GPUs. 

Process:
* "During this process, we carried out extensive hyperparameter exploration to optimize learning rates, batch sizes, and other related parameters.  We observed a pattern in alignment with findings from [Brown et al.,2020], where a larger batch size and a smaller learning rate proved beneficial. Implemented in PyTorch, TimeGPT was trained using the Adam with a learning rate decay strategy that reduced the rate to 12% of its initial value."

Key Take away:
* Use smaller dataset(finance, economics, banking) as example to get shorter training period.  
* Use PyTorch to build the Transformer model, manage the training loop, compute loss, and perform backpropagation.
* Use Adam optimizer. Adam (Adaptive Moment Estimation) is a widely used gradient descent optimization algorithm.
* Use Learning Rate Decay Strategy. Start training with a higher learning rate to learn global patterns quickly; Use a smaller rate later for fine-tuning and stable convergence.
* Large Batch Size + Small Learning Rate. Use a larger batch size to improve GPU efficiency and provide more stable gradient estimates. Use a smaller learning rate to enhance training stability and generalization.

## Conformal prediction

This is a non-parametric framework, offers a compelling approach to generating prediction intervals with a pre-specified level of coverage accuracy. (I think this is similar to confidence interval), but conformal prediction does not require strict distributional assumptions, making it more flexible and agnostic to the model or time series domain. 

## Testing

### Testing data:

Instead of spliting each time serie data into trainning data, validation data and testing data, timegpt test model on over 300 thousand time series from multiple domains, including finance, web traffic, IoT, weather, demand, and electricity

Because the main property of Timegpt is the capability to accurately predict completely novel series. So they use large and diverse set of time series that were **never seen by the model during training** as testing data. 

### Benchmark models:

TimeGPT was benchmarked against a broad spectrum of baseline, statistical, machine learning, and neural forecasting models to provide a comprehensive performance analysis.

* Baselines and statistical models are individually trained on each time series of the test set, utilizing the historical values， preceding the last forecasting window.

* Machine learning and deep learning models are trained using a "global model approach".

  >**Local Model**: **A separate model** is trained for **each individual time series**. For example, if there are 1,000 time series in the test set, then 1,000 separate models are trained.
  
  >**Global Model**: **A single model** is trained to handle **all time series** at once. It is trained only once and used to make predictions across the entire dataset. 

* Some popular models like Prophet [Taylor and Letham, 2018] and ARIMA were excluded from the analysis due to their prohibitive computational requirements (maybe stationary requirement) and extensive training times.

Below list all the models used as comparison to Timegpt:

🟠 1. Baseline Models
Simple, non-trainable models often used as benchmarks.

| Method Name      | Description                                                  |
|------------------|--------------------------------------------------------------|
| **ZeroModel**    | Always predicts zero (or near-zero); used as an extreme baseline. |
| **HistoricAverage** | Predicts the average of past historical values.            |
| **SeasonalNaive** | Predicts the value from the same position in the previous seasonal cycle. |

---

🔵 2. Statistical Models
Traditional forecasting methods, typically trained individually per time series.

| Method Name     | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| **Theta**       | Model based on trend and exponential smoothing; strong in M3 competition.  |
| **DOTheta**     | Dynamic Theta version that auto-optimizes parameters for varied series.     |
| **ETS**         | Exponential Smoothing State Space Model (Error, Trend, Seasonality).        |
| **CES**         | Complex Exponential Smoothing, an extension of ETS.                        |
| **ADIDA**       | Aggregated Demand Intermittent Demand Approach, for intermittent demand.    |
| **IMAPA**       | Intermittent Multiple Aggregation Prediction Algorithm, suited for sparse data. |
| **CrostonClassic** | Classic method designed for intermittent demand forecasting.              |

---

🟢 3. Machine Learning Models
Feature-based models that learn from multiple series in a global way.

| Method Name | Description                                     |
|-------------|-------------------------------------------------|
| **LGBM**   | LightGBM, a gradient boosting tree model ideal for feature-based forecasting. |

---

🔴 4. Deep Learning Models
Neural network-based models suited for large-scale, complex forecasting.

| Method Name  | Description                                                                 |
|--------------|------------------------------------------------------------------------------|
| **LSTM**     | Long Short-Term Memory networks, classic RNN architecture for forecasting.  |
| **DeepAR**   | Autoregressive RNN model predicting full probabilistic distributions.       |
| **TFT**      | Temporal Fusion Transformer; combines attention and time-aware features.    |
| **NHITS**    | Multi-resolution deep residual forecasting model for long-horizon tasks.    |

---

### Evaluation value:

relative Mean Absolute Error (rMAE) and the relative Root Mean Square Error (rRMSE), both normalized against the performance of the Seasonal Naive model. 

<img width="1079" height="159" alt="image" src="https://github.com/user-attachments/assets/c867ebb3-f675-4706-927f-2f9a2f2e08ab" />

## Transfer learning

Transfer learning is a machine learning technique where a model that has already been trained on one task (usually a large, general dataset) is reused for a different but related task. Instead of training a model from scratch—which takes time and data, you transfer knowledge from the pretrained model to your task.

🧭 Two Main Approaches

1. 🟣 Zero-Shot Learning

**Zero-shot learning** means using a pre-trained model *without any additional training* for a new task.

✅ Description
- Apply the model directly on unseen time series.
- Leverages the model’s generalization ability from pretraining.

🔍 Pros
- ✅ No training required
- ⚡ Fast deployment
  * (For zero-shot inference, internal tests for Timegpt recorded an average GPU inference speed of **0.6 milliseconds per series**, which nearly mirrors that of the simple Seasonal Naive. As points of
comparison, parallel computing-optimized statistical methods, which, when complemented with Numba compiling, averaged a speed of **600 milliseconds per series** for training and inference. On the other hand, global models such as LGBM, LSTM, and NHITS demonstrated a
more prolonged average of **57 milliseconds per series**, considering both training and inference. Due to its zero-shot capabilities, TimeGPT outperforms traditional statistical methods and global models with total speed by orders of magnitude.)

- 🧪 Useful for quick experimentation or when data is scarce

⚠️ Cons
- ❗ May be less accurate for very domain-specific data
- 🚫 Limited customization

---

2. 🟠 Fine-Tuning

**Fine-tuning** involves continuing to train the pre-trained model on a smaller, domain-specific dataset.

✅ Description
- Adjusts the model parameters on your task-specific data.
- Customizes the model's understanding to better suit your needs.

💡 Example
You have proprietary IoT sensor data for industrial equipment. Fine-tuning TimeGPT on this data helps it learn machine-specific patterns.

🔍 Pros
- 🎯 High accuracy on specific tasks
- 🔧 Better control over model behavior
- 💪 Learns domain-specific nuances

⚠️ Cons
- 🖥️ Requires compute resources
- 🧠 Needs tuning of hyperparameters (learning rate, batch size, etc.)
- 📉 Risk of overfitting on small datasets

---
