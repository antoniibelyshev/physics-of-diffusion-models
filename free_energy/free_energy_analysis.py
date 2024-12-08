import numpy as np
from numpy.typing import NDArray
from typing import Any
import utils
from scipy.optimize import curve_fit # type: ignore
import matplotlib.pyplot as plt


ndarray = NDArray[np.float32 | np.float64]


def sigmoid(
    log_temp: ndarray,
    a: float,
    k: float,
    log_temp_c: float,
    b: float,
) -> ndarray:
    return a / (1 + np.exp(-k * (log_temp - log_temp_c))) + b # type: ignore


def analyze_free_energy(avg_free_energy: ndarray, temp: ndarray) -> dict[str, Any]:
    S = -utils.finite_diff_derivative(temp, avg_free_energy)

    popt, _ = curve_fit(sigmoid, np.log(temp[1:]), S, p0=[1, 1, 1, S.mean()])
    a, k, log_temp_c, b = popt

    sigmoid_params = {
        "a": a,
        "k": k,
        "log_temp_c": log_temp_c,
        "b": b,
    }
    transition_characteristics = {
        "temp_c": np.exp(log_temp_c),
        "transition_start": np.exp(log_temp_c - np.log(1 / 0.1 - 1) / k),
        "transition_end":  np.exp(log_temp_c - np.log(1 / 0.9 - 1) / k),
    }

    return {
        "avg_free_energy": avg_free_energy,
        "temp": temp,
        "S": S,
        "sigmoid_params": sigmoid_params,
        "S_est": sigmoid(np.log(temp[1:]), *popt),
        "transition_characteristics": transition_characteristics,
    }


def present_analysis(analysis_results: dict[str, Any]) -> None:
    S = analysis_results["S"]
    temp = analysis_results["temp"]

    transition_characteristics = analysis_results["transition_characteristics"]
    temp_c = transition_characteristics["temp_c"]
    transition_start = transition_characteristics["transition_start"]
    transition_end = transition_characteristics["transition_end"]

    S_est = analysis_results["S_est"]

    plt.figure(figsize=(8, 6))
    plt.scatter(temp[1:], S, label="Noisy Data")
    plt.plot(temp[1:], S_est, label="Fitted Sigmoid", color="red")
    plt.axvline(x=temp_c, color="purple", linestyle=":", label=f"Critical temperature: {temp_c:.2f})")
    plt.xlabel("temperature")
    plt.ylabel("-dF/dT")
    plt.title("Sigmoid Fit and Transition Point")
    plt.xscale("log")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Critical temperature: {temp_c:.2f}")
    print(f"Noise std: {np.sqrt(temp_c):.2f}")
    print(f"Transition interval: [{transition_start:.2f}, {transition_end:.2f}]")
