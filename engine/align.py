"""
Utility functions for aligning and comparing datasets (e.g., true_data and user_data).
"""
import numpy as np
import logging

def make_same_length(job_id: str, true_data: np.ndarray, user_data: np.ndarray) -> np.ndarray:
    """
    Align the shape of the trueDataSet to that of the userDataSet (by extending or truncating trueDataSet).
    """
    logger = logging.getLogger(job_id).getChild("make_same_length")
    if true_data.size == 0 or user_data.size == 0:
        logger.error("            One of the datasets is empty. Cannot align shapes.")
        return true_data
    logger.info(
        f"            Aligning shapes of userDataSet ({user_data.shape}) and trueDataSet ({true_data.shape})."
    )
    # if the trueDataSet is shorter, extend it to match the length of the userDataSet
    if true_data.shape[1] < user_data.shape[1]:
        length_difference = user_data.shape[1] - true_data.shape[1]
        logger.info(
            f"            The trueDataSet is shorter than the userDataSet, so we will extend it (by {length_difference} time steps) to match the length of the userDataSet."
        )
        final_concentrations = true_data[:, -1].reshape(-1, 1)
        patch = np.repeat(final_concentrations, length_difference, axis=1)
        true_data = np.append(
            true_data,
            patch,
            axis=1,
        )
        logger.info(f"            The trueDataSet is now {true_data.shape}.")
    elif true_data.shape[1] > user_data.shape[1]:
        logger.info(
            "            The trueDataSet is longer. Truncating it to match the length of the userDataSet."
        )
        true_data = true_data[:, : user_data.shape[1]]
    return true_data

def align_for_scoring(true_data: np.ndarray, user_data: np.ndarray) -> tuple:
    """
    Align both arrays to the same shape for scoring. Truncate to the minimum length.
    Returns (true_data_aligned, user_data_aligned)
    """
    min_len = min(true_data.shape[1], user_data.shape[1])
    return true_data[:, :min_len], user_data[:, :min_len]
