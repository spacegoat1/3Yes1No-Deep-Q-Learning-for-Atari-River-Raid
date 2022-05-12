from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import os

# PLEASE SET YOUR OWN WORKING_DIRECTORY WHEN RUNNING LOCALLY
RUNNING_LOCALLY = False

WORKING_DIRECTORY = "/home/yash/Desktop/Courses/CS2470/Final_Project/working_dir/"

if not RUNNING_LOCALLY:
    os.chdir("/home/yash/")
    print("Current Directory ->", os.getcwd())

    WORKING_DIRECTORY = "/home/yash/working_dir/"

def write_to_log(statement, include_blank_line=False):
    try:
        with open(LOG_FILE, "a") as myfile:
            if include_blank_line:
                myfile.write("\n\n" + statement)
            else:
                myfile.write("\n" + statement)
    except:
        # Running this locally may cause errors, and isn't required
        pass

LOG_FILE = WORKING_DIRECTORY + "log_file.txt"


def render_image(data):
    plt.imshow(data, interpolation='nearest')
    plt.show()

def describe_arr(arr):
    out = f"count:{len(arr)}, min:{np.min(arr)}, max:{np.max(arr)}, mean:{np.mean(arr)}, std:{np.std(arr)}"
    print(out)

def plot_line(arr, title, fig_size=(7,5), dpi=300):
    x = np.arange(len(arr))
    figure(figsize=fig_size, dpi=dpi)
    plt.plot(x, arr)
    plt.title(title)
    plt.show()

def save_models_and_arrays(curr_model_name, curr_model_name_target, curr_model, curr_model_target,
                            reward_history, n_steps_history):
    # Save models
    model_path = WORKING_DIRECTORY + "model/" + curr_model_name
    target_model_path = WORKING_DIRECTORY + "model/" + curr_model_name_target
    curr_model.save(model_path)
    curr_model_target.save(target_model_path)
    np.save(WORKING_DIRECTORY + curr_model_name + "_reward_history", reward_history)
    np.save(WORKING_DIRECTORY + curr_model_name + "_n_steps_history", n_steps_history)
