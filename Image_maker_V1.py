import os, math, itertools, time
import numpy as np
import tkinter as tk  # GUI window
import random

from collections import Counter
import math
from statistics import mean, stdev
from tkinter import ttk , messagebox # input fields
from tkinter import filedialog as fd  # file open
from tkinter.messagebox import showinfo  # info box
from PIL import Image, ImageTk, ImageDraw
from matplotlib import pyplot as plt
from datetime import datetime
from collections.abc import Iterable
from collections import defaultdict
from scipy.signal import convolve2d
from scipy.ndimage import generic_filter, binary_dilation, binary_erosion, minimum_filter, maximum_filter
from math import sqrt
from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, euclidean, minkowski, hamming
from sklearn.naive_bayes import GaussianNB # Correct import
from sklearn.preprocessing import StandardScaler


window = tk.Tk()
window.config(background="#e7e7e7")
window.geometry("1000x800")
window.resizable(False, False)
window.title("Program zdjeciowy")

image_name = ""
image_location = ""
# threshold = 0  # zmienia binaryzacje, mniej = wiecej bialego
image = []
state = "o"
stretch = tk.BooleanVar()
stretch.set(False)
equalise = tk.BooleanVar()
equalise.set(False)
otsu = tk.BooleanVar()
otsu.set(False)
window_size_var = tk.StringVar(value="")
k_value_var = tk.StringVar(value="")
r_value_var = tk.StringVar(value="")
r_weight = tk.StringVar()
g_weight = tk.StringVar()
b_weight = tk.StringVar()
poin="1"
tolerance=tk.StringVar()
global_flood_enabled = False
compare_all=False
drawing_image = None
original_drawing_image = None
drawing_draw = None
drawing_tk_image = None
selected_color = [255, 0, 0, 255]  # Default red color
global_flood_mode = False

sample_paths = []
labels = []
parsed_samples = []
feature_vectors = []

all_keys = [chr(i) for i in range(65, 91)]  # A-Z
all_keys.append('SPACE')
all_keys.append('LSHIFT')
selected_method=''
selected_metric=''
kvalue=3

# mono_image = Image.fromarray(binary_array);  nie dziala, binary array: signed, PIL potrzebuje unsigned

def kill_UI():
    for widget in window.winfo_children():
        widget.destroy()
    window.update_idletasks()

def HELP_window(message):
    message_window = tk.Toplevel()
    message_window.title("Info")
    message_window.geometry("500x350")

    label = ttk.Label(message_window, text=message, wraplength=400)
    label.pack(pady=20)

    close_button = ttk.Button(message_window, text="Zamknij", command=message_window.destroy)
    close_button.pack(pady=10)


def select_file():  # wybiera plik (zdjecie)
    kill_UI()
    global full_path, dataframe, filename, image_location, image_name, image, timestamped_folder_path
    filetypes = (
        ('jpg files', '*.jpg'),
        ('All files', '*.*'))
    filename = fd.askopenfilename(
        title='Wybierz plik',
        initialdir='/',
        filetypes=filetypes)
    if filename:
        try:
            image_location, image_name = os.path.split(filename)
            full_path = os.path.join(image_location, image_name).replace('\\', '/')
            image = Image.open(full_path)
            showinfo(title="plik otwarty", message=f"plik otwarty: {filename}")
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            timestamped_folder_path = os.path.join(image_location, timestamp)
            timestamped_folder_path = os.path.normpath(timestamped_folder_path).replace('\\', '/')

            kill_UI()
            get_thresh()


        except Exception as e:
            showinfo(title="Blad", message=f"{e}")



def select_folder():
    kill_UI()
    global full_path, dataframe, filename, image_location, image_name, image, timestamped_folder_path, folder
    global sample_paths, labels


    # ──> Ask for a folder instead
    folder = fd.askdirectory(title="Wybierz folder z próbkami klawiatury")
    if folder:
        try:
            sample_paths = []
            labels = []

            for file in os.listdir(folder):
                if file.endswith('.txt') or file.endswith('.csv'):
                    full_path = os.path.join(folder, file).replace('\\', '/')
                    sample_paths.append(full_path)
                    labels.append(os.path.splitext(file)[0])

            labels = [label.lower().strip() for label in labels]
            showinfo(title="Folder przetworzony", message=f"Znaleziono {len(sample_paths)} plików.")
            classifier_buttons()

        except Exception as e:
            showinfo(title="Błąd folderu", message=f"{e}")

def set_metric(metric):
    global selected_metric
    selected_metric = metric

def set_method(method):
    global selected_method
    selected_method = method


    
def classifier_buttons():
    global btn_toggle_mode
    metric_mannhattan = ttk.Button(window, text="mannhattan metric", width=20, command=lambda: set_metric("mannhattan"))
    metric_mannhattan.place(x=200, y=100)
    metric_euclidean = ttk.Button(window, text="euclidean metric", width=20, command=lambda: set_metric("euclidean"))
    metric_euclidean.place(x=200, y=200)
    metric_chebyshev = ttk.Button(window, text="chebyshev metric", width=20, command=lambda: set_metric("chebyshev"))
    metric_chebyshev.place(x=200, y=300)

    metric_braycurtis = ttk.Button(window, text="braycurtis metric", width=20, command=lambda: set_metric("braycurtis"))
    metric_braycurtis.place(x=200, y=400)
    metric_canberra = ttk.Button(window, text="canbera metric", width=20, command=lambda: set_metric("canberra"))
    metric_canberra.place(x=200, y=500)
    metric_minkowski = ttk.Button(window, text="minkowski metric", width=20, command=lambda: set_metric("minkowski"))
    metric_minkowski.place(x=200, y=600)
    metric_hamming = ttk.Button(window, text="hamming metric", width=20, command=lambda: set_metric("hamming"))
    metric_hamming.place(x=200, y=700)

    method_knn = ttk.Button(window, text="k-NN method", width=20, command=lambda: set_method("k-NN"))
    method_knn.place(x=400, y=300)
    method_Nbayes = ttk.Button(window, text="Naive Bayes", width=20, command=lambda: set_method("Naive Bayes"))
    method_Nbayes.place(x=400, y=500)

    btn_toggle_mode = ttk.Button(window, text='compare all?', width=20, command=toggle_mode)
    btn_toggle_mode.place(x=600, y=150)

    calculate_metrics = ttk.Button(window, text="Calculate", width=20, command=lambda: run_keystroke_analysis(method=selected_method,num_k=kvalue,distance_metric=selected_metric ,custom_keys=None,compare_mode=compare_all))
    calculate_metrics.place(x=600, y=400)


    kval = tk.StringVar()
    feedback4 = ttk.Label(window, text="", background="#e7e7e7")
    feedback4.place(x=600, y=350)
    entry11 = ttk.Entry(window, textvariable=kval, width=30)
    entry11.place(x=600, y=300)
    kval.trace_add("write", lambda *args: validate_kval(kval, feedback4, calculate_metrics))


def validate_kval(kval, feedback4, calculate_metrics):  # walidacja tego progu
    global kvalue
    val = kval.get().strip()

    if val.isdigit() and (0 <= int(val) <= 255) and (int(val)%2==1):
        kvalue = int(val)
        feedback4.config(text="Prawidlowe", foreground="green")
        calculate_metrics.config(state="normal")
    else:
        feedback4.config(text="Nie Prawidlowe \n\n\n (wartosci nieparzyste tylko od 0 do 255)", foreground="red")
        calculate_metrics.config(state="disabled")


def toggle_mode():
    global compare_all, btn_toggle_mode  # Both need to be global

    compare_all = not compare_all
    if compare_all:
        btn_toggle_mode.config(text="tryb porównania Wł")
    else:
        btn_toggle_mode.config(text="tryb porównania Wył")




def run_keystroke_analysis(
                           method="k-NN",
                           num_k=3,
                           distance_metric="manhattan",
                           custom_keys=None,
                           compare_mode=False):
    global folder
    samples_dir=folder
    # Use custom keys if specified, otherwise use all keys
    target_keys = custom_keys if custom_keys is not None else all_keys

    print(f"=== KEYSTROKE CLASSIFIER ===")
    print(f"Directory: {samples_dir}")
    print(f"Method: {method}")

    # If compare_mode is True, run multiple configurations
    if compare_mode:
        # Store results for each configuration
        all_results = []

        # Define distance metrics to test
        distance_metrics = [
            "manhattan", "euclidean", "chebyshev",
            "braycurtis", "canberra", "minkowski", "hamming"
        ]

        # Try different k values and distance metrics
        k_values = [1, 3, 5]

        # Define key sets to try
        key_sets = [
            ['A', 'E', 'O', 'I', 'SPACE'],
            ['LSHIFT', 'A', 'T', 'E', 'N'],
            ['B', 'C', 'D', 'F', 'G'],
            ['SPACE', 'A', 'E', 'T', 'N', 'O', 'I', 'S', 'R']  # Common letters
        ]

        print("\n1. K-NEAREST NEIGHBORS")
        # Test all distance metrics with k=3 and all keys
        for metric in distance_metrics:
            accuracy = evaluate_classifier(samples_dir, all_keys, 3, metric)
            all_results.append((3, all_keys.copy(), accuracy, "k-NN", metric))
            print(f"k=3, metric={metric}, using all keys: Accuracy = {accuracy:.2%}")

        # Try different k values with best metric
        best_metric = max(all_results, key=lambda x: x[2])[4]
        print(f"\nBest metric so far: {best_metric}. Testing with different k values...")

        for k in k_values:
            if k != 3:  # Skip k=3 as we already tested it
                accuracy = evaluate_classifier(samples_dir, all_keys, k, best_metric)
                all_results.append((k, all_keys.copy(), accuracy, "k-NN", best_metric))
                print(f"k={k}, metric={best_metric}, using all keys: Accuracy = {accuracy:.2%}")

        # Try different key sets with best k and metric
        best_k = max(all_results, key=lambda x: x[2])[0]
        print(f"\nBest k so far: {best_k}. Testing with different key sets...")

        for key_set in key_sets:
            accuracy = evaluate_classifier(samples_dir, key_set, best_k, best_metric)
            all_results.append((best_k, key_set.copy(), accuracy, "k-NN", best_metric))
            print(f"k={best_k}, metric={best_metric}, keys={key_set}: Accuracy = {accuracy:.2%}")

        print("\n2. NAIVE BAYES")
        # Evaluate Naive Bayes with all keys
        nb_accuracy = evaluate_naive_bayes(samples_dir, all_keys)
        all_results.append((None, all_keys.copy(), nb_accuracy, "Naive Bayes", None))
        print(f"Naive Bayes, using all keys: Accuracy = {nb_accuracy:.2%}")

        # Try with different key sets
        for key_set in key_sets:
            nb_accuracy = evaluate_naive_bayes(samples_dir, key_set)
            all_results.append((None, key_set.copy(), nb_accuracy, "Naive Bayes", None))
            print(f"Naive Bayes, keys={key_set}: Accuracy = {nb_accuracy:.2%}")

        # Sort all results
        all_results.sort(key=lambda x: x[2], reverse=True)

        # Print the top configurations
        print("\n=== TOP FIVE BEST CONFIGURATIONS ===")

        # Prepare message for tkinter dialog
        top_results_msg = "=== TOP FIVE BEST CONFIGURATIONS ===\n\n"

        for i in range(min(5, len(all_results))):
            k, key_set, accuracy, method, metric = all_results[i]

            if method == "k-NN":
                result_line = f"#{i + 1}: Method={method}, k={k}, metric={metric}, Accuracy={accuracy:.2%}"
                print(result_line)
            else:
                result_line = f"#{i + 1}: Method={method}, Accuracy={accuracy:.2%}"
                print(result_line)

            top_results_msg += result_line + "\n"

            if len(key_set) > 10:
                keys_info = f"   Keys used: {len(key_set)} keys including {', '.join(key_set[:5])}... and others"
                print(keys_info)
            else:
                keys_info = f"   Keys used: {', '.join(key_set)}"
                print(keys_info)

            top_results_msg += keys_info + "\n\n"
            print()

        # Display comparison results in tkinter showinfo box
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showinfo("Keystroke Comparison Results", top_results_msg)
        root.destroy()

        # Return the best configuration
        if all_results:
            return all_results[0][2]  # Return the accuracy of the best configuration
        return 0.0

    # Otherwise, run a single configuration with specified parameters
    else:
        print(
            f"Keys: {target_keys if len(target_keys) <= 5 else f'{len(target_keys)} keys including {target_keys[:5]}...'}")

        if method == "k-NN":
            print(f"k value: {num_k}")
            print(f"Distance metric: {distance_metric}")
            accuracy = evaluate_classifier(samples_dir, target_keys, num_k, distance_metric)
            print(f"\nResults: Accuracy = {accuracy:.2%}")

            # Display results in tkinter showinfo box
            import tkinter as tk
            from tkinter import messagebox

            keys_str = ', '.join(target_keys) if len(
                target_keys) <= 5 else f"{len(target_keys)} keys including {', '.join(target_keys[:5])}..."
            result_msg = f"Method: {method}\nk value: {num_k}\nDistance metric: {distance_metric}\nKeys: {keys_str}\n\nAccuracy: {accuracy:.2%}"

            root = tk.Tk()
            root.withdraw()  # Hide the main window
            messagebox.showinfo("Keystroke Analysis Results", result_msg)
            root.destroy()

            return accuracy

        elif method == "Naive Bayes":
            accuracy = evaluate_naive_bayes(samples_dir, target_keys)
            print(f"\nResults: Accuracy = {accuracy:.2%}")

            # Display results in tkinter showinfo box
            import tkinter as tk
            from tkinter import messagebox

            keys_str = ', '.join(target_keys) if len(
                target_keys) <= 5 else f"{len(target_keys)} keys including {', '.join(target_keys[:5])}..."
            result_msg = f"Method: {method}\nKeys: {keys_str}\n\nAccuracy: {accuracy:.2%}"

            root = tk.Tk()
            root.withdraw()  # Hide the main window
            messagebox.showinfo("Keystroke Analysis Results", result_msg)
            root.destroy()

            return accuracy

        else:
            print(f"Error: Unknown method '{method}'. Please choose 'k-NN' or 'Naive Bayes'.")

            # Display error in tkinter showinfo box
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()  # Hide the main window
            messagebox.showerror("Keystroke Analysis Error",
                                 f"Unknown method '{method}'.\nPlease choose 'k-NN' or 'Naive Bayes'.")
            root.destroy()

            return 0.0


def naive_bayes_train(training_features, training_labels, target_keys=None):
    """
    Train a Naive Bayes classifier on keystroke data

    Args:
        training_features: List of feature dictionaries for training samples
        training_labels: List of labels for training samples
        target_keys: Keys to consider for features

    Returns:
        Dictionary of model parameters (priors and likelihoods)
    """
    if target_keys is None:
        target_keys = ['X', 'Y', 'Z']

    # Get unique classes (users)
    classes = list(set(training_labels))

    # Calculate class priors P(C)
    class_counts = Counter(training_labels)
    total_samples = len(training_labels)
    priors = {cls: class_counts[cls] / total_samples for cls in classes}

    # Calculate likelihood parameters (mean and stdev) for each feature in each class
    # We assume each feature follows a normal distribution within a class
    likelihoods = {}

    for cls in classes:
        likelihoods[cls] = {}
        # Get all samples for this class
        class_samples = [training_features[i] for i in range(len(training_features))
                         if training_labels[i] == cls]

        # Calculate statistics for each key
        for key in target_keys:
            # Extract values for this feature (key) across all samples of this class
            values = [sample.get(key, 0) for sample in class_samples]

            # Need at least 2 samples to calculate standard deviation
            if len(values) >= 2:
                # Calculate mean and standard deviation
                mean_val = mean(values)
                std_val = stdev(values)
                # Ensure std is not zero to avoid division by zero
                std_val = max(std_val, 0.0001)
            else:
                # If only one sample, use a default std
                mean_val = values[0] if values else 0
                std_val = 0.1  # Small default value

            likelihoods[cls][key] = {'mean': mean_val, 'std': std_val}

    return {'priors': priors, 'likelihoods': likelihoods}


def gaussian_probability(x, mean, std):
    """Calculate the probability of x using Gaussian PDF"""
    exponent = math.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent


def naive_bayes_classify(sample_features, model, target_keys=None):
    """
    Classify a sample using trained Naive Bayes model

    Args:
        sample_features: Features of the sample to classify
        model: Trained model parameters (priors and likelihoods)
        target_keys: Keys to consider for features

    Returns:
        Predicted label for the sample
    """
    if target_keys is None:
        target_keys = ['X', 'Y', 'Z']

    priors = model['priors']
    likelihoods = model['likelihoods']
    classes = list(priors.keys())

    # Calculate posterior probability for each class
    posteriors = {}

    for cls in classes:
        # Start with log prior
        posterior = math.log(priors[cls])

        # Add log likelihoods for each feature
        for key in target_keys:
            # Get sample value for this feature
            value = sample_features.get(key, 0)

            # Get likelihood parameters for this feature in this class
            params = likelihoods[cls].get(key, {'mean': 0, 'std': 0.1})
            mean_val = params['mean']
            std_val = params['std']

            # Calculate log likelihood
            try:
                likelihood = gaussian_probability(value, mean_val, std_val)
                # Avoid log(0) by setting a minimum likelihood
                likelihood = max(likelihood, 1e-10)
                posterior += math.log(likelihood)
            except (ValueError, OverflowError):
                # Skip this feature if there's a numerical error
                continue

        posteriors[cls] = posterior

    # Return class with highest posterior probability
    return max(posteriors.items(), key=lambda x: x[1])[0]


def evaluate_naive_bayes(samples_dir, target_keys=None):
    """
    Evaluate Naive Bayes classifier using leave-one-out cross-validation

    Args:
        samples_dir: Directory containing sample files
        target_keys: Keys to use for features

    Returns:
        Accuracy of the classifier
    """
    if target_keys is None:
        target_keys = ['X', 'Y', 'Z']

    # Get all sample files
    try:
        sample_files = [f for f in os.listdir(samples_dir) if
                        os.path.isfile(os.path.join(samples_dir, f)) and f.endswith(('.txt', '.csv'))]

        if not sample_files:
            print(f"No text or CSV files found in {samples_dir}")
            return 0
    except Exception as e:
        print(f"Error accessing directory {samples_dir}: {e}")
        return 0

    # Extract user labels and parse samples
    all_samples = []
    all_labels = []

    for file in sample_files:
        try:
            # Extract user label from filename
            user_label = file.split('_')[0]

            # Parse sample
            path = os.path.join(samples_dir, file)
            strokes = parse_sample(path)

            if strokes:
                # Extract features
                features = extract_features(strokes, target_keys)

                # Store sample and label
                all_samples.append(features)
                all_labels.append(user_label)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    if not all_samples:
        print("No valid samples could be processed")
        return 0

    if len(set(all_labels)) < 2:
        print("Need at least two different users for classification")
        return 0

    # Evaluate using leave-one-out cross-validation
    correct = 0
    total = 0

    for i in range(len(all_samples)):
        # Create training set without current sample
        training_features = all_samples[:i] + all_samples[i + 1:]
        training_labels = all_labels[:i] + all_labels[i + 1:]

        # Train Naive Bayes model
        model = naive_bayes_train(training_features, training_labels, target_keys)

        # Classify current sample
        predicted_label = naive_bayes_classify(all_samples[i], model, target_keys)

        # Check if prediction is correct
        if predicted_label == all_labels[i]:
            correct += 1
        total += 1

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0

    return accuracy





def parse_sample(path):
    strokes = []
    try:
        with open(path, 'r') as file:
            for line in file:
                # Clean and split
                parts = [p.strip() for p in line.strip().split(',')]
                if len(parts) != 3:
                    print(f"Skipping line in {path} due to incorrect number of fields: {line.strip()}")
                    continue  # Skip bad lines

                key = parts[0].upper().strip()  # Ensure consistent case and remove extra whitespace
                try:
                    flight = float(parts[1])
                    dwell = float(parts[2])
                    strokes.append((key, round(flight, 2), round(dwell, 2)))
                except ValueError:
                    print(f"Skipping line in {path} due to invalid number format: {line.strip()}")
                    continue
    except Exception as e:
        print(f"Error processing file {path}: {e}")
        return []

    return strokes


def extract_features(strokes, target_keys=None):
    """
    Extract average dwell times for specified target keys

    Args:
        strokes: List of (key, flight, dwell) tuples
        target_keys: List of keys to calculate average dwell times for
                     If None, defaults to ['X', 'Y', 'Z']

    Returns:
        Dictionary mapping keys to their average dwell times
    """
    if target_keys is None:
        target_keys = ['X', 'Y', 'Z']

    features = {}
    for key in target_keys:
        # Handle 'SPACE' key which might be represented as space character or 'SPACE'
        if key == 'SPACE':
            key_strokes = [stroke[2] for stroke in strokes if stroke[0] == key or stroke[0] == ' ']
        else:
            key_strokes = [stroke[2] for stroke in strokes if stroke[0] == key]

        if key_strokes:
            features[key] = sum(key_strokes) / len(key_strokes)
        else:
            features[key] = 0  # No instances of this key

    return features


def calculate_distance(features1, features2, target_keys=None, metric="manhattan"):
    """
    Calculate distance between two feature vectors using various metrics

    Args:
        features1, features2: Feature dictionaries to compare
        target_keys: Keys to consider for distance calculation
        metric: Distance metric to use (manhattan, euclidean, chebyshev, braycurtis,
                canberra, minkowski, hamming)

    Returns:
        Distance between the feature vectors
    """
    if target_keys is None:
        target_keys = ['X', 'Y', 'Z']

    # Extract values as vectors, replacing missing values with 0
    vector1 = [features1.get(key, 0) for key in target_keys]
    vector2 = [features2.get(key, 0) for key in target_keys]

    # Convert to numpy arrays for easier calculation
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    if metric == "manhattan":
        # Manhattan (L1) distance: sum of absolute differences
        return np.sum(np.abs(v1 - v2))

    elif metric == "euclidean":
        # Euclidean (L2) distance: square root of sum of squared differences
        return np.sqrt(np.sum((v1 - v2) ** 2))

    elif metric == "chebyshev":
        # Chebyshev (L∞) distance: maximum absolute difference
        return np.max(np.abs(v1 - v2))

    elif metric == "braycurtis":
        # Bray-Curtis distance: sum of absolute differences divided by sum of absolute values
        # Avoid division by zero
        denominator = np.sum(np.abs(v1) + np.abs(v2))
        if denominator == 0:
            return 0
        return np.sum(np.abs(v1 - v2)) / denominator

    elif metric == "canberra":
        # Canberra distance: sum of (|x_i - y_i| / (|x_i| + |y_i|))
        # Handle division by zero
        result = 0
        for i in range(len(v1)):
            denominator = np.abs(v1[i]) + np.abs(v2[i])
            if denominator > 0:  # Avoid division by zero
                result += np.abs(v1[i] - v2[i]) / denominator
        return result

    elif metric == "minkowski":
        # Minkowski distance with p=3
        p = 3
        return np.power(np.sum(np.power(np.abs(v1 - v2), p)), 1 / p)

    elif metric == "hamming":
        # Hamming distance: percentage of components that differ
        # For continuous data, we consider values "different" if they differ by more than a threshold
        threshold = 0.001
        return np.mean(np.abs(v1 - v2) > threshold)

    else:
        # Default to Manhattan distance
        return np.sum(np.abs(v1 - v2))


def knn_classifier(sample_features, training_features, training_labels, k=3, target_keys=None,
                   distance_metric="manhattan"):
    """
    Classify sample using k-NN

    Args:
        sample_features: Features of the sample to classify
        training_features: List of feature dictionaries for training samples
        training_labels: List of labels for training samples
        k: Number of nearest neighbors to consider
        target_keys: Keys to consider for features
        distance_metric: Distance metric to use for comparing samples

    Returns:
        Predicted label for the sample
    """
    if target_keys is None:
        target_keys = ['X', 'Y', 'Z']

    # Calculate distances to all training samples
    distances = []
    for i, features in enumerate(training_features):
        dist = calculate_distance(sample_features, features, target_keys, distance_metric)
        distances.append((dist, training_labels[i]))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Get k nearest neighbors
    k_nearest = distances[:k]

    # Count labels
    label_counts = Counter([label for _, label in k_nearest])

    # Find the most common label
    most_common = label_counts.most_common()

    # If there are ties, choose the closest one
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        # Find the closest sample with the tied labels
        for dist, label in k_nearest:
            if label in [l for l, count in most_common if count == most_common[0][1]]:
                return label

    return most_common[0][0]


def evaluate_classifier(samples_dir, target_keys=None, k=3, distance_metric="manhattan"):
    """
    Evaluate k-NN classifier using leave-one-out cross-validation

    Args:
        samples_dir: Directory containing sample files
        target_keys: Keys to use for features
        k: Number of nearest neighbors
        distance_metric: Distance metric to use

    Returns:
        Accuracy of the classifier
    """
    if target_keys is None:
        target_keys = ['X', 'Y', 'Z']

    # Get all sample files
    try:
        sample_files = [f for f in os.listdir(samples_dir) if
                        os.path.isfile(os.path.join(samples_dir, f)) and f.endswith(('.txt', '.csv'))]

        if not sample_files:
            print(f"No text or CSV files found in {samples_dir}")
            return 0
    except Exception as e:
        print(f"Error accessing directory {samples_dir}: {e}")
        return 0

    # Extract user labels and parse samples
    users = {}
    all_samples = []
    all_labels = []

    for file in sample_files:
        try:
            # Extract user label from filename
            # Assuming filename format like "#01_2.txt" where "#01" is the user
            user_label = file.split('_')[0]

            # Parse sample
            path = os.path.join(samples_dir, file)
            strokes = parse_sample(path)

            if strokes:
                # Extract features
                features = extract_features(strokes, target_keys)

                # Store sample and label
                all_samples.append(features)
                all_labels.append(user_label)

                # Group by user
                if user_label not in users:
                    users[user_label] = []
                users[user_label].append(features)
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue

    if not all_samples:
        print("No valid samples could be processed")
        return 0

    if len(users) < 2:
        print("Need at least two different users for classification")
        return 0

    # Make sure k is not larger than the number of samples
    effective_k = min(k, len(all_samples) - 1)
    if effective_k < k:
        print(f"Warning: Reducing k from {k} to {effective_k} due to limited number of samples")

    # Evaluate using leave-one-out cross-validation
    correct = 0
    total = 0

    for i in range(len(all_samples)):
        # Create training set without current sample
        training_features = all_samples[:i] + all_samples[i + 1:]
        training_labels = all_labels[:i] + all_labels[i + 1:]

        # Classify current sample
        predicted_label = knn_classifier(all_samples[i], training_features, training_labels,
                                         effective_k, target_keys, distance_metric)

        # Check if prediction is correct
        if predicted_label == all_labels[i]:
            correct += 1
        total += 1

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0

    return accuracy













def get_thresh():  # wpisywanie progu dla binaryzacji

    global threshold
    kill_UI()

    rat = tk.StringVar()
    label4 = ttk.Label(window, text="Prog:", background="#e7e7e7")
    label4.place(x=50, y=50)
    entry4 = ttk.Entry(window, textvariable=rat, width=30)
    entry4.place(x=250, y=50)
    feedback4 = ttk.Label(window, text="", background="#e7e7e7")
    feedback4.place(x=450, y=50)

    info_button = ttk.Button(
        window,
        text="Pokaz Info",
        command=lambda: HELP_window(
            "zmienia czułość binaryzacji. \n\n\n "
            "Niższe wartości spowodują, że obrazy będą bardziej białe, a wyższe - ciemniejsze. \n\n\n")
    )
    info_button.place(x=250, y=400)

    submit_button = ttk.Button(
        window,
        text="Dalej",
        state="disabled",
        command=lambda: (kill_UI(), wybor_dzialan())
    )
    submit_button.place(x=250, y=150)

    rat.trace_add("write", lambda *args: validate_thresh(rat, feedback4, submit_button))

def validate_thresh(rat, feedback4, submit_button):  # walidacja tego progu
    global threshold
    val = rat.get().strip()

    if val.isdigit() and (0 <= int(val) <= 255):
        threshold = int(val)
        feedback4.config(text="Prawidlowe", foreground="green")
        submit_button.config(state="normal")
    else:
        feedback4.config(text="Nie Prawidlowe \n\n\n (wartosci tylko od 0 do 255)", foreground="red")
        submit_button.config(state="disabled")

def wybor_dzialan():
    kill_UI()
    global pytaniea, bina_button, hisa_button, fila_button
    pytaniea = ttk.Label(window, text="co robimy?", background="#e7e7e7")
    pytaniea.place(x=250, y=150)

    bina_button = ttk.Button(window, text="binaryzacje", width=20, command=lambda: (kill_UI(), binaryz()))
    bina_button.place(x=100, y=200)
    hisa_button = ttk.Button(window, text="histogramy", width=20, command=lambda: (kill_UI(), histy()))
    hisa_button.place(x=300, y=200)
    fila_button = ttk.Button(window, text="filtry", width=20, command=lambda: (kill_UI(), filtry()))
    fila_button.place(x=500, y=200)
    rysow_button = ttk.Button(window, text="rysowanie", width=20, command=lambda: (kill_UI(), enable_drawing()))
    rysow_button.place(x=300, y=300)


def enable_drawing():
    global btn_toggle_flood, drawing_canvas, r_weight, g_weight, b_weight, drawing_image, original_drawing_image, drawing_draw

    kill_UI()

    MAX_WIDTH, MAX_HEIGHT = 900, 500  # Define max size for the image

    drawing_canvas = tk.Canvas(window, width=MAX_WIDTH, height=MAX_HEIGHT)
    drawing_canvas.place(x=50, y=1)  # Position the canvas

    enable_drawing_actions()  # Ensure drawing actions are bound IMMEDIATELY

    # If an image is already loaded, prepare it for drawing right away
    if image is not None:
        img_width, img_height = image.size
        if img_width > MAX_WIDTH or img_height > MAX_HEIGHT:
            scale = min(MAX_WIDTH / img_width, MAX_HEIGHT / img_height)  # Scale factor
            new_size = (int(img_width * scale), int(img_height * scale))
            drawing_image = image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            drawing_image = image.copy()

        original_drawing_image = drawing_image.copy()
        drawing_draw = ImageDraw.Draw(drawing_image)  # Enable drawing

        drawing_canvas.config(width=drawing_image.width, height=drawing_image.height)  # Resize canvas
        update_drawing_canvas()  # Show the image immediately

    # Create buttons
    btn_toggle_flood = ttk.Button(window, text='tryb globalny', command=toggle_global_flood)
    btn_toggle_flood.place(x=100, y=600)

    btn_toggle_expand = ttk.Button(window, text='Rozszerzanie', command=expand_selection)
    btn_toggle_expand.place(x=300, y=600)

    btn_toggle_contract = ttk.Button(window, text='zmniejszanie', command=contract_selection)
    btn_toggle_contract.place(x=500, y=600)

    # Color selection


    r_weight = tk.StringVar()
    g_weight = tk.StringVar()
    b_weight = tk.StringVar()

    ttk.Label(window, text="Czerwony").place(x=300, y=640)
    ttk.Label(window, text="Zielony").place(x=400, y=640)
    ttk.Label(window, text="Niebieski").place(x=500, y=640)

    entry_red = ttk.Entry(window, textvariable=r_weight, width=10, background="white")
    entry_red.place(x=300, y=670)

    entry_green = ttk.Entry(window, textvariable=g_weight, width=10, background="white")
    entry_green.place(x=400, y=670)

    entry_blue = ttk.Entry(window, textvariable=b_weight, width=10, background="white")
    entry_blue.place(x=500, y=670)

    update_color_btn = ttk.Button(window, text="ustaw kolor", command=update_color)
    update_color_btn.place(x=650, y=670)

    label_tolerance = ttk.Label(window, text="tolerancja:", background="#e7e7e7")
    label_tolerance.place(x=800, y=640)
    entry_tolerance = ttk.Entry(window, textvariable=tolerance, width=5)
    entry_tolerance.place(x=800, y=670)
    feedback7 = ttk.Label(window, text="", background="#e7e7e7")
    feedback7.place(x=800, y=700)
    tolerance.trace_add("write", lambda *args: validate_tolerance(tolerance, feedback7))

    # Back button
    inne_op = ttk.Button(window, text="inne operacje", width=20, command=wybor_dzialan)
    inne_op.place(x=500, y=750)

    save_button = ttk.Button(window, text="zapisz", width=20, command=save_drawing)
    save_button.place(x=100, y=750)


def save_drawing():
    drawn_image_path = os.path.join(timestamped_folder_path, f"rysowane_{datetime.now().strftime('%H-%M-%S')}_{image_name}").replace('\\', '/')
    os.makedirs(os.path.dirname(drawn_image_path), exist_ok=True)
    drawing_image.save(drawn_image_path)

def validate_tolerance(tol, feedback7):  # walidacja tego progu

    val = tol.get().strip()

    if val.isdigit() and (0 <= int(val) <= 255):

        feedback7.config(text="Prawidlowe", foreground="green")
        re_enable_flood()
    else:
        feedback7.config(text="Nie Prawidlowe \n\n\n (wartosci tylko od 0 do 255)", foreground="red")
        disable_flood()


def enable_drawing_actions():
    drawing_canvas.bind("<B1-Motion>", paint)  # Left-click drag to draw
    drawing_canvas.bind("<Button-3>", flood_fill)  # Right-click to flood fill
def disable_flood():
    drawing_canvas.unbind("<Button-3>")  # Unbind left-click
def re_enable_flood():
    drawing_canvas.bind("<Button-3>", flood_fill)
def toggle_global_flood():
    global global_flood_enabled, btn_toggle_flood

    # Toggle the state of the global flood mode
    global_flood_enabled = not global_flood_enabled

    # Change the button text based on the current state
    if global_flood_enabled:
        btn_toggle_flood.config(text="tryb globalny Wł")
    else:
        btn_toggle_flood.config(text="tryb globalny Wył")


def paint(event):
    global drawing_image, drawing_draw
    if drawing_image is None:
        return

    x, y = event.x, event.y
    drawing_draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=tuple(selected_color), outline=tuple(selected_color))
    update_drawing_canvas()


def update_drawing_canvas():
    global drawing_tk_image

    # Clear previous image before updating
    drawing_canvas.delete("all")

    # Resize canvas to match the image exactly
    drawing_canvas.config(width=drawing_image.width, height=drawing_image.height)

    drawing_tk_image = ImageTk.PhotoImage(drawing_image)
    drawing_canvas.create_image(0, 0, anchor=tk.NW, image=drawing_tk_image)

def flood_fill(event):
    global drawing_image, drawing_draw,tolerance
    if drawing_image is None:
        return
    tolerance3=int(tolerance.get())
    x, y = event.x, event.y
    target_color = np.array(drawing_image.getpixel((x, y)))  # Color at the clicked point
    new_color = np.array(selected_color)  # Color to fill
    pixels = np.array(drawing_image)

    if global_flood_enabled:
        # For global flood mode, fill the whole image where the color matches within tolerance
        mask = np.all(abs(pixels - target_color) <= tolerance3, axis=-1)
    else:
        # For regular flood fill, we use a connected component algorithm
        mask = np.all(abs(pixels - target_color) <= tolerance3, axis=-1)
        connected_pixels = np.zeros_like(mask, dtype=bool)
        stack = [(x, y)]

        while stack:
            cx, cy = stack.pop()
            if 0 <= cx < pixels.shape[1] and 0 <= cy < pixels.shape[0] and not connected_pixels[cy, cx] and mask[cy, cx]:
                connected_pixels[cy, cx] = True
                # Add the neighboring pixels to the stack
                stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])

        mask = connected_pixels  # Update the mask to reflect the connected region

    # Update the image pixels based on the mode (RGB or RGBA)
    if pixels.shape[-1] == 4:  # RGBA image (with alpha channel)
        # Set the RGB values and keep the alpha channel intact
        pixels[mask] = np.concatenate([new_color[:3], pixels[mask][:, -1][:, None]], axis=-1)
    else:  # RGB image (no alpha channel)
        pixels[mask] = new_color[:3]

    # Update the image and drawing object
    drawing_image = Image.fromarray(pixels)
    drawing_draw = ImageDraw.Draw(drawing_image)  # Re-enable drawing after filling

    # Refresh the canvas to reflect the changes
    update_drawing_canvas()


def expand_selection(tolerance_value=None):
    global drawing_image, drawing_draw
    if drawing_image is None:
        return

    # Use provided tolerance or get from UI
    if tolerance_value is None:
        tolerance_value = int(tolerance.get())

    pixels = np.array(drawing_image)
    target_color = np.array(selected_color)[:3]  # Get RGB of selected color

    # Create mask where pixels are within tolerance of selected color
    color_diffs = np.abs(pixels[:, :, :3] - target_color)
    mask = np.all(color_diffs <= tolerance_value, axis=-1)

    # Apply dilation to the mask
    expanded_mask = binary_dilation(mask)

    # Update only the newly expanded pixels (those in expanded_mask but not in original mask)
    new_pixels = expanded_mask & ~mask
    pixels[new_pixels, :3] = target_color

    drawing_image = Image.fromarray(pixels)
    drawing_draw = ImageDraw.Draw(drawing_image)
    update_drawing_canvas()


def contract_selection(tolerance_value=None):
    global drawing_image, drawing_draw, original_drawing_image
    if drawing_image is None or original_drawing_image is None:
        return

    # Use provided tolerance or get from UI
    if tolerance_value is None:
        tolerance_value = int(tolerance.get())

    pixels = np.array(drawing_image)
    original_pixels = np.array(original_drawing_image)
    target_color = np.array(selected_color)[:3]  # Get RGB of selected color

    # Create mask where pixels are within tolerance of selected color
    color_diffs = np.abs(pixels[:, :, :3] - target_color)
    mask = np.all(color_diffs <= tolerance_value, axis=-1)

    # Apply erosion to the mask
    contracted_mask = binary_erosion(mask)

    # Identify pixels that were in the original mask but not in the contracted mask
    removed_pixels = mask & ~contracted_mask

    # Restore original colors for removed pixels
    pixels[removed_pixels, :3] = original_pixels[removed_pixels, :3]

    drawing_image = Image.fromarray(pixels)
    drawing_draw = ImageDraw.Draw(drawing_image)
    update_drawing_canvas()


def update_color():
    global selected_color
    try:
        r = int(r_weight.get() or 0)
        g = int(g_weight.get() or 0)
        b = int(b_weight.get() or 0)
        selected_color = [r, g, b, 255]
    except ValueError as e:
        showinfo(title="Blad", message=f"{e}")











def filtry():

    fil = tk.StringVar()
    pix = tk.StringVar()

    label1 = ttk.Label(window, text="Pliki zostaną zapisany w folderze: {}".format(timestamped_folder_path),
                       background="#e7e7e7")
    label1.place(x=255, y=20)

    inne_op = ttk.Button(window, text="inne operacje", width=20, command=wybor_dzialan)
    inne_op.place(x=400, y=600)

    sobel_button = ttk.Button(window, text="sobel method", width=20, command=lambda: sobel())
    sobel_button.place(x=200, y=100)




    median_button = ttk.Button(window, text="median method", width=20, state='disabled', command=lambda: median_meth(fil))
    median_button.place(x=200, y=200)

    kuwah_button = ttk.Button(window, text="kuwahara method", width=20, state='disabled', command=lambda: apply_kuwahara(fil))
    kuwah_button.place(x=200, y=300)

    pixel_button = ttk.Button(window, text="pixelezacja method", width=20, state='disabled', command=lambda:  pixelate(pix))
    pixel_button.place(x=200, y=400)

    mirgb_button = ttk.Button(window, text="MinRGB method", width=20, command=lambda: min_rgb())
    mirgb_button.place(x=200, y=500)

    predator_button = ttk.Button(window, text="predator method",state='disabled', width=20, command=lambda: predator(pix))
    predator_button.place(x=500, y=300)


    label_median = ttk.Label(window, text="moc filtru:", background="#e7e7e7")
    label_median.place(x=400, y=225)
    entry_median = ttk.Entry(window, textvariable=fil, width=5)
    entry_median.place(x=400, y=250)
    feedback6 = ttk.Label(window, text="", background="#e7e7e7")
    feedback6.place(x=450, y=275)
    fil.trace_add("write", lambda *args: validate_fil(fil, feedback6, median_button,kuwah_button))

    label_pix = ttk.Label(window, text="moc pixelizacji:", background="#e7e7e7")
    label_pix.place(x=400, y=375)
    entry_pix = ttk.Entry(window, textvariable=pix, width=5)
    entry_pix.place(x=400, y=400)
    feedback9 = ttk.Label(window, text="", background="#e7e7e7")
    feedback9.place(x=450, y=425)
    pix.trace_add("write", lambda *args: validate_pix(pix, feedback9, pixel_button,predator_button))
    close_button = ttk.Button(window, text="Zamknij okienko", width=20, command=window.destroy)
    close_button.place(x=50, y=700)

    restart_button = ttk.Button(window, text="zacznij od nowa", width=20, command=lambda: select_file())
    restart_button.place(x=600, y=700)




def validate_pix(pix, feedback9, pixel_button,predator_button):  # walidacja tego progu
    global threshold
    val = pix.get().strip()

    if val.isdigit() and (0 <= int(val) <= 50):
        threshold = int(val)
        feedback9.config(text="Prawidlowe", foreground="green")
        pixel_button.config(state="normal")
        predator_button.config(state="normal")

    else:
        feedback9.config(text="Nie Prawidlowe \n\n\n (wartosci tylko od 0 do 50)", foreground="red")
        pixel_button.config(state="disabled")
        predator_button.config(state="disabled")



def validate_fil(fil, feedback6, median_button,kuwah_button):  # walidacja tego progu
    global threshold
    val = fil.get().strip()

    if val.isdigit() and (0 <= int(val) <= 20):
        threshold = int(val)
        feedback6.config(text="Prawidlowe", foreground="green")
        median_button.config(state="normal")
        kuwah_button.config(state="normal")
    else:
        feedback6.config(text="Nie Prawidlowe \n\n\n (wartosci tylko od 0 do 20)", foreground="red")
        median_button.config(state="disabled")
        kuwah_button.config(state="disabled")


def sobel():
    global full_path, poin
    input_image = plt.imread(full_path)

    # Extracting RGB components
    r_img, g_img, b_img = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]

    # Convert to grayscale with gamma correction (vectorized)
    gamma = 1.400
    r_const, g_const, b_const = 0.2126, 0.7152, 0.0722
    grayscale_image = (r_const * r_img ** gamma +
                       g_const * g_img ** gamma +
                       b_const * b_img ** gamma)

    # Define Sobel kernels
    Gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

    # Use convolution for much faster edge detection
    gx = convolve2d(grayscale_image, Gx, mode='same', boundary='symm')
    gy = convolve2d(grayscale_image, Gy, mode='same', boundary='symm')

    # Compute magnitude of gradients
    sobel_filtered_image = np.sqrt(gx ** 2 + gy ** 2)

    sobel_normalized = ((sobel_filtered_image - sobel_filtered_image.min()) /
                        (sobel_filtered_image.max() - sobel_filtered_image.min()) * 255).astype(np.uint8)

    # Prepare save paths
    save_path = os.path.join(timestamped_folder_path, f"Sobel_{image_name}").replace('\\', '/')
    filtered_image_path = os.path.join(timestamped_folder_path, f"Sobel_filtered_{image_name}").replace('\\', '/')

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the Sobel filtered image
    sobel_image=Image.fromarray(sobel_normalized)
    sobel_image.save(filtered_image_path)

    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(input_image)
    plt.axis('off')

    plt.subplot(122)
    plt.title('Sobel Edge Detection')
    plt.imshow(sobel_filtered_image, cmap=plt.get_cmap('gray'))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    if poin == "1":
        plt.show()
        sobel_image.show()

    plt.close()  # Close the plot to free up memory


    sobel_edges=Image.fromarray(sobel_normalized)
    return sobel_edges


def sobel_edge_detection(pro_image=image):
    # Convert image to numpy array
    img_array = np.array(pro_image)

    # Separate RGB channels
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2]

    # Sobel kernels
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Function to apply Sobel to a single channel
    def apply_sobel(channel):
        # Compute x and y gradients using convolve2d
        x_grad = np.abs(convolve2d(channel, Gx, mode='same', boundary='symm'))
        y_grad = np.abs(convolve2d(channel, Gy, mode='same', boundary='symm'))

        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(x_grad ** 2 + y_grad ** 2)

        # Normalize
        return ((gradient_magnitude - gradient_magnitude.min()) /
                (gradient_magnitude.max() - gradient_magnitude.min()) * 255).astype(np.uint8)

    # Apply Sobel to each channel
    r_edges = apply_sobel(r_channel)
    g_edges = apply_sobel(g_channel)
    b_edges = apply_sobel(b_channel)

    # Recombine channels
    edges_array = np.stack([r_edges, g_edges, b_edges], axis=-1)

    egded_image=Image.fromarray(edges_array)
    return egded_image

def median_meth(filter_size=5):
    global full_path
    # Open image and convert to grayscale
    img = Image.open(full_path).convert('RGB')
    data = np.array(img)

    filter_size=int(filter_size.get())
    # Prepare output array
    rows, cols, channels = data.shape
    filtered_data = np.zeros_like(data, dtype=data.dtype)

    # Padding calculation
    pad = filter_size // 2

    # Extend the image with reflect padding (better than zero padding)
    padded_data = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    # Apply median filter for each pixel
    for i in range(rows):
        for j in range(cols):
            # Extract local window
            window = padded_data[i:i + filter_size, j:j + filter_size, :]

            # Flatten window and find median across all color channels
            filtered_data[i, j] = np.median(window.reshape(-1, channels), axis=0)

    # Visualization
    save_path = os.path.join(timestamped_folder_path, f"color_median_{image_name}").replace('\\', '/')
    filtered_image_path = os.path.join(timestamped_folder_path, f"color_median_filtered_{filter_size}_{image_name}").replace('\\','/')

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save filtered image
    median_image = Image.fromarray(filtered_data)
    median_image.save(filtered_image_path)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.title('Original Image')
    plt.imshow(data)
    plt.axis('off')

    plt.subplot(122)
    plt.title(f'Median Filtered Image ({filter_size}x{filter_size})')
    plt.imshow(filtered_data)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    if poin == "1":
        plt.show()
        median_image.show()
    plt.close()

    return filtered_data



def apply_kuwahara(window_size=5):
    global image
    image_array = np.array(image)

    # If it's a color image, apply to each channel
    if len(image_array.shape) == 3:  # Color image
        filtered_channels = [kuwahara_filter(image_array[:, :, i], window_size) for i in range(image_array.shape[2])]
        filtered_image = np.stack(filtered_channels, axis=-1)
    else:  # Grayscale
        filtered_image = kuwahara_filter(image_array, window_size)

    filtered_image = Image.fromarray(filtered_image)
    filtered_image.show()


    filtered_image_path = os.path.join(timestamped_folder_path, f"kuwah_{int(window_size.get())}_{image_name}").replace('\\','/')

    filtered_image.save(filtered_image_path)



    return filtered_image

def kuwahara_filter_region(values):
    # Extract the 4 regions from the values (which correspond to the 4 quadrants)
    half = len(values) // 4
    region1 = values[:half]  # North-west
    region2 = values[half:2 * half]  # North-east
    region3 = values[2 * half:3 * half]  # South-west
    region4 = values[3 * half:]  # South-east

    # Compute the mean and stddev for each region
    means = [np.mean(region1), np.mean(region2), np.mean(region3), np.mean(region4)]
    stddevs = [np.std(region1), np.std(region2), np.std(region3), np.std(region4)]

    # Return the mean of the region with the smallest stddev
    return means[np.argmin(stddevs)]

def kuwahara_filter(image, winsize=5):

    winsize=int(winsize.get())
    image = image.astype(np.float64)

    half = (winsize - 1) // 2

    # Create a zero-padded image for boundary handling
    padded_image = np.pad(image, ((half, half), (half, half)), mode='reflect')

    # Define the neighborhood size (4 quadrants per pixel)
    neighborhood = winsize * winsize

    # Apply the sliding window function to each pixel using generic_filter
    filtered_image = generic_filter(padded_image, kuwahara_filter_region, size=(winsize, winsize), mode='reflect')

    # Convert back to uint8
    return filtered_image.astype(np.uint8)

def pixelate(pixelate_lvl,provided_image=image):

    org_size = image.size
    if type(pixelate_lvl) != int:
        pixelate_lvl=int(pixelate_lvl.get())
    # scale it down
    image_smol = image.resize(
        size=(org_size[0] // pixelate_lvl, org_size[1] // pixelate_lvl),
        resample=0)
    # and scale it up to get pixelate effect
    image_big = image_smol.resize(org_size, resample=0)

    new_file_path = os.path.join(timestamped_folder_path, f"Pixelated_{pixelate_lvl}__{image_name}").replace('\\', '/')
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    image_big.save(new_file_path)
    if poin == "1":
        image_big.show()
    return image_big


def min_rgb(provided_image=image):
    if poin=="1":
        global image
        provided_image=image

    # Convert to numpy array
    img_array = np.array(provided_image)

    # Ensure image is in RGB format
    if len(img_array.shape) == 2:  # Grayscale image
        raise ValueError("Provided image is grayscale. Expected RGB image.")

    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {img_array.shape}. Expected (H, W, 3).")

    # Find the index of the minimal channel for each pixel
    min_channel_indices = np.argmin(img_array, axis=2)

    # Create output array
    out_array = np.zeros_like(img_array)

    # Set the minimal channel to its original value, others to 0
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            min_channel = min_channel_indices[i, j]
            out_array[i, j, min_channel] = img_array[i, j, min_channel]

    # Convert back to PIL Image
    mirgb_image = Image.fromarray(out_array)

    new_file_path = os.path.join(timestamped_folder_path, f"MinRGB__{image_name}").replace('\\', '/')
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    mirgb_image.save(new_file_path)


    if poin == "1":
        mirgb_image.show()
    return mirgb_image

def predator(pixelate_level=10):
    global poin
    poin="0"
    pixelated=pixelate(pixelate_level)


    mined=min_rgb(pixelated)
    predate_image=sobel_edge_detection(mined)
    poin="1" #keep poin
    predate_image.show()
    new_file_path = os.path.join(timestamped_folder_path, f"PREDATOR!!!_{image_name}").replace('\\', '/')
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    predate_image.save(new_file_path)
    return predate_image





def histy():
    kill_UI()
    left_col = 100
    right_col = 500
    mid_col = 300

    stretching_checkbox = ttk.Checkbutton(window, text="Rozciagniecie histogramu", variable=stretch)
    stretching_checkbox.place(x=left_col, y=100)

    equalisation_checkbox = ttk.Checkbutton(window, text="wyrownanie histogramu", variable=equalise)
    equalisation_checkbox.place(x=mid_col, y=100)

    otsu_checkbox = ttk.Checkbutton(window, text="metoda Otsu", variable=otsu)
    otsu_checkbox.place(x=right_col, y=100)


    label1 = ttk.Label(window, text="Pliki zostaną zapisany w folderze: {}".format(timestamped_folder_path),
                       background="#e7e7e7")
    label1.place(x=mid_col, y=20)

    show_button = ttk.Button(window, text="pokaz obraz", width=20, command=lambda: image.show())
    show_button.place(x=right_col, y=50)

    hist_all_button = ttk.Button(window, text="sredni histogram", width=20, command=lambda: create_histogram())
    hist_all_button.place(x=right_col, y=200)

    hist_red_button = ttk.Button(window, text="czerwony histogram ", width=20, command=lambda: create_histogram('r'))
    hist_red_button.place(x=right_col, y=250)

    hist_green_button = ttk.Button(window, text="zielony histogram", width=20, command=lambda: create_histogram('g'))
    hist_green_button.place(x=right_col, y=300)

    hist_blue_button = ttk.Button(window, text="niebieski histogram", width=20, command=lambda: create_histogram('b'))
    hist_blue_button.place(x=right_col, y=350)

    hist_gray_button = ttk.Button(window, text="szary histogram", width=20, command=lambda: create_histogram('average'))
    hist_gray_button.place(x=right_col, y=400)

    hist_full_button = ttk.Button(window, text="wszystkie histogramy", width=20, command=lambda: all_hists())
    hist_full_button.place(x=right_col, y=550)

    close_button = ttk.Button(window, text="Zamknij okienko", width=20, command=window.destroy)
    close_button.place(x=left_col, y=700)

    restart_button = ttk.Button(window, text="zacznij od nowa", width=20, command=lambda: select_file())
    restart_button.place(x=right_col, y=700)

    info_button = ttk.Button(
        window,
        text="Pokaz Info",
        width=20,
        command=lambda: HELP_window(
            "Pokaz obraz pokazuje wybrany obraz \n\n\n "
            "Histogram  sredni/czerwony/zielony/niebieski/szary robi histogram wybranego typu, wynikujacy plik zapisuje\n\n\n"
            "wszystkie binaryzacje/histogramy wykonuje binaryzacje/histogramy kazdego typu, wynikujace pliki zapisuje\n\n\n"
        )
    )
    info_button.place(x=mid_col, y=650)

    inne_op = ttk.Button(window, text="inne operacje", width=20, command=wybor_dzialan)
    inne_op.place(x=left_col, y=600)

#########################################################################################################
def equalise_single_channel(image):
    # Flatten the image into a 1D array
    flat = image.flatten()
    hist, bins = np.histogram(flat, bins=256, range=[0, 256])

    # Normalize the histogram to get the probability distribution function (PDF)
    pdf = hist / flat.size

    # Compute the cumulative distribution function (CDF)
    cdf = np.cumsum(pdf)

    # Normalize the CDF
    cdf_normalized = (cdf * 255).astype(np.uint8)

    # Map the original image pixels to equalized values
    equalized = cdf_normalized[flat]

    # Reshape back to the original image shape
    return equalized.reshape(image.shape)


def equalise_histogram(image):
    # Check if image is grayscale or RGB
    if len(image.shape) == 2:
        return equalise_single_channel(image)
    else:
        # Apply to each channel separately
        result = np.zeros_like(image)
        for i in range(3):
            result[:, :, i] = equalise_single_channel(image[:, :, i])
        return result


###########################################################################################################

def create_histogram(channel='all'):
    global image, image_location, image_name, state

    figure, axis = plt.subplots(figsize=(10, 6))

    if channel == 'all':

        rgb_image = image.convert('RGB')
        rgb_array = np.array(rgb_image)

        if equalise.get():
            rgb_array = equalise_histogram(rgb_array)

        if stretch.get():
            stretched_array = np.zeros_like(rgb_array)
            for i in range(3):  # Process each RGB channel
                channel_data = rgb_array[:, :,
                               i]  # 1sze : wybiera wszystkie wiersze, 2gie- kolumne, 3cie, pokazuje R,G, czy B
                vmin = np.min(channel_data)
                vmax = np.max(channel_data)
                imax = 255
                #  LUT(i) = (imax/(vmax-vmin))*(i-vmin)
                stretched_array[:, :, i] = (imax / (vmax - vmin)) * (channel_data - vmin)
            rgb_array = stretched_array

        colors = ['r', 'g', 'b']
        channel_names = ('Red', 'Green', 'Blue')

        for i, color in enumerate(colors):
            channel_data = rgb_array[:, :, i].flatten()
            axis.hist(channel_data, bins=256, range=(0, 255),
                      color=color, alpha=0.7, label=channel_names[i])

        axis.set_title('RGB Histogram')
        axis.legend()

        if equalise.get() or stretch.get():
            processed_image = Image.fromarray(rgb_array.astype('uint8'), 'RGB')
        else:
            processed_image = image.convert('RGB')


    elif channel in ['r', 'g', 'b']:
        rgb_image = image.convert('RGB')
        rgb_array = np.array(rgb_image)

        # get specific channel
        channel_index = {'r': 0, 'g': 1, 'b': 2}[channel]
        channel_data = rgb_array[:, :, channel_index].copy()
        flat_channel_data = channel_data.flatten()

        if equalise.get():
            processed_flat_data = equalise_single_channel(flat_channel_data)
            channel_data = processed_flat_data.reshape(channel_data.shape)

        if stretch.get():
            vmin = np.min(channel_data)
            vmax = np.max(channel_data)
            imax = 255
            # Apply the LUT formula: LUT(i) = (imax/(vmax-vmin))*(i-vmin)
            channel_data = (imax / (vmax - vmin)) * (channel_data - vmin)

        color_name = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}[channel]
        axis.hist(channel_data.flatten(), bins=256, range=(0, 255),
                  color=channel, alpha=0.7, label=color_name)

        axis.set_title(f'Histogram kanalu: {color_name}')
        axis.legend()

        processed_image = Image.fromarray(channel_data.astype('uint8'), 'L')

    elif channel == 'average':
        gray_image = image.convert('L')
        gray_array = np.array(gray_image)
        flat_gray_array = gray_array.flatten()

        if equalise.get():
            processed_flat_data = equalise_single_channel(flat_gray_array)
            gray_array = processed_flat_data.reshape(gray_array.shape)

        if stretch.get():
            vmin = np.min(gray_array)
            vmax = np.max(gray_array)
            imax = 255
            # Apply the LUT formula: LUT(i) = (imax/(vmax-vmin))*(i-vmin)
            gray_array = (imax / (vmax - vmin)) * (gray_array - vmin)

        axis.hist(gray_array.flatten(), bins=256, range=(0, 255),
                  color='gray', alpha=0.7, label='Grayscale')

        axis.set_title('histogram szary')
        axis.legend()

        processed_image = Image.fromarray(gray_array.astype('uint8'), 'L')

    axis.set_xlabel('Wartosc pikseli')
    axis.set_ylabel('czestosc')
    axis.set_xlim([0, 255])
    axis.grid(True, alpha=0.3)

    is_stretched = "Rozciagniete" * int(stretch.get())
    is_equalised = "Wyrownane" * int(equalise.get())
    plt.tight_layout()
    save_path = os.path.join(timestamped_folder_path,
                             f"histogram_{is_stretched}_{is_equalised}_{channel}_{image_name}").replace('\\', '/')

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)  # Always save first
    if state == "o":
        plt.show()

    if otsu.get() and processed_image is not None:
        # Convert to grayscale if not already
        gray_img = processed_image.convert('L')
        gray_array = np.array(gray_img)

        # Calculate histogram
        hist, bins = np.histogram(gray_array.flatten(), 256, [0, 256])

        # Calculate cumulative sums
        cum_sum = hist.cumsum()
        cum_mean = (hist * np.arange(256)).cumsum()
        total_pixels = cum_sum[-1]

        # Initialize variables
        max_variance = 0
        t_hold = 0

        # Find threshold with maximum between-class variance
        for t in range(1, 256):
            # Weights
            w0 = cum_sum[t - 1]
            w1 = total_pixels - w0

            # Skip if any class is empty
            if w0 == 0 or w1 == 0:
                continue

            # Class means
            mean0 = cum_mean[t - 1] / w0
            mean1 = (cum_mean[-1] - cum_mean[t - 1]) / w1

            # Calculate between-class variance
            variance = w0 * w1 * ((mean0 - mean1) ** 2)

            # Update threshold if variance is higher
            if variance > max_variance:
                max_variance = variance
                t_hold = t

        # Apply threshold to create binary image
        binary_img = gray_array > t_hold
        binary_array = binary_img.astype('uint8') * 255
        binary_image = Image.fromarray(binary_array)

        # Save the binary image
        binary_save_path = os.path.join(timestamped_folder_path,
                                        f"otsu_threshold_{t_hold}_{is_stretched}_{is_equalised}_{channel}_{image_name}").replace(
            '\\', '/')
        binary_image.save(binary_save_path)

        if state == "o":
            binary_image.show()
        # Display t_hold on histogram
        axis.axvline(x=t_hold, color='k', linestyle='--', alpha=0.7,
                     label=f'Otsu Threshold: {t_hold}')
        axis.legend()

    return figure, axis


def all_hists():  # wszystkie histogramy, +timer
    start_time = time.time()
    global state
    state = "all"

    create_histogram()
    create_histogram("r")
    create_histogram("g")
    create_histogram("b")
    create_histogram("average")
    state = "o"
    elapsed_time = time.time() - start_time
    showinfo(title="Czas", message=f"Ukończono w ciągu {elapsed_time:.4f} sekund")




def binaryz():  # przyciski
    kill_UI()
    left_col = 100
    right_col = 500
    mid_col = 300
    label1 = ttk.Label(window, text="Pliki zostaną zapisany w folderze: {}".format(timestamped_folder_path),
                       background="#e7e7e7")
    label1.place(x=mid_col, y=20)

    show_button = ttk.Button(window, text="pokaz obraz", width=20, command=lambda: image.show())
    show_button.place(x=right_col, y=50)



    bin_normal_button = ttk.Button(window, text="binaryzacja srednia", width=20, command=lambda: binarize_image())
    bin_normal_button.place(x=left_col, y=200)

    bin_red_button = ttk.Button(window, text="binaryzacja czerwona ", width=20, command=lambda: binarize_image('r'))
    bin_red_button.place(x=left_col, y=250)

    bin_green_button = ttk.Button(window, text="binaryzacja zielona", width=20, command=lambda: binarize_image('g'))
    bin_green_button.place(x=left_col, y=300)

    bin_blue_button = ttk.Button(window, text="binaryzacja niebieska", width=20, command=lambda: binarize_image('b'))
    bin_blue_button.place(x=left_col, y=350)

    bin_other_button = ttk.Button(window, text="inne", width=20, command=lambda: (kill_UI(), other_bins_window()))
    bin_other_button.place(x=left_col, y=400)



    bin_full_button = ttk.Button(window, text="wszystkie binaryzacje", width=20, command=lambda: all_bins())
    bin_full_button.place(x=left_col, y=550)


    close_button = ttk.Button(window, text="Zamknij okienko", width=20, command=window.destroy)
    close_button.place(x=left_col, y=700)

    restart_button = ttk.Button(window, text="zacznij od nowa", width=20, command=lambda: select_file())
    restart_button.place(x=right_col, y=700)

    inne_op = ttk.Button(window, text="inne operacje", width=20, command=wybor_dzialan)
    inne_op.place(x=left_col, y=600)

    info_button = ttk.Button(
        window,
        text="Pokaz Info",
        width=20,
        command=lambda: HELP_window(
            "Pokaz obraz pokazuje wybrany obraz \n\n\n "
            "Binaryzacja  srednia/czerwona/zielona/niebieska wykonuje binaryzacju wybranego typu, wynikujacy plik zapisuje\n\n\n"
            "wszystkie binaryzacje/histogramy wykonuje binaryzacje/histogramy kazdego typu, wynikujace pliki zapisuje\n\n\n"
        )
    )
    info_button.place(x=mid_col, y=650)

def binarize_image(method='average'):
    # global image, image_location, image_name, threshold,state

    if method == 'average':  # binaruzje srednia (?)
        grayscale_image = image.convert('L')
        grayscale_array = np.array(grayscale_image)
        binary_array = (grayscale_array > threshold) * 255
        result_image = Image.fromarray(binary_array.astype(np.uint8))
        new_file_path = os.path.join(timestamped_folder_path, f"{threshold}__BlackAndWhite_{image_name}").replace('\\',
                                                                                                                  '/')
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)


    elif method in ['r', 'g', 'b']:  # binaryzuje tylko red, green, blue
        rgb_image = image.convert('RGB')
        rgb_array = np.array(rgb_image)

        channel_index = {'r': 0, 'g': 1, 'b': 2}[method]
        channel_array = rgb_array[:, :, channel_index]
        binary_array = (channel_array > threshold) * 255
        result_image = Image.fromarray(binary_array.astype(np.uint8))
        new_file_path = os.path.join(timestamped_folder_path, f"{threshold}__{method}_{image_name}").replace('\\', '/')
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)

    else:
        raise ValueError(f"metody: 'average', 'r', 'g', 'b'")

    if state == "o":
        result_image.show()
    result_image.save(new_file_path)
    return result_image

def all_bins():  # wszystkie binaryzacje, +timer

    start_time = time.time()
    global state
    state = "all"

    binarize_image()
    binarize_image("r")
    binarize_image("g")
    binarize_image("b")
    state = "o"
    elapsed_time = time.time() - start_time
    showinfo(title="Czas", message=f"Ukończono w ciągu {elapsed_time:.4f} sekund")



def other_bins_window():
    global lc_tresh, sq_size_var, lc_tresh_var

    sq_size_var = tk.StringVar()
    lc_tresh_var = tk.StringVar()

    # K and R value entries (unchanged)
    label_k_val = ttk.Label(window, text="wartosc k:", background="#e7e7e7")
    label_k_val.place(x=50, y=150)

    entry_k_value = ttk.Entry(window, textvariable=k_value_var, width=10)
    entry_k_value.place(x=150, y=150)

    feedback_k_value = ttk.Label(window, text="", background="#e7e7e7")
    feedback_k_value.place(x=150, y=180)

    label_r_val = ttk.Label(window, text="wartosc r:", background="#e7e7e7")
    label_r_val.place(x=250, y=150)

    entry_r_value = ttk.Entry(window, textvariable=r_value_var, width=10)
    entry_r_value.place(x=450, y=150)

    feedback_r_value = ttk.Label(window, text="", background="#e7e7e7")
    feedback_r_value.place(x=450, y=180)

    # Sq size + threshold
    label_sq_size = ttk.Label(window, text="Rozmiar kwadratu:", background="#e7e7e7")
    label_sq_size.place(x=50, y=50)

    entry_sq_size = ttk.Entry(window, textvariable=sq_size_var, width=10)
    entry_sq_size.place(x=150, y=50)

    feedback_sq_size = ttk.Label(window, text="", background="#e7e7e7")
    feedback_sq_size.place(x=150, y=80)

    label_lc_tresh = ttk.Label(window, text="Próg niskiego kontrastu:", background="#e7e7e7")
    label_lc_tresh.place(x=250, y=50)

    entry_lc_tresh = ttk.Entry(window, textvariable=lc_tresh_var, width=10)
    entry_lc_tresh.place(x=450, y=50)

    feedback_lc_tresh = ttk.Label(window, text="", background="#e7e7e7")
    feedback_lc_tresh.place(x=450, y=80)

    # Weighted RGB
    ttk.Label(window, text="czerwony", background="#e7e7e7").place(x=150, y=320)
    ttk.Label(window, text="zielony", background="#e7e7e7").place(x=250, y=320)
    ttk.Label(window, text="niebieski", background="#e7e7e7").place(x=350, y=320)

    entry_red = ttk.Entry(window, textvariable=r_weight, width=10)
    entry_red.place(x=150, y=350)

    entry_green = ttk.Entry(window, textvariable=g_weight, width=10)
    entry_green.place(x=250, y=350)

    entry_blue = ttk.Entry(window, textvariable=b_weight, width=10)
    entry_blue.place(x=350, y=350)

    feedback4 = ttk.Label(window, text="", background="#e7e7e7")
    feedback4.place(x=700, y=380)

    weighted_button = ttk.Button(window, text="weighted", state="disabled", width=20, command=lambda: weighted_rgb_binarization())
    weighted_button.place(x=700, y=350)

    # Buttons
    brensen_button = ttk.Button(window, text="Brensen", width=20, state="disabled", command=lambda: brensen())
    brensen_button.place(x=700, y=50)

    niblack_button = ttk.Button(window, text="niblack", width=20, state="disabled", command=lambda: niblack())
    niblack_button.place(x=700, y=150)

    sauvola_button = ttk.Button(window, text="sauvola", width=20, state="disabled", command=lambda: sauvola())
    sauvola_button.place(x=700, y=250)

    # Tracing inputs
    sq_size_var.trace_add("write", lambda *args: validate_sq_size(
        sq_size_var, feedback_sq_size, brensen_button, niblack_button, sauvola_button,
        lc_tresh_var, k_value_var, r_value_var))

    lc_tresh_var.trace_add("write", lambda *args: validate_lc_tresh(
        lc_tresh_var, feedback_lc_tresh, brensen_button, sq_size_var))

    k_value_var.trace_add("write", lambda *args: validate_k_value(
        k_value_var, feedback_k_value, niblack_button, sauvola_button, sq_size_var, r_value_var))

    r_value_var.trace_add("write", lambda *args: validate_r_value(
        r_value_var, feedback_r_value, sauvola_button, sq_size_var, k_value_var))

    r_weight.trace_add("write", lambda *args: validate_weightsrgb(
        r_weight, g_weight, b_weight, feedback4, weighted_button))
    g_weight.trace_add("write", lambda *args: validate_weightsrgb(
        r_weight, g_weight, b_weight, feedback4, weighted_button))
    b_weight.trace_add("write", lambda *args: validate_weightsrgb(
        r_weight, g_weight, b_weight, feedback4, weighted_button))

    back_button = ttk.Button(window, text="back", width=20, command=lambda: binaryz())
    back_button.place(x=400, y=600)







def brensen():
    method = "brensen"
    global threshold, lc_tresh, sq_size

    # Convert image to grayscale and then to a NumPy array
    img_array = np.array(image.convert('L'))

    # Compute min and max over the sliding window
    z_low = minimum_filter(img_array, size=sq_size, mode='reflect')
    z_high = maximum_filter(img_array, size=sq_size, mode='reflect')

    # Compute threshold
    threshold_bern = (z_low + z_high) // 2

    # Check contrast
    low_contrast_mask = (z_high - z_low) < lc_tresh

    # Apply global threshold to low-contrast regions
    global_threshold = threshold
    output = np.where(low_contrast_mask, np.where(img_array < global_threshold, 0, 255),
                      np.where(img_array < threshold_bern, 0, 255))

    # Ensure the output array is of type uint8
    output = output.astype(np.uint8)

    # Convert back to PIL Image and save
    result = Image.fromarray(output)
    new_file_path = os.path.join(timestamped_folder_path, f"{lc_tresh}_{sq_size}_{method}_{image_name}").replace('\\', '/')
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    result.save(new_file_path)
    result.show()

    return result


def niblack():
    method = "niblack"
    global image, threshold, sq_size, k_value


    # Convert image to grayscale and then to a NumPy array
    img_array = np.array(image.convert('L'))

    # Calculate mean and standard deviation
    m, s = mean_std(img_array, sq_size)

    # Check and fix dimensions if needed
    if m.shape != img_array.shape:
        print(f"Warning: Reshaping threshold from {m.shape} to {img_array.shape}")
        # Option 1: Resize the threshold to match image dimensions
        from scipy.ndimage import zoom
        zoom_factor = (img_array.shape[0] / m.shape[0], img_array.shape[1] / m.shape[1])
        m = zoom(m, zoom_factor, order=1)
        s = zoom(s, zoom_factor, order=1)

    # Compute threshold (T = m(x,y) - k * s(x,y))
    threshold_niblack = m - k_value * s

    # Apply thresholding
    output = np.where(img_array < threshold_niblack, 0, 255)

    # Ensure the output array is of type uint8
    output = output.astype(np.uint8)

    # Convert back to PIL Image and save
    result = Image.fromarray(output)
    new_file_path = os.path.join(timestamped_folder_path, f"{k_value}_{sq_size}_{method}_{image_name}").replace('\\', '/')
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    result.save(new_file_path)
    result.show()

    return result


def sauvola():
    method = "sauvola"
    global image, threshold, sq_size, k_value, r_value

    # Convert image to grayscale and then to a NumPy array
    img_array = np.array(image.convert('L'))

    # Calculate mean and standard deviation using scipy's uniform_filter
    from scipy.ndimage import uniform_filter

    # Ensure window_size is odd
    if isinstance(sq_size, int):
        w_size = sq_size
    else:
        w_size = sq_size[0] if hasattr(sq_size, '__getitem__') else sq_size

    # Calculate local mean
    mean = uniform_filter(img_array.astype(float), size=w_size, mode='reflect')

    # Calculate local squared mean
    squared = uniform_filter(img_array.astype(float) ** 2, size=w_size, mode='reflect')

    # Calculate local std dev
    variance = squared - mean ** 2
    # Handle numeric errors
    variance = np.maximum(variance, 0)
    std_dev = np.sqrt(variance)

    # If r is not specified, use default calculation
    if r_value is None:
        r_value = 0.5 * (np.max(img_array) - np.min(img_array))

    # Compute threshold (T = m(x,y) * (1 + k * ((s(x,y) / R) - 1)))
    threshold_sauvola = mean * (1 + k_value * ((std_dev / r_value) - 1))

    # Apply thresholding
    output = np.where(img_array < threshold_sauvola, 0, 255)

    # Ensure the output array is of type uint8
    output = output.astype(np.uint8)

    # Convert back to PIL Image and save
    result = Image.fromarray(output)
    new_file_path = os.path.join(timestamped_folder_path, f"{k_value}_{r_value}_{sq_size}_{method}_{image_name}").replace('\\', '/')
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    result.save(new_file_path)
    result.show()

    return result


def weighted_rgb_binarization():
    global image, r_weight, g_weight, b_weight, threshold

    # Convert to RGB if not already
    rgb_image = image.convert('RGB')
    rgb_array = np.array(rgb_image)

    # Get individual channels
    r_channel = rgb_array[:, :, 0].astype(float)
    g_channel = rgb_array[:, :, 1].astype(float)
    b_channel = rgb_array[:, :, 2].astype(float)

    # Get weight values as floats (normalized between 0 and 1)
    r_weight_val = int(r_weight.get()) / 255
    g_weight_val = int(g_weight.get()) / 255
    b_weight_val = int(b_weight.get()) / 255

    # Sum of weights for normalization
    weight_sum = r_weight_val + g_weight_val + b_weight_val
    if weight_sum == 0:  # Avoid division by zero
        weight_sum = 1

    # Apply weights to each channel
    r_weighted = r_channel * r_weight_val
    g_weighted = g_channel * g_weight_val
    b_weighted = b_channel * b_weight_val

    # Calculate weighted average
    weighted_sum = (r_weighted + g_weighted + b_weighted) / weight_sum

    # Apply threshold to the weighted sum
    final_binary = (weighted_sum > threshold) * 255

    # Convert to image
    result_image = Image.fromarray(final_binary.astype(np.uint8), mode='L')


    new_file_path = os.path.join(timestamped_folder_path, f"{r_weight}_{g_weight}_{b_weight}_'wagi_rgb'_{image_name}").replace('\\', '/')
    os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    result_image.save(new_file_path)
    result_image.show()

    return result_image




def validate_weightsrgb(r_weight, g_weight, b_weight, feedback4, weighted_button):
    val1 = r_weight.get().strip()
    val2 = g_weight.get().strip()
    val3 = b_weight.get().strip()

    # Fixed the condition to check val2 correctly
    if (val1.isdigit() and (0 <= int(val1) <= 255)) and \
       (val2.isdigit() and (0 <= int(val2) <= 255)) and \
       (val3.isdigit() and (0 <= int(val3) <= 255)):
        feedback4.config(text="git", foreground="green")
        weighted_button.config(state="normal")
    else:
        feedback4.config(text="Invalid (0-255 tylko)", foreground="red")
        weighted_button.config(state="disabled")

def validate_sq_size(sq_size_var, feedback, brensen_button, niblack_button, sauvola_button, lc_tresh_var, k_value_var, r_value_var):
    global sq_size
    val = sq_size_var.get().strip()
    try:
        num_val = int(val)
        if 3 <= num_val <= 101 and num_val % 2 == 1:
            sq_size = num_val  # Save the **actual int** for later use
            feedback.config(text="Git", foreground="green")

            if is_valid_lc_tresh(lc_tresh_var.get().strip()):
                brensen_button.config(state="normal")
            if is_valid_k_value(k_value_var.get().strip()):
                niblack_button.config(state="normal")
            if is_valid_k_value(k_value_var.get().strip()) and is_valid_r_value(r_value_var.get().strip()):
                sauvola_button.config(state="normal")
        else:
            raise ValueError
    except ValueError:
        feedback.config(text="Niepoprawne (3-101, nieparzyste)", foreground="red")
        brensen_button.config(state="disabled")
        niblack_button.config(state="disabled")
        sauvola_button.config(state="disabled")

def validate_lc_tresh(lc_tresh_var, feedback, submit_button, sq_size_var):
    global lc_tresh
    val = lc_tresh_var.get().strip()

    if is_valid_lc_tresh(val):
        lc_tresh = int(val)
        feedback.config(text="Git", foreground="green")
        # Check if both inputs are valid to enable submit button
        if is_valid_sq_size(sq_size_var.get().strip()):
            submit_button.config(state="normal")
    else:
        feedback.config(text="Invalid\n\n\n(wartości tylko od 0 do 255)", foreground="red")
        submit_button.config(state="disabled")

def is_valid_lc_tresh(val):
    try:
        num_val = int(val)
        return 0 <= num_val <= 255
    except ValueError:
        return False

def is_valid_sq_size(val):
    try:
        num_val = int(val)
        return 3 <= num_val <= 101 and num_val % 2 == 1
    except ValueError:
        return False

def _correlate_sparse(image, kernel_shape, kernel_indices, kernel_values):
    idx, val = kernel_indices[0], kernel_values[0]
    if tuple(idx) != (0,) * image.ndim:
        raise RuntimeError("Unexpected initial index in kernel_indices")
    out = image[tuple(slice(None, s) for s in image.shape)][:kernel_shape[0], :kernel_shape[1]].copy()
    for idx, val in zip(kernel_indices[1:], kernel_values[1:]):
        out += image[tuple(slice(i, i + s) for i, s in zip(idx, kernel_shape))] * val
    return out

def mean_std(image, w):
    if not isinstance(w, Iterable):
        w = (w,) * image.ndim

    pad_width = tuple((k // 2 + 1, k // 2) for k in w)
    padded = np.pad(image.astype(np.float64, copy=False), pad_width, mode='reflect')

    integral = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    padded_sq = padded * padded
    integral_sq = np.cumsum(np.cumsum(padded_sq, axis=0), axis=1)

    kernel_indices = list(itertools.product(*tuple([(0, _w) for _w in w])))
    kernel_values = [(-1) ** (image.ndim % 2 != np.sum(indices) % 2) for indices in kernel_indices]

    total_window_size = math.prod(w)
    kernel_shape = tuple(_w + 1 for _w in w)

    m = _correlate_sparse(integral, kernel_shape, kernel_indices, kernel_values)
    m = m.astype(np.float64, copy=False) / total_window_size

    g2 = _correlate_sparse(integral_sq, kernel_shape, kernel_indices, kernel_values)
    g2 = g2.astype(np.float64, copy=False) / total_window_size

    s = np.sqrt(np.clip(g2 - m * m, 0, None))
    return m, s

def validate_k_value(k_value_var, feedback, niblack_button, sauvola_button, window_size_var, r_value_var):
    global k_value
    val = k_value_var.get().strip()

    try:
        num_val = float(val)
        if -0.5 <= num_val <= 0.5:
            k_value = num_val
            feedback.config(text="git", foreground="green")
            # Check if other inputs are valid to enable submit button
            if is_valid_window_size(window_size_var.get().strip()) and is_valid_r_value(r_value_var.get().strip()):
                niblack_button.config(state="normal")
                sauvola_button.config(state="normal")
        else:
            feedback.config(text="Invalid\n\n(musi byc miedzy -0.5 i 0.5)", foreground="red")
            niblack_button.config(state="disabled")
            sauvola_button.config(state="disabled")
    except ValueError:
        feedback.config(text="Invalid\n\n(musi byc liczba dziesietna)", foreground="red")
        niblack_button.config(state="disabled")
        sauvola_button.config(state="disabled")

def validate_r_value(r_value_var, feedback, sauvola_button, window_size_var, k_value_var):
    global r_value
    val = r_value_var.get().strip()

    if val.lower() == "none" or val == "":
        r_value = None
        feedback.config(text="git", foreground="green")
        # Check if other inputs are valid to enable sauvola_button
        if is_valid_window_size(window_size_var.get().strip()) and is_valid_k_value(k_value_var.get().strip()):
            sauvola_button.config(state="normal")
        else:
            sauvola_button.config(state="disabled")
        return True

    try:
        num_val = float(val)
        if 1 <= num_val <= 255:
            r_value = num_val
            feedback.config(text="git", foreground="green")
            # Check if other inputs are valid to enable sauvola_button
            if is_valid_window_size(window_size_var.get().strip()) and is_valid_k_value(k_value_var.get().strip()):
                sauvola_button.config(state="normal")
            else:
                sauvola_button.config(state="disabled")
            return True
        else:
            feedback.config(text="Invalid\n\nmusi byc od 0 do 255)", foreground="red")
            sauvola_button.config(state="disabled")
            return False
    except ValueError:
        feedback.config(text="Invalid\n\n(musi byc liczba')", foreground="red")
        sauvola_button.config(state="disabled")
        return False

def is_valid_window_size(val):
    try:
        num_val = int(val)
        return 3 <= num_val <= 51 and num_val % 2 == 1
    except ValueError:
        return False

def is_valid_k_value(val):
    try:
        num_val = float(val)
        return -0.5 <= num_val <= 0.5
    except ValueError:
        return False

def is_valid_r_value(val):
    if val.lower() == "none" or val == "":
        return True
    try:
        num_val = float(val)
        return 1 <= num_val <= 255
    except ValueError:
        return False










open_button1 = ttk.Button(
    window,
    text='Otworz plik',
    command=select_file)
open_button1.place(x=250, y=400)
open_button2 = ttk.Button(
    window,
    text='Otworz folder',
    command=select_folder)
open_button2.place(x=500, y=400)



window.mainloop()