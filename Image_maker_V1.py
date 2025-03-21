import os
import tkinter as tk  # GUI window
from tkinter import ttk  # input fields
from tkinter import filedialog as fd  # file open
from tkinter.messagebox import showinfo  # info box
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from datetime import datetime
from scipy.ndimage import minimum_filter, maximum_filter
import math
import itertools
from collections.abc import Iterable

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


# image.show()


# mono_image = Image.fromarray(binary_array);  nie dziala, binary array: signed, PIL potrzebuje unsigned


def select_file():  # wybiera plik (zdjecie)
    kill_UI()
    global dataframe, filename, image_location, image_name, image, timestamped_folder_path
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
            showinfo(title="plik otwarty", message=f"otwarto plik: {filename}")
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            timestamped_folder_path = os.path.join(image_location, timestamp)
            timestamped_folder_path = os.path.normpath(timestamped_folder_path).replace('\\', '/')

            kill_UI()
            get_thresh()


        except Exception as e:
            showinfo(title="Blad", message=f"{e}")


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
        command=lambda: (kill_UI(), make_images())
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


def make_images():  # przyciski
    kill_UI()
    left_col = 100
    right_col = 500
    mid_col = 300
    label1 = ttk.Label(window, text="Pliki zostaną zapisany w folderze: {}".format(image_location),
                       background="#e7e7e7")
    label1.place(x=mid_col, y=20)

    show_button = ttk.Button(window, text="pokaz obraz", width=20, command=lambda: image.show())
    show_button.place(x=right_col, y=50)

    stretching_checkbox = ttk.Checkbutton(window, text="Rozciagniecie histogramu", variable=stretch)
    stretching_checkbox.place(x=left_col, y=100)

    equalisation_checkbox = ttk.Checkbutton(window, text="wyrownanie histogramu", variable=equalise)
    equalisation_checkbox.place(x=mid_col, y=100)

    otsu_checkbox = ttk.Checkbutton(window, text="metoda Otsu", variable=otsu)
    otsu_checkbox.place(x=right_col, y=100)

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

    bin_full_button = ttk.Button(window, text="wszystkie binaryzacje", width=20, command=lambda: all_bins())
    bin_full_button.place(x=left_col, y=550)

    hist_full_button = ttk.Button(window, text="wszystkie histogramy", width=20, command=lambda: all_hists())
    hist_full_button.place(x=right_col, y=550)

    all_button = ttk.Button(window, text="wszystko", width=20, command=lambda: all_everything())
    all_button.place(x=mid_col, y=600)

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
            "Binaryzacja  srednia/czerwona/zielona/niebieska wykonuje binaryzacju wybranego typu, wynikujacy plik zapisuje\n\n\n"
            "Histogram  sredni/czerwony/zielony/niebieski/szary robi histogram wybranego typu, wynikujacy plik zapisuje\n\n\n"
            "wszystkie binaryzacje/histogramy wykonuje binaryzacje/histogramy kazdego typu, wynikujace pliki zapisuje\n\n\n"
            "przycisk \"wszystko\" wykonuje wszystkie binaryzacje/histogramy, wynikujace pliki zapisuje\n\n\n"
        )
    )
    info_button.place(x=mid_col, y=650)


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

    back_button = ttk.Button(window, text="back", width=20, command=lambda: make_images())
    back_button.place(x=400, y=600)



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


def all_everything():  # binaryzacje + histogramy, +timer
    start_time = time.time()
    global state
    state = "all"

    binarize_image()
    binarize_image("r")
    binarize_image("g")
    binarize_image("b")
    create_histogram()
    create_histogram("r")
    create_histogram("g")
    create_histogram("b")
    create_histogram("average")
    state = "o"
    elapsed_time = time.time() - start_time
    showinfo(title="Czas", message=f"Ukończono w ciągu {elapsed_time:.4f} sekund")


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


#########################################################################################################
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


def kill_UI():
    for widget in window.winfo_children():
        widget.place_forget()


def HELP_window(message):
    message_window = tk.Toplevel()
    message_window.title("Info")
    message_window.geometry("500x350")

    label = ttk.Label(message_window, text=message, wraplength=400)
    label.pack(pady=20)

    close_button = ttk.Button(message_window, text="Zamknij", command=message_window.destroy)
    close_button.pack(pady=10)


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

# Helper functions to check validity
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







open_button = ttk.Button(
    window,
    text='Otworz plik',
    command=select_file)
open_button.place(x=250, y=400)

window.mainloop()