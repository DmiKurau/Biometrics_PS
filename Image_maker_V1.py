import os
import tkinter as tk #GUI window
from tkinter import ttk #input fields
from tkinter import filedialog as fd #file open
from tkinter.messagebox import showinfo #info box
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime


window = tk.Tk()
window.config(background="#e7e7e7")
window.geometry("600x800")
window.resizable(False, False)
window.title("Program zdjeciowy")


image_name = ""
image_location = ""
threshold = 0  # zmienia binaryzacje, mniej = wiecej bialego
image = []


# image.show()


# mono_image = Image.fromarray(binary_array);  nie dziala, binary array: signed, PIL potrzebuje unsigned



def select_file(): #wybiera plik (zdjecie)
    global dataframe, filename, image_location, image_name, image, timestamped_folder_path
    filetypes =(
        ('jpg files', '*.jpg'),
        ('All files', '*.*'))
    filename = fd.askopenfilename(
        title='Wybierz plik',
        initialdir='/',
        filetypes=filetypes)
    if filename:
        try:
            image_location, image_name = os.path.split(filename)
            full_path = os.path.join(image_location, image_name)
            image = Image.open(full_path)
            showinfo(title="plik otwarty", message=f"otwarto plik: {filename}")
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            timestamped_folder_path = os.path.join(image_location, timestamp)
            timestamped_folder_path = os.path.normpath(timestamped_folder_path).replace('\\', '/')

            kill_UI()
            get_thresh()


        except Exception as e:
            showinfo(title="Blad", message=f"{e}")


def get_thresh(): #wpisywanie progu dla binaryzacji
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


def validate_thresh(rat, feedback4, submit_button): #walidacja tego progu
    global threshold
    val = rat.get().strip()

    if val.isdigit() and (0 <= int(val) <= 255):
        threshold = int(val)
        feedback4.config(text="Prawidlowe", foreground="green")
        submit_button.config(state="normal")
    else:
        feedback4.config(text="Nie Prawidlowe \n\n\n (wartosci tylko od 0 do 255)", foreground="red")
        submit_button.config(state="disabled")







def make_images(): #przyciski
    show_button = ttk.Button(window, text="pokaz obraz", width=20, command= lambda: image.show())
    show_button.place(x=190, y=50)

    bin_normal_button = ttk.Button(window, text="binaryzacja srednia", width=20, command= lambda: binarize_image())
    bin_normal_button.place(x=90, y=100)

    bin_red_button = ttk.Button(window, text="binaryzacja czerwona ", width=20, command= lambda: binarize_image('r'))
    bin_red_button.place(x=90, y=150)

    bin_green_button = ttk.Button(window, text="binaryzacja zielona", width=20, command= lambda: binarize_image('g'))
    bin_green_button.place(x=90, y=200)

    bin_blue_button = ttk.Button(window, text="binaryzacja niebieska", width=20, command= lambda: binarize_image('b'))
    bin_blue_button.place(x=90, y=250)

    hist_all_button = ttk.Button(window, text="sredni histogram", width=20, command= lambda: create_histogram())
    hist_all_button.place(x=290, y=100)

    hist_red_button = ttk.Button(window, text="czerwony histogram ", width=20, command= lambda: create_histogram('r'))
    hist_red_button.place(x=290, y=150)

    hist_green_button = ttk.Button(window, text="zielony histogram", width=20, command= lambda: create_histogram('g'))
    hist_green_button.place(x=290, y=200)

    hist_blue_button = ttk.Button(window, text="niebieski histogram", width=20, command= lambda: create_histogram('b'))
    hist_blue_button.place(x=290, y=250)

    hist_gray_button = ttk.Button(window, text="szary histogram", width=20, command= lambda: create_histogram('average'))
    hist_gray_button.place(x=290, y=300)



    bin_full_button = ttk.Button(window, text="wszystkie binaryzacje", width=20, command= lambda: all_bins())
    bin_full_button.place(x=90, y=400)

    hist_full_button = ttk.Button(window, text="wszystkie histogramy", width=20, command= lambda: all_hists())
    hist_full_button.place(x=290, y=400)

    all_button = ttk.Button(window, text="wszystko", width=20, command= lambda: all_everything())
    all_button.place(x=190, y=500)

    close_button = ttk.Button(window, text="Zamknij okienko", width=20, command=window.destroy)
    close_button.place(x=90, y=700)

    restart_button = ttk.Button(window, text="zacznij od nowa", width=20, command= lambda: select_file())
    restart_button.place(x=290, y=700)


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
    info_button.place(x=190, y=600)





def all_bins(): #wszystkie binaryzacje, +timer
    start_time = time.time()

    binarize_image()
    binarize_image("r")
    binarize_image("g")
    binarize_image("b")

    elapsed_time = time.time() - start_time
    showinfo(title="Czas", message=f"Ukończono w ciągu {elapsed_time:.4f} sekund")

def all_hists(): #wszystkie histogramy, +timer
    start_time = time.time()

    create_histogram()
    create_histogram("r")
    create_histogram("g")
    create_histogram("b")
    create_histogram("average")

    elapsed_time = time.time() - start_time
    showinfo(title="Czas", message=f"Ukończono w ciągu {elapsed_time:.4f} sekund")

def all_everything(): #binaryzacje + histogramy, +timer
    start_time = time.time()

    binarize_image()
    binarize_image("r")
    binarize_image("g")
    binarize_image("b")
    create_histogram()
    create_histogram("r")
    create_histogram("g")
    create_histogram("b")
    create_histogram("average")

    elapsed_time = time.time() - start_time
    showinfo(title="Czas", message=f"Ukończono w ciągu {elapsed_time:.4f} sekund")




def binarize_image(method='average'):
    global image, image_location, image_name, threshold
    if method == 'average':  # binaruzje srednia (?)
        grayscale_image = image.convert('L')
        grayscale_array = np.array(grayscale_image)
        binary_array = (grayscale_array > threshold) * 255
        result_image = Image.fromarray(binary_array.astype(np.uint8))
        new_file_path = os.path.join(timestamped_folder_path, f"{threshold}__BlackAndWhite_{image_name}").replace('\\', '/')
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

    result_image.save(new_file_path)
    return result_image


def create_histogram(channel='all'):
    global image, image_location, image_name

    figure, axis = plt.subplots(figsize=(10, 6))

    if channel == 'all':

        rgb_image = image.convert('RGB')
        rgb_array = np.array(rgb_image)

        colors = ['r', 'g', 'b']
        channel_names = ('Red', 'Green', 'Blue')

        for i, color in enumerate(colors):
            channel_data = rgb_array[:, :, i].flatten()
            axis.hist(channel_data, bins=256, range=(0, 255),
                      color=color, alpha=0.7, label=channel_names[i])

        axis.set_title('RGB Histogram')
        axis.legend()

    elif channel in ['r', 'g', 'b']:
        rgb_image = image.convert('RGB')
        rgb_array = np.array(rgb_image)

        #get specific channel
        channel_index = {'r': 0, 'g': 1, 'b': 2}[channel]
        channel_data = rgb_array[:, :, channel_index].flatten()


        color_name = {'r': 'Red', 'g': 'Green', 'b': 'Blue'}[channel]
        axis.hist(channel_data, bins=256, range=(0, 255),
                  color=channel, alpha=0.7, label=color_name)

        axis.set_title(f'Histogram kanalu: {color_name}')
        axis.legend()

    elif channel == 'average':
        gray_image = image.convert('L')
        gray_array = np.array(gray_image).flatten()


        axis.hist(gray_array, bins=256, range=(0, 255),
                  color='gray', alpha=0.7, label='Grayscale')

        axis.set_title('histogram szary')
        axis.legend()


    axis.set_xlabel('Wartosc pikseli')
    axis.set_ylabel('czestosc')
    axis.set_xlim([0, 255])
    axis.grid(True, alpha=0.3)




    plt.tight_layout()
    save_path = os.path.join(timestamped_folder_path, f"histogram_{channel}_{image_name}").replace('\\', '/')

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


    plt.savefig(save_path)




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




open_button = ttk.Button  (
    window,
    text='Otworz plik',
    command=select_file   )
open_button.place(x=250, y=400)

window.mainloop()