import pandas as pd
import nltk
import string
import re
import tkinter as tk
from tkinter import filedialog
import os

# Define the main window
root = tk.Tk()
root.title("Pre-Processing Tool")

# Define the widgets
file_label = tk.Label(root, text="Choose a CSV file to pre-process:")
file_label.pack(pady=10)

def browse_file():
    global df, csv_file_path
    csv_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if csv_file_path:
        df = pd.read_csv(csv_file_path)
        output.config(state=tk.NORMAL)
        output.delete(1.0, tk.END)
        output.insert(tk.END, df.head())
        output.config(state=tk.DISABLED)

browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.pack(pady=5)

stem_label = tk.Label(root, text="Do you want to perform stemming?")
stem_label.pack(pady=10)

stem_var = tk.BooleanVar()
stem_var.set(False)

stem_radio1 = tk.Radiobutton(root, text="No", variable=stem_var, value=False)
stem_radio2 = tk.Radiobutton(root, text="Yes", variable=stem_var, value=True)
stem_radio1.pack(pady=5)
stem_radio2.pack(pady=5)

stopword_label = tk.Label(root, text="Choose a stopword file:")
stopword_label.pack(pady=10)

def browse_stopword_file():
    global stopword_filename
    stopword_filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    stopword_path_label.config(text=stopword_filename)  # Update the label with the selected path

stopword_button = tk.Button(root, text="Browse Stopword File", command=browse_stopword_file)
stopword_button.pack(pady=5)

stopword_path_label = tk.Label(root, text="")
stopword_path_label.pack(pady=5)

output = tk.Text(root, height=10, width=80, state=tk.DISABLED)
output.pack(pady=10)

def preprocess():
    global df, stopword_filename
    output.config(state=tk.NORMAL)
    output.delete(1.0, tk.END)

    # Pre-processing
    df['judul_prep'] = df['judul'].str.lower()
    df['judul_prep'] = df['judul_prep'].apply(lambda x: re.sub(r"\b[a-zA-Z]\b", "", x))
    df['judul_prep'] = df['judul_prep'].str.replace("[^A-Za-z\s]+"," ")

    if stopword_filename:
        list_stopwords =  nltk.corpus.stopwords.words('indonesian')
        txt_stopword = pd.read_csv(stopword_filename, names= ["stopwords"], header = None)
        list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
        df['judul_prep'] = df['judul_prep'].apply(lambda x: ' '.join([word for word in x.split() if word not in (list_stopwords)]))
    
    if stem_var.get():
        factory = nltk.stem.snowball.SnowballStemmer('indonesian')
        df['judul_prep'] = df['judul_prep'].apply(lambda x: ' '.join([factory.stem(word) for word in x.split()]))

    # Display the pre-processed data
    output.insert(tk.END, df.head())

    # Save the pre-processed data as a CSV file in the same folder as the script
    if csv_file_path:
        output_file_path = os.path.splitext(csv_file_path)[0] + "_preprocessed.csv"
        df.to_csv(output_file_path, index=False)
        output.insert(tk.END, f"\n\nPre-processed data saved as '{output_file_path}' in the same folder as the script.")

    output.config(state=tk.DISABLED)

preprocess_button = tk.Button(root, text="Pre-Process", command=preprocess)
preprocess_button.pack(pady=10)

root.mainloop()
