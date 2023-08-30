import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from gensim import corpora
from gensim.models import TfidfModel, CoherenceModel
from gensim.models.nmf import Nmf
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import tempfile
from tkinter import ttk
from tkhtmlview import HTMLLabel
import webbrowser
import html
import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App:
    def __init__(self, master):
        self.master = master
        master.title("NMF Model")

        self.csv_label = tk.Label(master, text="Choose CSV file:")
        self.csv_label.pack()

        self.csv_button = tk.Button(master, text="Choose", command=self.choose_csv)
        self.csv_button.pack()

        # File Path Label
        self.file_path_label = tk.Label(master, text="")
        self.file_path_label.pack()

        self.num_topics_label = tk.Label(master, text="Enter number of topics:")
        self.num_topics_label.pack()

        self.num_topics_entry = tk.Entry(master)
        self.num_topics_entry.pack()

        self.build_button = tk.Button(master, text="Build Model", command=self.build_nmf)
        self.build_button.pack()

        self.coherence_label = tk.Label(master, text="Coherence: ")
        self.coherence_label.pack()

        self.tmp = None
        self.nmf_model = None
        self.corpus = None
        self.dictionary = None

        self.vis_button = tk.Button(master, text="Show Visualization", command=self.show_vis)
        self.vis_button.pack()

        self.save_button = tk.Button(master, text="Save Model", command=self.save_model)
        self.save_button.pack()

        self.result_label = tk.Label(master, text="Result Top Keyword Each Topic :")
        self.result_textbox = tk.Text(master, height=10, width=80)
        self.result_label.pack()
        self.result_textbox.pack()

        self.browser = None


    def choose_csv(self):
        self.file_path = filedialog.askopenfilename(filetypes=(("CSV Files", "*.csv"),))
        print("CSV file:", self.file_path)

        # Update file path label
        self.file_path_label.config(text="File Path: " + self.file_path)


        # Read CSV file and extract judul_prep column data
        df = pd.read_csv(self.file_path)
        self.judul_prep = df['judul_prep']

        
    def build_nmf(self):
        

        texts = [[word for word in document.lower().split()] for document in self.judul_prep]

        dictionary = corpora.Dictionary(texts)
        corpus_bow = [dictionary.doc2bow(text) for text in texts]
        tfidf = TfidfModel(corpus_bow)
        corpus_tfidf = tfidf[corpus_bow]
        num_topics = int(self.num_topics_entry.get())
        nmf_model = Nmf(corpus=corpus_tfidf,num_topics= num_topics, id2word=dictionary, chunksize=2000,
                              passes=5,kappa=1.0,minimum_probability=0.01,w_max_iter=200,w_stop_condition=0.0001,
                              h_max_iter=50,h_stop_condition=0.001,eval_every=10,normalize=True,random_state=42)
        
        coherence_model_nmf = CoherenceModel(model=nmf_model, corpus=corpus_tfidf, dictionary=dictionary, coherence='u_mass')
        coherence_nmf = coherence_model_nmf.get_coherence()


        self.coherence_label.config(text=f"Coherence: {coherence_nmf}")

        self.nmf_model = nmf_model
        self.dictionary = dictionary
        self.corpus = corpus_tfidf

    # Display the top topics and their related keywords
        topics = nmf_model.show_topics(num_topics=num_topics, num_words=10)
        topics_str = ""
        for topic in topics:
            topic_str = f"Topic {topic[0]}: "
            topic_str += ", ".join([word[0] for word in nmf_model.show_topic(topic[0], topn=10)])
            topic_str += "\n"
            topics_str += topic_str
          
        save_results = messagebox.askyesno("Save Results", "Do you want to save the results to a CSV file?")
        if save_results:
            for topic in topics:
                # Get the topic assignment for each document in the corpus
                topic_assignment = nmf_model.get_document_topics(corpus_tfidf)
                topic_assignment = [(doc_id, topic_id, prob) for doc_id, topics in enumerate(topic_assignment) for topic_id, prob in topics]

                # Create a pandas DataFrame for the topic assignment
                df_topic = pd.DataFrame(topic_assignment, columns=['doc_id', 'topic_id', 'topic_prob'])

                # Filter the DataFrame for the current topic
                df_topic = df_topic[df_topic['topic_id'] == topic[0]]

                # Join the topic keywords to the DataFrame
                df_topic['topic_keywords'] = "".join([word[0] for word in topic[1]])

                # Join the original document text to the DataFrame
                df_topic['document_text'] = self.judul_prep

                # Save the DataFrame to a CSV file for the current topic
                filename = f"topic_NMF_{topic[0]}.csv"
                df_topic.to_csv(filename, index=False)

        self.result_textbox.delete("1.0", tk.END) # clear previous results
        self.result_textbox.insert(tk.END, topics_str)
        
        messagebox.showinfo("NMF Model", "Model built successfully!")


    def show_graph(self,nmf_model, num_topics, dictionary):
        # create a graph object
        G = nx.Graph()

        # add nodes for topics
        for i in range(num_topics):
            G.add_node(f"Topic {i}")

        # add edges for top 10 most relevant words for each topic
        for topic_id, topic_words in nmf_model.show_topics(num_topics=num_topics, num_words=10, formatted=False):
            for word, relevance in topic_words:
                G.add_edge(f"Topic {topic_id}", word, weight=relevance)

        # draw the graph
        pos = nx.spring_layout(G, k=0.3, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1000)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        nx.draw_networkx_edges(G, pos, width=2, edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), font_size=10)

        # create a canvas to display the graph
        fig = plt.gcf()
        fig.set_size_inches(10, 8)  # adjust figure size as desired
        
        # create a new window to display the canvas
        window = tk.Toplevel(self.master)
        window.geometry("800x600")
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        window.mainloop()
        
    def show_vis(self):
        if self.nmf_model is None:
            messagebox.showerror("Error", "Please build a model first")
            return

        num_topics = int(self.num_topics_entry.get())
        self.show_graph(self.nmf_model, num_topics, self.dictionary)

    def save_model(self):
        try:
            filename = filedialog.asksaveasfilename(defaultextension=".model", filetypes=(("NMF Model", "*.model"),))
            self.nmf_model.save(filename)
            messagebox.showinfo("Save Model", "Model saved successfully!")
        except AttributeError:
            messagebox.showerror("Error", "Please build a model first.")

root = tk.Tk()
app = App(root)
root.mainloop()