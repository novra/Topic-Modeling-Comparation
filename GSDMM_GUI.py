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
from tkinter import Toplevel
import numpy as np
import webbrowser
import html
import os
from gsdmm import MovieGroupProcess
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App:
    def __init__(self, master):
        self.master = master
        master.title("GSDMM Model")

        self.csv_label = tk.Label(master, text="Choose CSV file:")
        self.csv_label.pack()

        self.csv_button = tk.Button(master, text="Choose", command=self.choose_csv)
        self.csv_button.pack()

        # File Path Label
        self.file_path_label = tk.Label(master, text="")
        self.file_path_label.pack()

        self.num_topics = None

        self.num_topics_label = tk.Label(master, text="Enter number of topics:")
        self.num_topics_label.pack()

        self.num_topics_entry = tk.Entry(master)
        self.num_topics_entry.pack()



        self.build_button = tk.Button(master, text="Build Model", command=self.build_gsdmm)
        self.build_button.pack()

        self.numdoc_label = tk.Label(master, text="Number of documents per topic : ")
        self.numdoc_label.pack()

        self.cluster_label = tk.Label(master, text="Most important clusters (by number of docs inside) : ")
        self.cluster_label.pack()

        self.coherence_label = tk.Label(master, text="Coherence: ")
        self.coherence_label.pack()

        self.vis_button = tk.Button(master, text="Show Visualization", command=self.show_vis)
        self.vis_button.pack()

        self.save_button = tk.Button(master, text="Save Model", command=self.save_model)
        self.save_button.pack()

        self.result_label = tk.Label(master, text="Result Top Keyword Each Topic :")
        self.result_textbox = tk.Text(master, height=20, width=100)
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
        self.judul = df['judul']
    
    def build_gsdmm(self):
        

        texts = [[word for word in document.lower().split()] for document in self.judul_prep]

        dictionary = corpora.Dictionary(texts)
        vocab_length = len(dictionary)
        corpus_bow = [dictionary.doc2bow(text) for text in texts]
        tfidf = TfidfModel(corpus_bow)
        corpus_tfidf = tfidf[corpus_bow]
        num_topics = int(self.num_topics_entry.get())
        
        self.num_topics = num_topics
        # initialize GSDMM
        gsdmm_model = MovieGroupProcess(K=num_topics, alpha=0.1, beta=0.3, n_iters=30)
        
        self.gsdmm_model = gsdmm_model
        # fit GSDMM model
        y = gsdmm_model.fit(texts, vocab_length)
        
        #display the result 
        import numpy as np
        # print number of documents per topic
        doc_count = np.array(gsdmm_model.cluster_doc_count)
        self.numdoc_label.config(text=f"Number of documents per topic : {doc_count}")

        # Topics sorted by the number of document they are allocated to
        top_index = doc_count.argsort()[-15:][::-1]
        self.cluster_label.config(text=f"Most important clusters (by number of docs inside): {top_index}")
        
         # define function to get words in topics
        def get_topics_lists(model, top_clusters, n_words):
    
            # create empty list to contain topics
            topics = []
    
            # iterate over top n clusters
            for cluster in top_clusters:
            #create sorted dictionary of word distributions
                sorted_dict = sorted(model.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:n_words]
         
                #create empty list to contain words
                topic = []
        
             #iterate over top n words in topic
                for k,v in sorted_dict:
                    #append words to topic list
                    topic.append(k)
            
                #append topics to topics list    
                topics.append(topic)
    
            return topics
        # get topics to feed to coherence model
        topics = get_topics_lists(gsdmm_model, top_index, num_topics) 
        
        self.topic = topics

        # evaluate model using Topic Coherence score
        cm_gsdmm = CoherenceModel(topics=topics, 
                                dictionary=dictionary, 
                                corpus=corpus_tfidf, 
                                texts=texts, 
                                coherence='u_mass')

        # get coherence value
        coherence_gsdmm = cm_gsdmm.get_coherence()  

        self.coherence_label.config(text=f"Coherence: {coherence_gsdmm}")

        def top_words(cluster_word_distribution, top_cluster, values):
            result_str = ""
            for cluster in top_cluster:
                sort_dicts = sorted(cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
                result_str += f"\nCluster {cluster} : {sort_dicts}"
            return result_str
        
        save_results = messagebox.askyesno("Save Results", "Do you want to save the results to a CSV file?")
        if save_results:
            
            def create_topics_dataframe(data_text=self.judul,  mgp=gsdmm_model, threshold=0.3, token=texts):
                topic_dict = {}
                for i, topic_num in enumerate(top_index):
                    topic_dict[topic_num] = f'Topic {i+1}'
                result = pd.DataFrame(columns=['text', 'topic', 'token'])
                for i, text in enumerate(data_text):
                    result.at[i, 'text'] = text
                    result.at[i, 'token'] = token[i]
                    prob = mgp.choose_best_label(token[i])
                    if prob[1] >= threshold:
                        result.at[i, 'topic'] = topic_dict[prob[0]]
                    else:
                        result.at[i, 'topic'] = 'Other'
                return result
           
            result_df = create_topics_dataframe()
            result_df.to_csv('GSDMM_result_topic.csv', index=False)

        self.result_textbox.delete("1.0", tk.END) # clear previous results
        top_words_str = top_words(gsdmm_model.cluster_word_distribution, top_index, 15)
        self.result_textbox.insert(tk.END, top_words_str)
        
        messagebox.showinfo("GSDMM Model", "Model built successfully!")

    def show_vis(self):
       
       # Get the top keywords for each topic
        top_keywords = self.topic
        # create a network graph using the GSDMM model
        g = nx.DiGraph()
        
        # add nodes to the graph
        for i in range(self.num_topics):
            g.add_node(i, label=f"Topic {i}")
        
        # add edges to the graph
        #for i, (cluster_distr, _) in enumerate(zip(self.gsdmm_model.cluster_word_distribution, self.gsdmm_model.cluster_doc_count)):
            #for j, score in cluster_distr.items():
                #g.add_edge(i, j, weight=score)

        for i, cluster_keywords in enumerate(top_keywords):
            for j, keyword in enumerate(cluster_keywords):
                g.add_edge(i, j, keyword=keyword)
        
        # show the graph in a new window
        graph_window = tk.Toplevel(self.master)
        graph_window.geometry("800x600")
        graph_window.title("GSDMM Model Network Graph")
        pos = nx.spring_layout(g)
        nx.draw_networkx(g, pos)
        labels = nx.get_node_attributes(g, 'label')
        nx.draw_networkx_labels(g, pos, labels, font_size=12)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g, 'keyword'))
        canvas = FigureCanvasTkAgg(plt.gcf(), master=graph_window)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()

        # show the graph in a new window
        #graph_window = tk.Toplevel(self.master)
        #graph_window.title("GSDMM Model Network Graph")
        #pos = nx.spring_layout(g)
        #nx.draw_networkx(g, pos)
        #labels = nx.get_node_attributes(g, 'label')
        #nx.draw_networkx_labels(g, pos, labels)
        #edge_labels = nx.get_edge_attributes(g, 'keyword')
        #nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
        #plt.axis('off')  # Disable the axis
        #canvas = FigureCanvasTkAgg(plt.gcf(), master=graph_window)
        #canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        #canvas.draw()


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