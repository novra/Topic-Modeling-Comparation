import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from gensim import corpora
from gensim.models import TfidfModel, LdaModel, CoherenceModel
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import tempfile
from tkinter import ttk
from tkhtmlview import HTMLLabel
import webbrowser
import html
import os
import subprocess
import sys

class App:
    def __init__(self, master):
        self.master = master
        master.title("LDA Model")

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

        self.build_button = tk.Button(master, text="Build Model", command=self.build_lda)
        self.build_button.pack()

        self.coherence_label = tk.Label(master, text="Coherence: ")
        self.coherence_label.pack()

        self.perplexity_label = tk.Label(master, text="Perplexity: ")
        self.perplexity_label.pack()

        self.tmp = None
        self.lda_model = None
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

        
    def build_lda(self):
        

        texts = [[word for word in document.lower().split()] for document in self.judul_prep]

        dictionary = corpora.Dictionary(texts)
        corpus_bow = [dictionary.doc2bow(text) for text in texts]
        tfidf = TfidfModel(corpus_bow)
        corpus_tfidf = tfidf[corpus_bow]
        num_topics = int(self.num_topics_entry.get())
        lda_model = LdaModel(corpus=corpus_tfidf,
                                           id2word=dictionary,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           per_word_topics=True)
        coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus_tfidf, dictionary=dictionary, coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()

        perplexity = lda_model.log_perplexity(corpus_tfidf)

        self.coherence_label.config(text=f"Coherence: {coherence_lda}")
        self.perplexity_label.config(text=f"Perplexity: {perplexity}")

        self.lda_model = lda_model
        self.dictionary = dictionary
        self.corpus = corpus_tfidf

    # Display the top topics and their related keywords
        topics = lda_model.show_topics(num_topics=num_topics, num_words=10)
        topics_str = ""
        for topic in topics:
            topic_str = f"Topic {topic[0]}: "
            topic_str += ", ".join([word[0] for word in lda_model.show_topic(topic[0], topn=10)])
            topic_str += "\n"
            topics_str += topic_str
          
        save_results = messagebox.askyesno("Save Results", "Do you want to save the results to a CSV file?")
        if save_results:
            for topic in topics:
                # Get the topic assignment for each document in the corpus
                topic_assignment = lda_model.get_document_topics(corpus_tfidf)
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
                filename = f"topic_LDA_{topic[0]}.csv"
                df_topic.to_csv(filename, index=False)

        self.result_textbox.delete("1.0", tk.END) # clear previous results
        self.result_textbox.insert(tk.END, topics_str)
        
        messagebox.showinfo("LDA Model", "Model built successfully!")

    # def show_vis(self):
    #    if not hasattr(self, 'lda_model') or not hasattr(self, 'corpus') or not hasattr(self, 'dictionary'):
    #         messagebox.showerror("Error", "Please build a model first.")
    #    else:
    #         vis = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)
    #         pyLDAvis.enable_notebook()
    #         prepared_data = pyLDAvis.prepared_data_to_html(vis)
    #         with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
    #             f.write(prepared_data)
    #             webbrowser.open(f.name)


    # def show_vis(self):
    #     if not hasattr(self, 'lda_model') or not hasattr(self, 'corpus') or not hasattr(self, 'dictionary'):
    #         messagebox.showerror("Error", "Please build a model first.")
    #     else:
    #         vis = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)

    #         # Instead of using pyLDAvis.enable_notebook(), we will save the prepared data to a file
    #         prepared_data = pyLDAvis.prepared_data_to_html(vis)

    #         # Create a temporary HTML file to store the visualization
    #         tmp_html_file = os.path.join(tempfile.gettempdir(), "lda_visualization.html")
    #         with open(tmp_html_file, 'w', encoding='utf-8') as f:
    #             f.write(prepared_data)

    #         # Open the temporary HTML file in the default web browser
    #         webbrowser.open_new_tab(tmp_html_file)

    # def show_vis(self):
    #     if not hasattr(self, 'lda_model') or not hasattr(self, 'corpus') or not hasattr(self, 'dictionary'):
    #         messagebox.showerror("Error", "Please build a model first.")
    #     else:
    #         vis = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)

    #         # Instead of using pyLDAvis.enable_notebook(), we will save the prepared data to a file
    #         prepared_data = pyLDAvis.prepared_data_to_html(vis)

    #         # Create a temporary HTML file to store the visualization
    #         tmp_html_file = os.path.join(tempfile.gettempdir(), "lda_visualization.html")
    #         with open(tmp_html_file, 'w', encoding='utf-8') as f:
    #             f.write(prepared_data)

    #         # Open the temporary HTML file in the default web browser
    #         #webbrowser.open_new_tab(tmp_html_file)
    #         webbrowser.open(f"file:///{os.path.abspath(tmp_html_file)}")

    # def show_vis(self):
    #     if not hasattr(self, 'lda_model') or not hasattr(self, 'corpus') or not hasattr(self, 'dictionary'):
    #         messagebox.showerror("Error", "Please build a model first.")
    #     else:
    #         vis = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)

    #         # Instead of using pyLDAvis.enable_notebook(), we will save the prepared data to a file
    #         prepared_data = pyLDAvis.prepared_data_to_html(vis)

    #         # Create a temporary HTML file to store the visualization
    #         tmp_html_file = os.path.join(tempfile.gettempdir(), "lda_visualization.html")
    #         with open(tmp_html_file, 'w', encoding='utf-8') as f:
    #             f.write(prepared_data)

    #         # Open the temporary HTML file in the default system web browser
    #         try:
    #             subprocess.Popen(['start', tmp_html_file], shell=True)
    #         except Exception as e:
    #             messagebox.showerror("Error", f"Failed to open the web browser: {e}")

    def show_vis(self):
        if not hasattr(self, 'lda_model') or not hasattr(self, 'corpus') or not hasattr(self, 'dictionary'):
            messagebox.showerror("Error", "Please build a model first.")
        else:
            vis = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary)

            # Instead of using pyLDAvis.enable_notebook(), we will save the prepared data to a file
            prepared_data = pyLDAvis.prepared_data_to_html(vis)

            # Create a temporary HTML file to store the visualization
            tmp_html_file = os.path.join(tempfile.gettempdir(), "lda_visualization.html")
            with open(tmp_html_file, 'w', encoding='utf-8') as f:
                f.write(prepared_data)

            # Open the temporary HTML file in the default web browser
            try:
                # Use platform-specific method to open the web browser
                if sys.platform == 'win32':
                    # On Windows, use the 'start' command to open the file in the default web browser
                    os.startfile(tmp_html_file)
                elif sys.platform == 'darwin':
                    # On macOS, use the 'open' command to open the file in the default web browser
                    subprocess.call(['open', tmp_html_file])
                else:
                    # On Linux, use the 'xdg-open' command to open the file in the default web browser
                    subprocess.call(['xdg-open', tmp_html_file])
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open the web browser: {e}")

    def save_model(self):
        try:
            filename = filedialog.asksaveasfilename(defaultextension=".model", filetypes=(("LDA Model", "*.model"),))
            self.lda_model.save(filename)
            messagebox.showinfo("Save Model", "Model saved successfully!")
        except AttributeError:
            messagebox.showerror("Error", "Please build a model first.")

root = tk.Tk()
app = App(root)
root.mainloop()