# Topic Modeling Comparation
Analyzing Indonesian EV news from kompas.com using LDA, NMF, and GSDMM topic modeling. Identifying best method for coherent topic clustering, aiding EV discourse insights.

This research focuses on an Indonesian dataset of news headlines about electric vehicles (EVs) from kompas.com. The goal is to uncover the main EV-related topics. Three topic modeling methods – LDA, NMF, and GSDMM – are fine-tuned and compared to find the best one for clustering topics based on human judgment and coherence. This study provides insights into EV discourse and the effectiveness of these methods in analyzing Indonesian text data.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

If you are interested in how to parameter tune for each topic modeling approach and utilize the content provided, please read and cite our paper:
<a href="https://dl.acm.org/doi/abs/10.1145/3575882.3575905">Nuraisa Novia Hidayati and Anne Parlina. 2023. Performance Comparison of Topic Modeling Algorithms on Indonesian Short Texts. In Proceedings of the 2022 International Conference on Computer, Control, Informatics and Its Applications (IC3INA '22). Association for Computing Machinery, New York, NY, USA, 117–120. https://doi.org/10.1145/3575882.3575905</a>

## Data Scrapping
If you want to find your own news data title, please scrape using the sample code scrapping_kompas.ipynb

## Data Preprocessing

Pre-processing of data which includes the process of:
    • changing the letters in the text to lowercase,
    • omit single characters,
    • Remove symbols and punctuation marks,
    • removing stopwords,
    • tokenization (separating text into word tokens).
    • Stemming (changing word forms into basic words)

   ![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/836660c3-72a8-4f2c-a3d0-69310a8f8802)

Click the "Browse" button to open a new directory window. Then, choose where the.csv file to be handled is located. Then, click one of the "yes" or "no" radio buttons to decide if the stemming process will be done. Then, click the Pre-Process button to start cleaning up the data. When the pre-processing is done, the data will show up in the white text area in the middle.

![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/904f548e-2f32-4b6b-b796-25455fe99d35)

The results of pre-processing will also be instantly saved in a file with the extension.csv. The original text and the pre-processed text will be saved in the same way as shown below:

![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/141f50de-0cb3-46b5-9fe5-0445626bb527)

## LDA Topic modeling 
Furthermore, if you are going to use the LDA method for topic modeling, the first appearance is as follows. Where the procedure is carried out in phases, beginning at the top and working down.

![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/28b32bc3-03ce-4867-8292-ff133d7db440)

The first step in the process is to select a file with the extension.csv in which the previous results were obtained by clicking the Choose button. Then, in the text field underneath the enter number of subjects, enter the number of topics. To begin the modeling process, select Build Model. When the process is completed, the following information will be displayed:

![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/8e168b81-5b74-4c80-8050-8c7ca25ae9fe)

A pop-up window will appear, asking whether you want to store the modeling findings. When saved, each text with its topic group is saved in the folder where the code is run in a format with the.csv extension. Each subject will be saved in a separate csv file, as illustrated below:

![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/ea5e6ac4-dce0-4c0e-8a56-a3fc2a86c701)

The appearance of each file is as shown below:

![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/62fa640b-30ee-46d9-9633-4c058df6a700)

Meanwhile the main window display is as below, there is a pop up with a message stating that the model has been run and was successful, you can click ok. Then the coherence and perplexity values are also displayed under the build model button. where the keywords that make up each topic will be shown in the text area at the bottom of the application's main window

![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/18b5dfc4-8dd2-44f5-87cc-2e2d3645bc84)

To observe the distribution of themes with the constituent terms, click the show visualization button, and a tab will open in your default browser, displaying a visualization similar to the one shown below:

![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/1cf95cc4-5a97-44f6-9d9a-df126334612b)


## NMF Topic Modeling
The first appearance to do topic modeling with the NMF algorithm is as follows. Where the procedure is carried out in phases, beginning at the top and working down.

![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/33682349-01ae-4474-98fa-a8c87cbce0b1)

The first step in the above display is to choose a file with the.csv extension that contains prior pre-processing results by clicking the Select button. Then, in the text field beneath the enter number of subjects, enter the appropriate number of topics. Click construct model to begin the topic modeling process. When the process is finished, the screen will look like this:

![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/a92c7c5c-0d41-4349-8340-4179d046ca2b)

A pop-up window will appear, asking whether you want to store the modeling findings. When saved, each text with its topic group is saved in the folder where the code is run in a format with the.csv extension. Each topic will be saved in a separate csv file, just as the prior topic modeling results were.

Meanwhile, the main window display is as seen below; there is a pop-up notification confirming that the model was run and was successful; you can click OK. The coherence value will then be displayed beneath the build model button. The NMF algorithm does not allow for the calculation of the perplexity value.

![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/134b50be-15f0-4f1a-8ce2-2b42bbcd5dcc)

To observe the distribution of themes with the constituent terms, click the show visualization button, and a new window will open with a visualization similar to the one shown below:

![image](https://github.com/novra/Topic-Modeling-Comparation/assets/22464171/8799a069-9cbd-427f-acee-58af98f62455)

## GSDMM Topic Modeling
Next, if you want to use the GSDMM algorithm to describe topics, the first screen will look like this. Where the process is done in steps, from the beginning to the end.

![image](https://github.com/novra/Topic-Modeling-Comparation-in-Indonesian-Short-Text/assets/22464171/d6665b2c-bdfb-463c-8cc7-a0120553b688)

The process sequence is the same as other modeling, the difference is how GSDMM generates and saves the result. GSDMM saves the result in one CSV extension file, as shown below 

![image](https://github.com/novra/Topic-Modeling-Comparation-in-Indonesian-Short-Text/assets/22464171/26716468-610c-497c-9cd1-537a0d29810e)

Meanwhile, the main window display is as seen below; there is a pop-up notification confirming that the model was run and was successful; you can click OK. The coherence value will then be displayed beneath the build model button. The GSDMM algorithm has a number of documents per subject and topic clusters below that. So, based on the image below, cluster 2 contains 305 documents, cluster 0 has 97 documents, cluster 4 has 400 documents, cluster 1 has 14 documents, and cluster 3 has 294 documents. To improve the subject grouping process, ensure that each cluster has document content and not 0 documents.

![image](https://github.com/novra/Topic-Modeling-Comparation-in-Indonesian-Short-Text/assets/22464171/e83b4d2b-4a54-453c-af32-018d082e9ff6)

The visualization for GSDMM looks like this 

![image](https://github.com/novra/Topic-Modeling-Comparation-in-Indonesian-Short-Text/assets/22464171/89a75e62-df45-4ecc-8949-1960be8639f3)

















