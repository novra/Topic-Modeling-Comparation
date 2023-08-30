# Topic-Modeling-Comparation
Analyzing Indonesian EV news from kompas.com using LDA, NMF, and GSDMM topic modeling. Identifying best method for coherent topic clustering, aiding EV discourse insights.

This research focuses on an Indonesian dataset of news headlines about electric vehicles (EVs) from kompas.com. The goal is to uncover the main EV-related topics. Three topic modeling methods – LDA, NMF, and GSDMM – are fine-tuned and compared to find the best one for clustering topics based on human judgment and coherence. This study provides insights into EV discourse and the effectiveness of these methods in analyzing Indonesian text data.

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

If you are interested in how to parameter tune for each topic modeling approach and utilize the content provided, please read and cite our paper:
<a href="https://dl.acm.org/doi/abs/10.1145/3575882.3575905">Nuraisa Novia Hidayati and Anne Parlina. 2023. Performance Comparison of Topic Modeling Algorithms on Indonesian Short Texts. In Proceedings of the 2022 International Conference on Computer, Control, Informatics and Its Applications (IC3INA '22). Association for Computing Machinery, New York, NY, USA, 117–120. https://doi.org/10.1145/3575882.3575905</a>

## Data Scrapping
If you want to find your own title, please scrape using the sample code scrapping_kompas.ipynb

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









