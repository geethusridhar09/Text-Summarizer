import os
import streamlit as st
import torch
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.cluster.util import cosine_distance
import networkx as nx
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration


#Checking if gpu is available otherwise using the cpu

if torch.cuda.is_available():
   device = torch.device("cuda")
else:
   device = torch.device("cpu")

#Loading the pegasus and bart model. Using cache to store the loaded data so that it doesn't execute  after every runtime.

@st.cache(allow_output_mutation=True)
def load_pegasus_model():
    model_name = "google/pegasus-xsum"
    summarizer = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    return summarizer, tokenizer

@st.cache(allow_output_mutation=True)
def load_bart_model():
    model_name = "facebook/bart-large-cnn"
    summarizer = BartForConditionalGeneration.from_pretrained(model_name).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    return summarizer, tokenizer

# EXTRACTIVE SUMMARY

#Cleaning the given text so that it gives better output for the extractive summarisation

def clean_text(text):
  article = text.split(".")
  article=[sentence for sentence in article if sentence!=""]
  # print(article)

  sentences = []

  for sentence in article:
      #print(sentence)
      sentence=sentence.replace(",", " , ").replace("'", " ' ").split(" ")
      #sentence=sentence.replace("[^a-zA-Z]", " ").split(" ")
      sentence=[word for word in sentence if word!=""]
      sentences.append(sentence)
    
  return sentences

def sentence_similarity(sent1, sent2, stopwords):   #Creating words in sentences to one hot encoding and then finding cosine distance between the vectors inorder to measure closeness
  
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences, stop_words):
  
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(text, top_n='2'):

    if top_n==  None or top_n=="":
      top_n=2
    top_n=int(top_n)
    # Step 1 - Clean text to generate sentences

    sentences=clean_text(text)
    stop_words = stopwords.words('english')
    stop_words.append(".")
    stop_words.append(",")
    summarize_text = []

    # Step 2 - Generate Similary Martix across sentences

    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    # print(sentence_similarity_martix)

    # Step 3 - Rank sentences in similarity martix

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    # print(sentence_similarity_graph)

    scores = nx.pagerank(sentence_similarity_graph)
    # print(scores)

    # Step 4 - Sort the rank and pick top sentences

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    #Sorting the scores in decending order
    # print("Indexes of top ranked_sentence order are ", ranked_sentence)   

    for i in range(top_n):
      ranked_sentence[i][1][0]=ranked_sentence[i][1][0].capitalize()    #Capitalising 1st letter of sentence
      # print(ranked_sentence[i][1][0]) 
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarized text

    extractive_summarized=". ".join(summarize_text).replace(" , ",", ").replace(" ' ","'") + "."
    return extractive_summarized



#ABSTRACTIVE SUMMARY

#Converting sentence into tokens and then extracting the output from the tokens

def run_model(model,input_text,min_length=30,max_length=128,num_return_sequences = 1):
    if model == "Bart":
        bart_model,bart_tokenizer=load_bart_model()
        input_text = ' '.join(input_text.split())
        input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt').to(device)
        summary_ids = bart_model.generate(input_tokenized,
                                    num_beams = 4,
                                    num_return_sequences = num_return_sequences,
                                    no_repeat_ngram_size = 2,
                                    length_penalty = 1,
                                    min_length = min_length,
                                    max_length = max_length,
                                    early_stopping = True)
    
        output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
        st.write('Summary')
        st.success(output)
    else:
        #pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)
        #pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        pegasus_model,pegasus_tokenizer=load_pegasus_model()
        input_text = ' '.join(input_text.split())
        batch = pegasus_tokenizer.prepare_seq2seq_batch(input_text, truncation=True, padding='longest', return_tensors="pt").to(device)
        
        summary_ids = pegasus_model.generate(**batch,
                                            num_beams=10,
                                            num_return_sequences=num_return_sequences,
                                            no_repeat_ngram_size = 2,
                                            length_penalty = 1,
                                            min_length = min_length,
                                            max_length = max_length,
                                            early_stopping = True)
        
        output = [pegasus_tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)][0][0]
        st.write("Summary")
        st.success(output)

def main():
  #text = """In an attempt to build an AI-ready workforce, Microsoft announced Intelligent Cloud Hub which has been launched to empower the next generation of students with AI-ready skills. Envisioned as a three-year collaborative program, Intelligent Cloud Hub will support around 100 institutions with AI infrastructure, course content and curriculum, developer support, development tools and give students access to cloud and AI services. As part of the program, the Redmond giant which wants to expand its reach and is planning to build a strong developer ecosystem in India with the program will set up the core AI infrastructure and IoT Hub for the selected campuses. The company will provide AI development tools and Azure AI services such as Microsoft Cognitive Services, Bot Services and Azure Machine Learning.According to Manish Prakash, Country General Manager-PS, Health and Education, Microsoft India, said, "With AI being the defining technology of our time, it is transforming lives and industry and the jobs of tomorrow will require a different skillset. This will require more collaborations and training and working with AI. That’s why it has become more critical than ever for educational institutions to integrate new cloud and AI technologies. The program is an attempt to ramp up the institutional set-up and build capabilities among the educators to educate the workforce of tomorrow." The program aims to build up the cognitive skills and in-depth understanding of developing intelligent cloud connected solutions for applications across industry. Earlier in April this year, the company announced Microsoft Professional Program In AI as a learning track open to the public. The program was developed to provide job ready skills to programmers who wanted to hone their skills in AI and data science with a series of online courses which featured hands-on labs and expert instructors as well. This program also included developer-focused AI school that provided a bunch of assets to help build AI skills."""

  st.title('Text Summarizer')
  text=st.text_input("Enter Text")
  
  

  extractive_summary=""
  abstractive_summary=""

  Summary = st.selectbox('Select Summary', ["Extractive Summary","Abstractive Summary"],key="Summary")
  if Summary=="Extractive Summary":
    with st.form("my_form"):
      st.write("Extractive Summary")
      no_of_sentences=st.text_input("Enter no of sentences to be summarised in (for extractive mode)",placeholder="Default is 2")
      submit_button = st.form_submit_button("Submit")
    if submit_button:
      extractive_summary=generate_summary(text, no_of_sentences)
      st.success(extractive_summary)
  else:
    with st.form("my_form2"):
      st.write("Abstractive Summary")
      model = st.selectbox('Model for abstractive Summary', ["Bart","Pegasus"])
      min_length = st.slider('minimum length of summary(Words)', 5, 70, 30,1)
      max_length = st.slider('maximum length of summary(Words)', 70, 150, 128,1)
      #num_return_sequences= st.slider('No of summaries to return', 1, 10, 1,1)
      submit_button = st.form_submit_button("Submit")
    if submit_button:
      run_model(model,text,min_length,max_length)





if __name__== '__main__':
  main()