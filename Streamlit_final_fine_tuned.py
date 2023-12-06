import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import base64


# load data
template_matched = pd.read_parquet('fine_tuned_template_matched.parquet')
case_matched = pd.read_parquet('fine_tuned_case_matched.parquet')

# Load the model from the file
from joblib import load

model = load('xgb_finetune.joblib')


# Load SetenteceTransformer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
import torch

# qa_mpnet_dot = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
qq_mpnet_base = SentenceTransformer('all-mpnet-base-v2')

tokenizer = AutoTokenizer.from_pretrained('./Fine_Tuned_Transformer_1_10')
qa_mpnet_dot = AutoModelForSequenceClassification.from_pretrained('./Fine_Tuned_Transformer_1_10')


def generate_embeds(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = inputs

    # Get the last hidden states
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = qa_mpnet_dot(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]

    sentence_embedding = last_hidden_states.mean(dim=1).cpu().numpy()[0]
    
    return sentence_embedding 


def case_test_creation(user_input, district_number):

    qa_embeddings = generate_embeds(user_input)

    user_case_test = pd.DataFrame({
        'CorpNo': [district_number],
        'qa_embeddings': [qa_embeddings],
        'qq_embeddings': [qq_mpnet_base.encode(user_input)],
        'TemplateId': [np.nan],
        'cleaned_description': [user_input]
    })

    return user_case_test


def collaborative_filtering_test(templates, cases, case_test,Corp):
    temp = templates[templates.MainCorpNo == Corp].reset_index(drop = True)
    case = cases[cases.CorpNo == Corp].reset_index(drop = True)
    n = len(temp)
    m = len(case)
    p = len(case_test)
    
    mat = [[0 for _ in range(n)] for __ in range(p)]
    
    temp_list = list(temp.TemplateId)
    
    df = pd.DataFrame( mat, columns = temp_list)
    
    for i in range(p):
        similarities = []
        for j in range(m):
            similarity = np.dot(case.qq_embeddings[i],case.qq_embeddings[j].T).item()
            similarities.append((j,similarity))
        similarities.sort(key = lambda X:X[1],reverse = True)
        similarities = similarities[:10]
        for pair in similarities:
            if not np.isnan(case.TemplateId[pair[0]]):
                df.loc[i,case.TemplateId[pair[0]]] += pair[1]
    
    
    return df

def jaccard_similarity_sentence(sentence1, sentence2):
    # Tokenize the sentences into words
    words1 = set(sentence1.split())
    words2 = set(sentence2.split())

    # Compute the intersection and union
    intersection = words1.intersection(words2)
    union = words1.union(words2)

    # Calculate Jaccard Similarity
    similarity = len(intersection) / len(union)
    
    return similarity

def jaccard_matrix(templates, case_test, Corp):
    temp = templates[templates.MainCorpNo == Corp].reset_index(drop = True)
    case = case_test[case_test.CorpNo == Corp].reset_index(drop = True)
    
    n = len(temp)
    m = len(case)
    
    mat = [[0 for _ in range(n)] for __ in range(m)]
    
    temp_list = list(temp.TemplateId)
    
    similarity_matrix = pd.DataFrame( mat, columns = temp_list)
    for case_id, case_body in zip(case.index, case['cleaned_description']):
        for template_id, template_body in zip(temp['TemplateId'], temp['cleaned_MessageBody']):
            similarity = jaccard_similarity_sentence(case_body, template_body)
            similarity_matrix.loc[case_id, template_id] = similarity
    return similarity_matrix

def compute_similarity_matrix_test(templates, case_test, Corp):

    temp = templates[templates.MainCorpNo == Corp].reset_index(drop = True)
    
    n = len(temp)
    m = len(case_test)
    
    mat = [[0 for _ in range(n)] for __ in range(m)]
    
    temp_list = list(temp.TemplateId)
    
    similarity_matrix = pd.DataFrame( mat, columns = temp_list)
    
    for case_id, case_embedding in zip(case_test.index, case_test['qa_embeddings']):
        for template_id, template_embedding in zip(temp['TemplateId'], temp['embeddings']):
            similarity = np.dot(case_embedding, template_embedding.T).item()
            similarity_matrix.loc[case_id, template_id] = similarity
    return similarity_matrix


def classification_df_test(templates,cases,case_test,Corp):
    temp = templates[templates.MainCorpNo == Corp].reset_index(drop = True)
    case = cases[cases.CorpNo == Corp].reset_index(drop = True)
    
    similarity_df = compute_similarity_matrix_test(temp, case_test, Corp)
    collaborative_df = collaborative_filtering_test(temp,case,case_test,Corp)
    jaccard_df = jaccard_matrix(temp,case_test,Corp)
    
    n = len(temp)
    m = len(case_test)
    temp_list = list(temp.TemplateId)
    
    mat = [[0 for _ in range(3)] for __ in range(m*n)]
    
    df = pd.DataFrame(mat,columns = ['similarity_score','collaborative_score','jaccard_score'])
    
    cnt = 0
    
    for i in range(m):
        for temp in temp_list:
            df.loc[cnt,'similarity_score'] = similarity_df.loc[i,temp]
            df.loc[cnt,'collaborative_score'] = collaborative_df.loc[i,temp]
            df.loc[cnt,'jaccard_score'] = jaccard_df.loc[i,temp]
            cnt += 1
    return df




def XGB_SHOW(templates, cases, case_test, Corp, model):

    Corp = float(Corp)
    temp = templates[templates.MainCorpNo == Corp].reset_index(drop=True)

    res_df = classification_df_test(templates, cases, case_test, Corp)
 
    res_df['similarity_score'] = pd.to_numeric(res_df['similarity_score'], errors='coerce')
    res_df['collaborative_score'] = pd.to_numeric(res_df['collaborative_score'], errors='coerce')
    res_df['jaccard_score'] = pd.to_numeric(res_df['jaccard_score'], errors='coerce')
    X = res_df[['similarity_score', 'collaborative_score', 'jaccard_score']]


    model_xgb = model
    
    m = len(case_test)
    n = len(temp)
    temp_list = list(temp.TemplateId)

    probabilities = model_xgb.predict_proba(X)

    for i in range(m):
        cnt = 0
        prob_list = []
        for j in range(n * i, n * i + n):
            prob_list.append([probabilities[j][1], cnt])
            cnt += 1
        sorted_list = sorted(prob_list, key=lambda x: x[0], reverse=True)
        rec_list = sorted_list[:5] if n >= 5 else sorted_list
        
    try:

        temp_res = []
        id_res = []

        for j in range(len(rec_list)):
            if rec_list[j][0] >= 0.5:
                id_res.append(temp_list[rec_list[j][1]])
                temp_res.append(temp.loc[rec_list[j][1], "MessageBody"])

        if len(temp_res) > 0:
            st.subheader('Here are some recommended templates!')
        for idx, tab in enumerate(st.tabs([str(id) for id in id_res])):
            with tab:
                components.html(temp_res[idx])

    except:
        st.subheader('No Relevant Templates!')

    st.write()



############################### Streamlit ###############################
def set_bg(image_file):
    with open(image_file, "rb") as file:
        base64_img = base64.b64encode(file.read()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_img}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(layout="wide")

# Set the background
set_bg('background.jpg')
    
# center the image
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.image("k12-lets-talk.jpg")
with col3:
    st.write(' ')
                
# Introduction or description
st.caption("This web application is designed to test the performance of our model. "
         "Please enter a district number and some text to get template recommendations.")

# User input fields
options = [11918, 12313, 12488, 12598, 13293, 13454, 13488, 13489, 13956, 13957,
    14434, 14438, 14695, 16037, 16082, 17000, 19030, 19106, 20257, 24287,
    24936, 25077, 29038, 30930, 31460, 32114, 32236, 32998, 34084, 35326,
    37475, 37656, 37988, 40332, 41272, 41380, 43417, 43436, 43477, 43525,
    44070, 44744, 44884, 45671, 45777, 46543, 46769, 47139, 47898, 48928,
    49871, 50543, 50566, 51214, 51989, 52069, 52117, 52639, 53280, 53916,
    53960, 55041, 55535, 55574, 56129, 56313, 56609, 56658, 56750, 57150,
    58018, 58761, 65095, 65153, 65282, 65401, 65469, 68712, 69866, 71462,
    79806, 80201, 82174, 82175, 83236, 83602, 83648, 83743, 83769, 85799,
    85817, 87168, 88014, 88045, 88375, 93251, 93846, 95325, 95776, 95937,
    96333, 96503, 100498]
district_number = st.selectbox("Select a school district", options)
user_input = st.text_area("Enter your text here")




# Submit button
if st.button("Submit"):
    if district_number and user_input:

        
        # create case_test
        user_case_test = case_test_creation(user_input, district_number)
        
        # Call function to get recommendations
        XGB_SHOW(template_matched, case_matched, user_case_test, district_number, model)

    else:
        st.error("Please enter both a district number and some text.")