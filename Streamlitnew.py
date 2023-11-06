import streamlit as st
import pandas as pd
import numpy as np


# load data
template_matched = pd.read_parquet('template_matched.parquet')
case_matched = pd.read_parquet('case_matched.parquet')

# Load the model from the file
from joblib import load

model = load('xgb_model_general_10.joblib')


# Load SetenteceTransformer
from sentence_transformers import SentenceTransformer

qa_mpnet_dot = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
qq_mpnet_base = SentenceTransformer('all-mpnet-base-v2')



def case_test_creation(user_input, district_number):

    user_case_test = pd.DataFrame({
        'CorpNo': [district_number],
        'qa_embeddings': [qa_mpnet_dot.encode(user_input)],
        'qq_embeddings': [qq_mpnet_base.encode(user_input)],
        'TemplateId': [np.nan]
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

def popularity_test(templates,cases,case_test,Corp):
    temp = templates[templates.MainCorpNo == Corp].reset_index(drop = True)

    n = len(temp)
    p = len(case_test)
    
    mat = [[0 for _ in range(n)] for __ in range(p)]
    
    temp_list = list(temp.TemplateId)
    
    df = pd.DataFrame( mat, columns = temp_list)
    
    freq_count = cases.TemplateId.value_counts()
    
    for i in range(p):
        for template in temp_list:
            if template in freq_count.keys():
                df.loc[i,template] = freq_count[template]
    return df 

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
    popularity_df = popularity_test(temp,case,case_test,Corp)
    
    n = len(temp)
    m = len(case_test)
    temp_list = list(temp.TemplateId)
    
    mat = [[0 for _ in range(3)] for __ in range(m*n)]
    
    df = pd.DataFrame(mat,columns = ['similarity_score','collaborative_score','popularity_score'])
    
    cnt = 0
    
    for i in range(m):
        for temp in temp_list:
            df.loc[cnt,'similarity_score'] = similarity_df.loc[i,temp]
            df.loc[cnt,'collaborative_score'] = collaborative_df.loc[i,temp]
            df.loc[cnt,'popularity_score'] = popularity_df.loc[i,temp]
            if (not np.isnan(case.TemplateId[i])) and case.TemplateId[i] == temp:
                df.loc[cnt,'match'] = 1
            cnt += 1
    return df


def classification_df_test(templates,cases,case_test,Corp):
    temp = templates[templates.MainCorpNo == Corp].reset_index(drop = True)
    case = cases[cases.CorpNo == Corp].reset_index(drop = True)
    
    similarity_df = compute_similarity_matrix_test(temp, case_test, Corp)
    collaborative_df = collaborative_filtering_test(temp,case,case_test,Corp)
    popularity_df = popularity_test(temp,case,case_test,Corp)
    
    n = len(temp)
    m = len(case_test)
    temp_list = list(temp.TemplateId)
    
    mat = [[0 for _ in range(4)] for __ in range(m*n)]
    
    df = pd.DataFrame(mat,columns = ['similarity_score','collaborative_score','popularity_score','match'])
    
    cnt = 0
    
    for i in range(m):
        for temp in temp_list:
            df.loc[cnt,'similarity_score'] = similarity_df.loc[i,temp]
            df.loc[cnt,'collaborative_score'] = collaborative_df.loc[i,temp]
            df.loc[cnt,'popularity_score'] = popularity_df.loc[i,temp]
            if (not np.isnan(case.TemplateId[i])) and case.TemplateId[i] == temp:
                df.loc[cnt,'match'] = 1
            cnt += 1
    return df


def XGB_SHOW(templates, cases, case_test, Corp, model):

    Corp = float(Corp)
    temp = templates[templates.MainCorpNo == Corp].reset_index(drop=True)

    res_df = classification_df_test(templates, cases, case_test, Corp)
 
    res_df['similarity_score'] = pd.to_numeric(res_df['similarity_score'], errors='coerce')
    res_df['collaborative_score'] = pd.to_numeric(res_df['collaborative_score'], errors='coerce')
    res_df['popularity_score'] = pd.to_numeric(res_df['popularity_score'], errors='coerce')
    X = res_df[['similarity_score', 'collaborative_score', 'popularity_score']]


    model_xgb = model
    
    m = len(case_test)
    n = len(temp)
    temp_list = list(temp.TemplateId)

    probabilities = model_xgb.predict_proba(X)


    for i in range(m):
        if not np.isnan(case_test.loc[i, 'TemplateId']):
            st.write(f'The true template for this case is {case_test.loc[i, "TemplateId"]}')
        cnt = 0
        prob_list = []
        for j in range(n * i, n * i + n):
            prob_list.append([probabilities[j][1], cnt])
            cnt += 1
        sorted_list = sorted(prob_list, key=lambda x: x[0], reverse=True)
        rec_list = sorted_list[:5] if n >= 5 else sorted_list
        
        st.write('Recommended templates are:')


        for j in range(len(rec_list)):
            if rec_list[j][0] >= 0.5:
                st.write(f'Template id: {temp_list[rec_list[j][1]]}')
                st.write(f'Message body of this template is: {temp.loc[rec_list[j][1], "cleaned_MessageBody"]}')

        st.write()



# UI design with a custom title using HTML and Markdown
st.markdown("<h1 style='text-align: center; color: #800020;'>Test Model</h1>", unsafe_allow_html=True)

# Web app title
st.title("Model Performance Test Web Application")

# Introduction or description
st.write("This web application is designed to test the performance of our model. "
         "Please enter a district number and some text to get template recommendations.")

# User input fields
options = [11918, 12313, 12488, 12598, 13293, 13454, 13488, 13489, 13956, 13957,
14434, 14438, 14695, 14722, 15576, 16037, 16082, 16290, 16594, 16913,
17000, 19030, 19106, 20011, 20119, 20120, 20257, 24287, 24936, 25047,
25077, 28264, 29038, 30930, 31460, 32086, 32114, 32236, 32998, 34084,
34084,34122,35258,35326,36992,37470, 37475, 37656, 37988, 38115, 40332, 
41272, 41380, 43375, 43417, 43426,43436, 43477, 43525, 44070, 44744, 44884, 45671, 
45777, 46543, 46769,46835, 47139, 47607, 47898, 48928, 49871, 50118, 50543, 50566, 51214,
51989, 52028, 52069, 52117, 52639, 52986, 53280, 53916, 53960, 54135,
54303, 54384, 55041, 55535, 55574, 56129, 56313, 56418, 56609, 56658,
56750, 56771, 57150, 57159, 58018, 58761, 65095, 65141, 65153, 65282,
65283, 65401, 65453, 65469, 68712, 69866, 70337, 70551, 71244, 71419,
71462, 71544, 78756, 78941, 79714, 79806, 80201, 81564, 81588, 82174,
82175, 83057, 83087, 83163, 83236, 83499, 83602, 83648, 83737, 83743,
83769, 84947, 85044, 85799, 85817, 86483, 87168, 88014, 88045, 88310,
88356, 88375, 92414, 92490, 92953, 93251, 93837, 93845, 93846, 94295,
94396, 94471, 95133, 95325, 95389, 95402, 95776, 95937, 96333, 96503,
98991, 99258, 99821, 100498, 100640, 100668, 101178, 101410, 102339, 102489]
district_number = st.selectbox("Select a CorpNo", options)
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