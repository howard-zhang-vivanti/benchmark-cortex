import streamlit as st
import pandas as pd
import os 
import numpy
import json

# import snowflake.snowpark as snowpark
from snowflake.snowpark.session import Session

# default run streamlit command: streamlit run streamlit_app.py --server.enableCORS false --server.enableXsrfProtection false

connection_parameters = {
  "account": st.secrets["SNOWFLAKE_ACCOUNT"],
  "user": st.secrets["SNOWFLAKE_USER_NAME"],
  "password": st.secrets["SNOWFLAKE_PASSWORD"],
  "role": st.secrets["SNOWFLAKE_ROLE"],
  "warehouse": "COMPUTE_WH",
  "database": "BENCHMARK_LANDING",
  "schema": "PUBLIC"
}
sf_session = Session.builder.configs(connection_parameters).create()

st.set_page_config(
     page_title='Benchmark Chatbot',
     layout='wide',
     initial_sidebar_state='expanded')

# session = get_active_session() # Get the current credentials
pd.set_option("max_colwidth",None)
num_chunks = 3 # Num-chunks provided as context. Play with this to check how it affects your accuracy
SYSTEM_SLIDE_WINDOW = 100 # how many last conversations to remember. This is the slide window.
SYSTEM_PROMPT = f"""
        You are an expert data warehouse assistance from a consultancy called Vivanti. 
        You offer a chat experience. Read all the input before producing the output. 
        Only provide the answer base on the knowledge provide to you, be concise and do not hallucinate. 
        If you don´t have the information just say so. 
        Provide the answer in a consistent style."""
SYSTEM_FILE_NAME = 'Snowflake Benchmark Study - Shared with Snowflake.pptx'
SYSTEM_COMPLETE_OPTIONS = (
    json.dumps({
                'temperature': 0,
                'max_tokens': 128000, 
                'guardrails' : True
            }, indent=3)
    .replace('\'', '\\\'')
    .replace('"', '\'')
)

def main():
    st.title("Benchmark Report Chatbot")
    st.write("""You can ask questions about our benchmark reprt.""")

    config_options()
    reset_chat()

    prompt_1st = create_initial_prompt(SYSTEM_FILE_NAME)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if question := st.chat_input("Ask me anything about the benchmark data: "):
        # Add user message to chat history
        st.session_state.messages.append({'role': 'user', 'content': question})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(question)
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            question = question.replace("'","")
    
            with st.spinner(f"{st.session_state.model_name} is thinking..."):
                response = complete(question, prompt_1st)  
                message_placeholder.markdown(response)
        
        st.session_state.messages.append({'role': 'assistant', 'content': response})

        
# combines the LLM, the prompt template and whether to use the context or not to generate a 
# response which includes a link to the asset from which the answer was obtained.
def complete(question, prompt_1st):
        
    # prompts doesn't contains all the chat history, only first 2 and last SYSTEM_SLIDE_WINDOW value
    prompts = get_chat_history(prompt_1st)
    # no summary or rag for now 
    # question_summary = summarize_question_with_history(chat_history, question)
    # prompt_context = get_similar_chunks(question_summary)
    
    # append user question in into the message 
    prompts.append({'role': 'user', 'content': question})

    prompts_str = (
        json.dumps(prompts, indent=3)
        .replace('\'', '\\\'')
        .replace('"', '\'')
    )
    
    sql = f"""
            select snowflake.cortex.complete('{st.session_state.model_name}', {prompts_str}, {SYSTEM_COMPLETE_OPTIONS}):"choices"[0]:"messages"::string as response
          """
    # st.write(sql)
    response = sf_session.sql(sql).collect()[0].as_dict()["RESPONSE"].replace("'", "")

    return response

    
def config_options():
    #Here you can choose what LLM to use. Please note that they will have different cost & performance
    st.sidebar.selectbox('Select your model:', (
                                        # 'llama3.1-70b',
                                        # 'mixtral-8x7b',
                                        'mistral-large2'
                                        ,'llama3.1-405b'), 
                         key="model_name")
                                           
    # For educational purposes. Users can chech the difference when using memory or not
    st.sidebar.button("Start Over", key="clear_conversation")
    st.sidebar.expander("Session State").write(st.session_state)


def reset_chat():
    # Initialize chat history
    if st.session_state.clear_conversation or "messages" not in st.session_state:
        st.session_state.messages = []        


# Get the history from the st.session_stage.messages
# it keep the first 2 messages (system and the initial prompt)
# then get the last 50 rolling messages 
def get_chat_history(prompt_1st):    
    chat_history = []
    # append 2 dict in, don't do [{1}, {2}], you get a nest array of one element of another array
    chat_history.append({'role': 'system', 'content': SYSTEM_PROMPT})
    chat_history.append({'role': 'user', 'content': prompt_1st})
    chat_history.append({'role': 'assistant', 'content': ''})
    
    # load the last SYSTEM_SLIDE_WINDOW number of messages 
    start_index = max(0, len(st.session_state.messages) - SYSTEM_SLIDE_WINDOW)
    for i in range(start_index , len(st.session_state.messages) -1):
         chat_history.append(st.session_state.messages[i])

    return chat_history


def display_response_rag (question, model):
    response, url_link, relative_path = complete(question, model)
    res_text = response[0].RESPONSE
    st.markdown(res_text)

    display_url = f"Link to [{relative_path}]({url_link}) that may be useful"
    st.markdown(display_url)


# get_similar_chunks() receives a question as an argument, use the vector similarity search 
# to get the cloest 3 chunks with the question
def get_similar_chunks_rag(myquestion):
  
    # take the quesiton and convert it to embedding then find the ANN
    sql = """
         with results as
         (SELECT RELATIVE_PATH,
           VECTOR_COSINE_SIMILARITY(benchmark_reporting.benchmark_cortex.DOCS_CHUNKS_TABLE.chunk_vec,
                    SNOWFLAKE.CORTEX.EMBED_TEXT_1024('multilingual-e5-large', ?)) as similarity,
           chunk
         from benchmark_reporting.benchmark_cortex.DOCS_CHUNKS_TABLE 
         order by similarity desc
         limit ?)
         select chunk, relative_path from results 
         """
        
    # run the sql, pass the parameters in 
    df_context = sf_session.sql(sql, params=[myquestion, num_chunks]).to_pandas()      
    context_lenght = len(df_context) -1

    prompt_context = ""
    for i in range (0, context_lenght):
        prompt_context += df_context._get_value(i, 'CHUNK')

    prompt_context = prompt_context.replace("'", "")

    return prompt_context

    
# To get the right context, use the LLM to first summarize the previous conversation
# This will be used to get embeddings and find similar chunks in the docs for context
def summarize_question_with_history(chat_history, question):
    
    prompt = f"""
        Based on the chat history below and the question, generate a query that extend the question
        with the chat history provided. The query should be in natual language. 
        Answer with only the query. Do not add any explanation.
        
        <chat_history>
        {chat_history}
        </chat_history>
        <question>
        {question}
        </question>
        """
    
    cmd = """
            select snowflake.cortex.complete(?, ?) as response
          """
    df_response = sf_session.sql(cmd, params=[st.session_state.model_name, prompt]).collect()
    sumary = df_response[0].RESPONSE     
    sumary = sumary.replace("'", "")

    st.sidebar.expander("Summary").write(sumary)

    return sumary


# read the entire doco 
def create_initial_prompt(file_name): 
    sql = """
        select LISTAGG('<Slide Number>: ' || page_number || '\n' || chunk) as ALL_CONTENTS
        from benchmark_reporting.benchmark_cortex.PPTX_CHUNKS
        where file_name = ?
        group by FILE_NAME
    """
    
    df_context = sf_session.sql(sql, params=[file_name]).collect()
    all_contents = df_context[0][0]

    init_prompt = f"""
        ############# 
        # CONTEXT # 
        I work for Vivanti, we an IT consultant company
        We have just produced a benchmark report on these four databases: Snowflake, Databricks, Bigquery and Redshift
        The report is in the pptx layout
        Now I have created a streamlit app with LLM to answer any questions related to this report 
        
        ############# 
        # OBJECTIVE # 
        - Each slide is indicated by '<Slide Number>' tag
        - Read and understand the entire document between <doc> </doc> tags 
        - User will ask questions about the document perform the following steps: 
            -- Convert user's question into a more refined and strucured question 
            -- Create answer base on the refind question 
            -- Refine your answer 3 times 
            -- Provide the answer

        ############# 
        # STYLE # 
        Australian English 

        ############# 
        # TONE # 
        Professional, Technical

        ############# 
        # AUDIENCE # 
        People who interested in the cloud data warehouse 

        ############# 
        # RESPONSE # 
        Rendered markdown style 
        
        ############# 
        the report you need to analysis is: 
        <doc>
        {all_contents}
        </doc>
    """

    return init_prompt
    
if __name__ == "__main__":
    main()

