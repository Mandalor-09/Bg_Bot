from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from helper import *
from langchain_core.prompts import PromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
import pandas as pd
from transformers import pipeline

df = pd.read_csv('dataset/cleaned_data.csv')
df = df.iloc[:, 1:-2]
tokenizer,model = loading_model_and_tokenizer('models/model', 'models/adapter')
llm = pipeline(model=model,tokenizer=tokenizer,task="text-generation")

class LLMRunnableWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, prompt, **kwargs):
        return self.pipeline(prompt, **kwargs)[0]['generated_text']

def reset_session(selected_product):
    st.session_state.messages = [{
        'role': 'assistant',
        'content': f'Hello! How may I help you with Product {selected_product}?'
    }]
    st.session_state.product_details = {'product_info': selected_product}

def main():
    # Get list of product names
    product_names = product_name_list(df)

    # Get product selection from sidebar
    selected_product = st.selectbox('Select a Product', product_names)

    # Initialize session state (if not already present)
    if 'messages' not in st.session_state:
        reset_session(selected_product)

    if 'product_details' not in st.session_state:
        st.session_state.product_details = {'product_info': ''}

    # Update assistant message based on selected product
    if selected_product is not None:
        st.session_state.messages[0]['content'] = f'Hello! How may I help you with Product {selected_product}?'

    # Display subheader based on selected product
    st.subheader(f'You have queries related to {selected_product}')

    # Get product details based on selected product
    product_details = get_product_details(df, selected_product)

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        template  = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(
                    f"Help user to solve is query just generate response {selected_product}"
                ),
                # The `variable_name` here is what must align with memory
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )    
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        #llm_wrapper = LLMRunnableWrapper(llm)

        conversation = LLMChain(llm=llm, prompt=template, verbose=True, memory=memory)
        final_response = conversation({"question": f"{prompt}"})
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(final_response)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": final_response})

        # Reset session if product selection changes
        if selected_product != st.session_state.product_details['product_info']:
            reset_session(selected_product)

if __name__ == '__main__':
    main()