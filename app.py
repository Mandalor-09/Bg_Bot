import streamlit as st
import pandas as pd

# Assuming you have defined these helper functions
from helper import (
    product_name_list,
    get_product_details,
    input_prompt,
    get_response_from_model,
    filtering_response,
    filter_final_response,
    initialize_model,
    loading_model_and_tokenizer
)

df = pd.read_csv('dataset/cleaned_data.csv')
df = df.iloc[:, 1:-2]

# Initialize global model and tokenizer (if needed)
#global_model = None
#global_tokenizer = None

#if global_model is None or global_tokenizer is None:
global_tokenizer, global_model = loading_model_and_tokenizer('models/model', 'models/adapter')

# Function to reset session state
def reset_session(selected_product):
    st.session_state.messages = [{
        'role': 'assistant',
        'content': f'Hello! How may I help you with Product {selected_product}?'
    }]
    st.session_state.product_details = {'product_info': selected_product}

# Main function for Streamlit app
def main():
    # Initialize session state (if not already present)
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'product_details' not in st.session_state:
        st.session_state.product_details = {'product_info': ''}

    # Get list of product names
    product_names = product_name_list(df)

    # Get product selection from sidebar
    selected_product = st.selectbox('Select a Product', product_names)

    # Check for changes in selected product
    if selected_product != st.session_state.product_details['product_info']:
        reset_session(selected_product)

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

        # Generate and process assistant response
        input_text = input_prompt(product_details, prompt)
        response = get_response_from_model(global_tokenizer, global_model, input_text)
        filtered_response = filtering_response(response) 
        final_response = filter_final_response(filtered_response)
        print('pppppppppppppppp',final_response)

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