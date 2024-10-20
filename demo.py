
import streamlit as st
from openai import AzureOpenAI
import cv2
import base64
import numpy as np


# Azure OpenAI API configuration
api_base = ''
api_key = ''
deployment_name = ''
api_version = ''

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{api_base}/openai/deployments/{deployment_name}"
)


st.title("Visual Data Analysis Assistant")

# Upload image
uploaded_file = st.file_uploader("Upload image which contains graph or Charts...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Display the image
    st.image(img, channels="BGR")

    # Encode image to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Get user input for the message
    message = st.text_input("Enter your message:")

    if st.button("Analyze"):
        # Prepare the content for the API
        user_content = [{"type": "text", "text": message}]
        if uploaded_file:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}})

        # Call the Azure OpenAI API
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': user_content}
            ]
        )

        # Display the response
        response_dict = response.to_dict()
        # st.write(response_dict['choices'][0]['message']['content'])
        default_text = response_dict['choices'][0]['message']['content']
        st.text_area("Response:", value=default_text, height=200)
