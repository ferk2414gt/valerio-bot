import streamlit as st 
from chatbot import predict_class, get_response, intents

st.title("Asistente virtual")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "first_message" not in st.session_state:
    st.session_state.first_message = True


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
         st.markdown(message["content"])

if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola,¿como puedo ayudarte?")


st.session_state.messages.append({"role":"assistant","content":"hola, ¿como puedo ayudarte?"})
st.session_state.first_message = False       

if prompt := st.chat_input("¿como puedo ayudarte?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})
    
#implementar IA
    insts=predict_class(prompt)
    res=get_response(insts,intents) 
   
    with st.chat_message("assistant"):
        st.markdown(res)

    st.session_state.messages.append({"role": "assistant", "content":res}) 

