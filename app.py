import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from html_template import logo
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import re

# Load environment variables
load_dotenv()

# Get API key from environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    st.error("Error: OPENAI_API_KEY not found in environment variables")
    st.stop()
    
st.set_page_config(page_title="BogoTIA", layout="centered")

class MunicipalDocumentProcessor:
    def __init__(self, pdf_directory="data", index_directory="faiss_index"):
        self.pdf_directory = pdf_directory
        self.index_directory = index_directory
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        os.makedirs(self.pdf_directory, exist_ok=True)
        os.makedirs(self.index_directory, exist_ok=True)

    def load_vector_store(self):
        """Carga el vector store existente"""
        index_path = os.path.join(self.index_directory, "index.faiss")
        try:
            if os.path.exists(index_path):
                return FAISS.load_local(
                    self.index_directory, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            return None
        except Exception as e:
            st.error(f"Error cargando vector store: {str(e)}")
            return None

    def process_documents(self):
        """Procesa los documentos PDF y crea el vector store"""
        try:
            pdf_files = [f for f in os.listdir(self.pdf_directory) if f.endswith('.pdf')]
            if not pdf_files:
                st.warning("No se encontraron archivos PDF en el directorio data.")
                return None

            documents = []
            successful_files = []
            failed_files = []
            
            for pdf_file in pdf_files:
                try:
                    file_path = os.path.join(self.pdf_directory, pdf_file)
                    
                    with open(file_path, 'rb') as file:
                        header = file.read(5)
                        if header != b'%PDF-':
                            failed_files.append((pdf_file, "Encabezado PDF inválido"))
                            continue
                    
                    loader = PyPDFLoader(file_path)
                    doc_pages = loader.load()
                    
                    if doc_pages:
                        documents.extend(doc_pages)
                        successful_files.append(pdf_file)
                        st.success(f"✅ Procesado exitosamente: {pdf_file} ({len(doc_pages)} páginas)")
                    else:
                        failed_files.append((pdf_file, "No se pudo extraer contenido"))
                
                except Exception as e:
                    failed_files.append((pdf_file, str(e)))
                    continue

            st.write("---")
            st.write("📊 Resumen de procesamiento:")
            st.write(f"- Total archivos: {len(pdf_files)}")
            st.write(f"- Procesados correctamente: {len(successful_files)}")
            st.write(f"- Fallidos: {len(failed_files)}")
            
            if failed_files:
                st.error("❌ Archivos que no se pudieron procesar:")
                for file, error in failed_files:
                    st.write(f"- {file}: {error}")

            if not documents:
                st.warning("⚠️ No se pudo extraer contenido de ningún PDF.")
                return None

            texts = self.text_splitter.split_documents(documents)
            vectorstore = FAISS.from_documents(texts, self.embeddings)
            vectorstore.save_local(self.index_directory)
            
            st.success(f"✅ Vector store creado exitosamente con {len(texts)} fragmentos de texto")
            return vectorstore
        
        except Exception as e:
            st.error(f"Error procesando documentos: {str(e)}")
            return None

def setup_retrieval_chain(vector_store):
    """Configura la cadena de recuperación para consultas"""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0),
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return retrieval_chain

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def get_municipal_context(vector_store, query):
    """Obtiene el contexto municipal relevante para una consulta"""
    similar_docs = vector_store.similarity_search(query, k=5)
    context = []
    
    for doc in similar_docs:
        content = doc.page_content
        source = doc.metadata.get('source', 'Documento desconocido')
        page = doc.metadata.get('page', 'N/A')
        
        # Extraer referencias específicas
        refs = re.findall(r'(?:Acuerdo|Decreto|Resolución|Plan)\s+\d+[^\n]*', content)
        
        context.append({
            'source': f"{source} (Pág. {page})",
            'content': content,
            'refs': refs
        })
    
    return context

SYSTEM_PROMPT = """
Eres AlcaldíaApp, un asistente especializado en gestión municipal y administración pública, enfocado en apoyar a funcionarios de la alcaldía.

ÁREAS DE ESPECIALIZACIÓN:

📋 PLANEACIÓN:
- Plan de Desarrollo Municipal
- Plan de Ordenamiento Territorial
- Planes de Acción
- Proyectos Estratégicos

💰 GESTIÓN FINANCIERA:
- Presupuesto Municipal
- Ejecución Presupuestal
- Fuentes de Financiación
- Control de Gastos

🏗️ INFRAESTRUCTURA:
- Proyectos de Obra Pública
- Mantenimiento de Infraestructura
- Inventario de Bienes
- Gestión de Activos

👥 GESTIÓN SOCIAL:
- Programas Sociales
- Participación Ciudadana
- Atención al Ciudadano
- Desarrollo Comunitario

📊 CONTROL Y SEGUIMIENTO:
- Indicadores de Gestión
- Informes de Seguimiento
- Evaluación de Proyectos
- Rendición de Cuentas

FORMATO DE RESPUESTA:

📌 CONTEXTO:
• [Descripción del tema/situación]

📝 PROCEDIMIENTO:
• [Pasos a seguir]

📄 DOCUMENTACIÓN:
• [Documentos relacionados]

⚖️ MARCO NORMATIVO:
• [Referencias legales]

🎯 PUNTOS CLAVE:
• [Aspectos importantes]

👥 ACTORES INVOLUCRADOS:
• [Dependencias/personas]

📊 SEGUIMIENTO:
• [Indicadores/métricas]

DIRECTRICES:
1. Priorizar eficiencia administrativa
2. Garantizar transparencia
3. Promover participación ciudadana
4. Asegurar cumplimiento normativo
"""

def detect_response_format(prompt):
    """Detecta el formato de respuesta más apropiado basado en la consulta"""
    # Keywords que sugieren una respuesta estructurada
    structured_keywords = [
        'procedimiento', 'pasos', 'cómo', 'proceso', 'requisitos',
        'documentos', 'trámite', 'gestión', 'proyecto', 'plan',
        'presupuesto', 'informe', 'evaluación'
    ]
    
    # Keywords que sugieren una respuesta simple
    simple_keywords = [
        'qué es', 'cuándo', 'dónde', 'quién', 'cuál',
        'define', 'explica', 'significado', 'concepto',
        'información', 'dato','que es', 'cuando', 'donde', 'quien', 'cual',
        'informacion'
    ]
    
    prompt_lower = prompt.lower()
    
    # Detectar si la pregunta es compleja por su longitud
    is_complex = len(prompt.split()) > 15
    
    # Detectar si contiene múltiples preguntas
    has_multiple_questions = prompt.count('?') > 1
    
    # Determinar el formato basado en las condiciones
    if has_multiple_questions or any(keyword in prompt_lower for keyword in structured_keywords):
        return 'STRUCTURED'
    elif any(keyword in prompt_lower for keyword in simple_keywords) and not is_complex:
        return 'SIMPLE'
    else:
        return 'STRUCTURED'

def format_structured_response(query_type, context):
    """Genera un prompt para respuesta estructurada"""
    return f"""
    Tipo de consulta: {query_type}
    
    Contexto municipal relevante:
    {context}
    
    Proporciona una respuesta estructurada que incluya:
    
    📌 CONTEXTO:
    • [Descripción del tema/situación]

    📝 PROCEDIMIENTO:
    • [Pasos a seguir]

    📄 DOCUMENTACIÓN:
    • [Documentos relacionados]

    ⚖️ MARCO NORMATIVO:
    • [Referencias legales]

    🎯 PUNTOS CLAVE:
    • [Aspectos importantes]
    """

def format_simple_response(query_type, context):
    """Genera un prompt para respuesta simple"""
    return f"""
    Tipo de consulta: {query_type}
    
    Contexto municipal relevante:
    {context}
    
    Proporciona una respuesta clara y concisa en formato de párrafo, sin usar viñetas ni secciones.
    La respuesta debe ser directa y enfocada en responder la pregunta específica.
    """

def format_municipal_context(context):
    """Formatea el contexto municipal para el prompt"""
    formatted = []
    for item in context:
        refs = '\n'.join(f"• {ref}" for ref in item['refs']) if item['refs'] else "No se encontraron referencias específicas"
        formatted.append(f"""
        📚 Fuente: {item['source']}
        
        📋 Referencias:
        {refs}
        
        💡 Contexto relevante:
        {item['content'][:500]}...
        """)
    return '\n'.join(formatted)

def detect_query_type(prompt):
    """Detecta el tipo de consulta para adaptar la respuesta"""
    keywords = {
        'PLANEACION': ['plan', 'desarrollo', 'pot', 'proyecto', 'estratégico'],
        'FINANZAS': ['presupuesto', 'financiero', 'gastos', 'recursos'],
        'INFRAESTRUCTURA': ['obra', 'mantenimiento', 'inventario', 'activos'],
        'SOCIAL': ['comunidad', 'ciudadano', 'participación', 'programa'],
        'CONTROL': ['seguimiento', 'indicador', 'evaluación', 'informe']
    }
    
    prompt_lower = prompt.lower()
    for category, terms in keywords.items():
        if any(term in prompt_lower for term in terms):
            return category
    return 'GENERAL'

def get_chat_response(prompt, vector_store, temperature=0.3):
    """Genera respuesta considerando el contexto municipal y el formato apropiado"""
    try:
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)
        
        # Detectar tipo de consulta y formato de respuesta
        query_type = detect_query_type(prompt)
        response_format = detect_response_format(prompt)
        municipal_context = get_municipal_context(vector_store, prompt)
        
        # Seleccionar el formato de prompt según el tipo de respuesta
        if response_format == 'STRUCTURED':
            enhanced_prompt = format_structured_response(
                query_type, 
                format_municipal_context(municipal_context)
            )
        else:
            enhanced_prompt = format_simple_response(
                query_type, 
                format_municipal_context(municipal_context)
            )
        
        # Ajustar la temperatura según el formato
        # Menor temperatura para respuestas estructuradas, mayor para respuestas simples
        adjusted_temperature = 0.3 if response_format == 'STRUCTURED' else 0.7
        
        chat_model = ChatOpenAI(
            model="gpt-4o",
            temperature=adjusted_temperature,
            api_key=API_KEY,
            streaming=True,
            callbacks=[stream_handler]
        )
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"{prompt}\n\n{enhanced_prompt}")
        ]
        
        response = chat_model.invoke(messages)
        return stream_handler.text
            
    except Exception as e:
        st.error(f"Error generando respuesta: {str(e)}")
        return "Lo siento, ocurrió un error al procesar su solicitud."

def main():
    processor = MunicipalDocumentProcessor()
    
    if os.path.exists(os.path.join("faiss_index", "index.faiss")):
        vector_store = processor.load_vector_store()
    else:
        st.warning("Procesando documentos municipales por primera vez...")
        vector_store = processor.process_documents()
    
    if vector_store is None:
        st.error("No se pudo inicializar la base de conocimientos")
        st.stop()

    st.write(logo, unsafe_allow_html=True)
    st.title("BogoTIA", anchor=False)
    st.markdown("**Asistente virtual para gestión municipal y administración pública**")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.markdown("""
        **Sistema de Gestión Municipal**
        
        Tipos de consultas:
        - Planes y proyectos
        - Gestión administrativa
        - Procedimientos internos
        - Control y seguimiento
        """)
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("¿En qué puedo ayudarte?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response = get_chat_response(prompt, vector_store)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()