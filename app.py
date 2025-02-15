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
    
st.set_page_config(page_title="BogotAI", layout="centered", menu_items= {
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    })

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
    """
    Obtiene el contexto relevante de los documentos para una consulta.
    """
    similar_docs = vector_store.similarity_search(query, k=3)
    context_list = []
    
    for doc in similar_docs:
        # Extraer información básica del documento
        content = doc.page_content
        source = doc.metadata.get('source', 'Documento sin especificar')
        
        # Extraer datos cuantitativos
        numbers = re.findall(r'(\d+(?:\.\d+)?(?:%|\s+(?:habitantes|personas|viviendas)))', content)
        metrics = numbers[:3] if numbers else []
        
        # Extraer referencias a políticas, programas o indicadores
        refs = re.findall(r'(?:Plan|Programa|Proyecto|Meta|Indicador)[\s:].*?(?=\n|$)', content)
        
        context_list.append({
            'source': source,
            'content': content[:300],  # Limitar longitud del contenido
            'metrics': metrics,
            'refs': refs[:3]  # Limitar número de referencias
        })
    
    return context_list

SYSTEM_PROMPT = """
Eres BogotAI, un asistente especializado para apoyar al equipo de la Alcaldía de Bogotá. Tu función es proporcionar información precisa y análisis basados en datos para apoyar la toma de decisiones.

## Tipos de Consultas y Respuestas

1. CONSULTAS SOBRE INDICADORES
Cuando te pregunten sobre indicadores específicos (ej: seguridad, movilidad, pobreza):
- Proporciona el dato más reciente disponible
- Muestra la evolución histórica si está disponible
- Compara con otras localidades o ciudades relevantes
- Identifica tendencias clave y puntos de atención

Ejemplo:
"¿Cómo ha evolucionado la pobreza en Bogotá?"
```
Datos actuales: [último dato disponible]
Tendencia: [análisis de evolución]
Comparación territorial: [diferencias por localidad]
Puntos clave: [factores relevantes]
```

2. CONSULTAS SOBRE POLÍTICAS ESPECÍFICAS
Para preguntas sobre programas o políticas concretas:
- Resume el objetivo y alcance
- Indica el estado actual de implementación
- Señala los principales logros y desafíos
- Sugiere oportunidades de mejora basadas en evidencia

3. CONSULTAS DE CONTEXTUALIZACIÓN
Cuando necesiten entender el contexto de un problema:
- Proporciona antecedentes relevantes
- Explica factores causales
- Describe intentos previos de solución
- Menciona experiencias exitosas de otras ciudades

4. CONSULTAS DE IMPLEMENTACIÓN
Para preguntas sobre ejecución de programas:
- Detalla pasos concretos y cronograma
- Identifica recursos necesarios
- Anticipa posibles obstáculos
- Sugiere indicadores de seguimiento

## Uso del Contexto

- Al citar datos o información, especifica siempre la fuente y fecha
- Prioriza la información más reciente disponible
- Usa el contexto local de Bogotá cuando esté disponible
- Indica claramente cuando la información esté desactualizada o sea limitada

## Principios de Respuesta

1. CLARIDAD Y PRECISIÓN
- Usa lenguaje claro y directo
- Estructura las respuestas en secciones
- Prioriza información accionable
- Destaca los puntos más importantes

2. ENFOQUE EN DATOS
- Basa las recomendaciones en evidencia
- Presenta datos de forma clara y contextualizada
- Señala limitaciones o vacíos en los datos
- Sugiere métricas de seguimiento

3. ORIENTACIÓN PRÁCTICA
- Enfócate en soluciones viables
- Considera restricciones presupuestales
- Ten en cuenta la capacidad institucional
- Prioriza acciones de alto impacto

## Temas Prioritarios

1. SEGURIDAD Y CONVIVENCIA
- Tasas de criminalidad
- Percepción de seguridad
- Programas de prevención
- Coordinación institucional

2. MOVILIDAD
- Estado de obras en curso
- Indicadores de transporte público
- Congestión vehicular
- Infraestructura de transporte

3. EQUIDAD SOCIAL
- Indicadores de pobreza
- Acceso a servicios
- Programas sociales
- Brechas territoriales

4. GESTIÓN PÚBLICA
- Ejecución presupuestal
- Indicadores de servicio
- Modernización administrativa
- Participación ciudadana

## Formato de Respuesta

Para cada consulta, estructura tu respuesta así:

1. RESUMEN EJECUTIVO
- Puntos clave
- Datos relevantes
- Principales hallazgos

2. ANÁLISIS DETALLADO
- Contexto
- Tendencias
- Factores causales
- Impactos

3. RECOMENDACIONES
- Acciones inmediatas
- Estrategias de mediano plazo
- Consideraciones de implementación

4. SEGUIMIENTO
- Indicadores clave
- Puntos de control
- Próximos pasos

## Advertencias y Limitaciones

- Indica claramente cuando la información esté desactualizada
- Señala áreas donde falten datos o evidencia
- Especifica cuando las recomendaciones sean preliminares
- Sugiere la consulta con expertos cuando sea necesaria  
- Si no tienes informacion sobre algo en especifico, responde con que no tienes suficiente informacion sobre eso o neesitas mas informacion sobre eso. 
- SIEMPRE RESPONDE EN ESPAÑOL  
- Si te saludan "Hola BogotAI" o preguntan quien eres respondeles de manera concisas diciendo quien ers y en que puedes ayudarlos. 

Recuerda: Tu rol es apoyar la toma de decisiones proporcionando información y análisis basado en evidencia, no tomar las decisiones finales.
"""

def detect_response_format(prompt):
    """
    Detecta si una consulta requiere una respuesta estructurada o simple.
    Retorna un string con el formato detectado.
    
    Parameters:
    prompt (str): La consulta del usuario
    
    Returns:
    str: 'STRUCTURED' o 'SIMPLE'
    """
    prompt = prompt.lower()
    
    # Indicadores de consulta estructurada
    structured_indicators = [
        # Análisis y comparación
        'analizar', 'comparar', 'evaluar', 'diferencia',
        'evolución', 'tendencia', 'impacto',
        
        # Planeación y gestión
        'plan', 'programa', 'proyecto', 'estrategia',
        'política', 'presupuesto', 'implementación',
        
        # Territorio y datos
        'localidad', 'territorio', 'zona', 'sector',
        'estadística', 'indicador', 'porcentaje', 'densidad',
        
        # Temáticas complejas
        'seguridad', 'movilidad', 'pobreza', 'desarrollo',
        'infraestructura', 'ambiente', 'educación', 'salud'
    ]
    
    # Indicadores de consulta simple
    simple_indicators = [
        # Preguntas básicas
        'qué es', 'que es', 'dónde', 'donde', 'cuándo', 'cuando',
        'quién', 'quien', 'cuál', 'cual', 'cuánto', 'cuanto',
        
        # Definiciones y datos puntuales
        'significa', 'define', 'explica', 'valor', 'dato',
        'horario', 'dirección', 'teléfono', 'requisito'
    ]
    
    # Criterios de complejidad
    is_complex = (
        len(prompt.split()) > 15 or              # Longitud de la pregunta
        prompt.count('?') > 1 or                 # Múltiples preguntas
        prompt.count(',') > 1 or                 # Múltiples elementos
        prompt.count(' y ') > 1 or              # Múltiples conceptos
        any(ind in prompt for ind in structured_indicators)  # Indicadores de estructura
    )
    
    # Criterios de simplicidad
    is_simple = (
        any(ind in prompt for ind in simple_indicators) and  # Indicadores simples
        not is_complex                                       # No es compleja
    )
    
    return 'SIMPLE' if is_simple else 'STRUCTURED'

def format_structured_response(query_type, context):
    """
    Formatea una respuesta estructurada seleccionando secciones relevantes
    según el tipo de consulta.
    """
    # Definir todas las secciones posibles con sus emojis y contenido
    sections = {
        'resumen': ('📋', 'RESUMEN EJECUTIVO', ['Síntesis del tema', 'Puntos clave', 'Contexto general']),
        'objetivos': ('🎯', 'OBJETIVOS Y ALCANCE', ['Objetivos principales', 'Población objetivo', 'Cobertura']),
        'indicadores': ('📊', 'INDICADORES CLAVE', ['Estado actual', 'Evolución', 'Metas']),
        'territorial': ('📍', 'ANÁLISIS TERRITORIAL', ['Impacto por localidades', 'Zonas críticas', 'Distribución']),
        'recursos': ('💰', 'RECURSOS Y PRESUPUESTO', ['Presupuesto', 'Fuentes', 'Ejecución']),
        'implementacion': ('📅', 'IMPLEMENTACIÓN', ['Estado actual', 'Cronograma', 'Hitos']),
        'recomendaciones': ('⚡', 'RECOMENDACIONES', ['Acciones sugeridas', 'Prioridades', 'Seguimiento']),
        'normativo': ('⚖️', 'MARCO NORMATIVO', ['Normativa aplicable', 'Competencias', 'Requisitos']),
        'actores': ('👥', 'ACTORES CLAVE', ['Responsables', 'Aliados', 'Grupos de interés'])
    }

    # Mapear tipos de consulta a secciones relevantes
    type_sections = {
        'SEGURIDAD_MOVILIDAD': ['resumen', 'indicadores', 'territorial', 'implementacion', 'recomendaciones'],
        'EQUIDAD_SOCIAL': ['resumen', 'objetivos', 'indicadores', 'recursos', 'recomendaciones'],
        'PLANEACION_TERRITORIO': ['resumen', 'objetivos', 'territorial', 'implementacion', 'normativo'],
        'GESTION_RECURSOS': ['resumen', 'recursos', 'indicadores', 'implementacion', 'actores'],
        'AMBIENTE_DESARROLLO': ['resumen', 'objetivos', 'territorial', 'implementacion', 'recomendaciones'],
        'SERVICIOS_CIUDADANOS': ['resumen', 'objetivos', 'normativo', 'actores', 'recomendaciones']
    }

    # Obtener secciones relevantes para el tipo de consulta
    relevant_sections = type_sections.get(query_type, ['resumen', 'recomendaciones'])

    # Construir el prompt
    prompt_parts = [f"Tipo de consulta: {query_type}\n"]
    
    # Agregar contexto si existe
    if context:
        prompt_parts.append(f"Contexto relevante:\n{context}\n")

    # Agregar secciones relevantes
    for section_key in relevant_sections:
        if section_key in sections:
            emoji, title, bullets = sections[section_key]
            prompt_parts.append(f"""
{emoji} {title}:
• {bullets[0]}
• {bullets[1]}
• {bullets[2]}
""")

    return "\n".join(prompt_parts)

def format_simple_response(query_type, context):
    """Genera un prompt para respuesta simple"""
    return f"""
    Tipo de consulta: {query_type}
    
    Contexto municipal relevante:
    {context}
    
    Proporciona una respuesta clara y concisa en formato de párrafo, sin usar viñetas ni secciones.
    La respuesta debe ser directa y enfocada en responder la pregunta específica.
    """

def format_municipal_context(context_list):
    """
    Formatea el contexto municipal para presentación.
    """
    if not isinstance(context_list, list):
        return "No se encontró contexto relevante."
        
    formatted_parts = []
    
    for item in context_list:
        # Formatear referencias
        refs = item.get('refs', [])
        refs_text = "\n• ".join(refs) if refs else "No hay referencias específicas"
        
        # Formatear métricas
        metrics = item.get('metrics', [])
        metrics_text = "\n• ".join(metrics) if metrics else "No hay datos cuantitativos específicos"
        
        section = f"""
📚 Fuente: {item['source']}

📊 Datos clave:
• {metrics_text}

📋 Referencias:
• {refs_text}

💡 Contexto relevante:
{item['content']}"""
        formatted_parts.append(section)
    
    return "\n---\n".join(formatted_parts)

def detect_query_type(prompt):
    """
    Detecta el tipo de consulta basado en los ejes principales del Plan de 
    Desarrollo de Bogotá y prioridades de la administración.
    
    Parameters:
    prompt (str): Consulta del usuario
    
    Returns:
    tuple: (tipo_principal, subtipo, score)
    """
    prompt = prompt.lower()
    
    keywords = {
        'SEGURIDAD_MOVILIDAD': [
            # Seguridad
            'seguridad', 'convivencia', 'delito', 'crimen', 'policía',
            'vigilancia', 'prevención', 'violencia', 'hurto',
            # Movilidad
            'transporte', 'metro', 'transmilenio', 'ciclovía', 'tráfico',
            'congestión', 'obras viales', 'infraestructura vial', 'peatones'
        ],
        
        'EQUIDAD_SOCIAL': [
            # Pobreza y desigualdad
            'pobreza', 'vulnerabilidad', 'inequidad', 'brecha social',
            'transferencias', 'subsidios', 'ayudas', 'inclusión',
            # Servicios sociales
            'educación', 'salud', 'vivienda', 'alimentación', 'cuidado',
            'primera infancia', 'adulto mayor', 'discapacidad', 'género'
        ],
        
        'PLANEACION_TERRITORIO': [
            # Planeación
            'plan de desarrollo', 'pot', 'ordenamiento', 'planeación',
            'estrategia', 'proyecto', 'programa', 'política pública',
            # Territorio
            'localidad', 'upz', 'territorio', 'densidad', 'uso del suelo',
            'espacio público', 'equipamientos', 'región metropolitana'
        ],
        
        'GESTION_RECURSOS': [
            # Gestión pública
            'presupuesto', 'inversión', 'recursos', 'contratación',
            'ejecución', 'gestión', 'administrativo', 'modernización',
            # Control
            'seguimiento', 'indicadores', 'evaluación', 'metas',
            'transparencia', 'rendición', 'control', 'auditoría'
        ],
        
        'AMBIENTE_DESARROLLO': [
            # Ambiente
            'ambiente', 'sostenibilidad', 'cambio climático', 'residuos',
            'reciclaje', 'contaminación', 'aire', 'agua', 'verde',
            # Desarrollo económico
            'economía', 'emprendimiento', 'empleo', 'productividad',
            'innovación', 'tecnología', 'competitividad', 'mipymes'
        ],
        
        'SERVICIOS_CIUDADANOS': [
            # Trámites
            'trámite', 'servicio', 'procedimiento', 'requisitos',
            'documentos', 'solicitud', 'atención', 'ventanilla',
            # Participación
            'participación', 'consulta', 'ciudadanía', 'comunidad',
            'socialización', 'encuentro', 'diálogo', 'concertación'
        ]
    }
    
    prompt_lower = prompt.lower()
    for category, terms in keywords.items():
        if any(term in prompt_lower for term in terms):
            return category
    return 'GENERAL'

def format_context_string(context_list):
    """
    Formatea la lista de contextos en un string estructurado.
    """
    if not context_list:
        return "No se encontró contexto relevante."
        
    formatted_parts = []
    
    for item in context_list:
        section = f"""
📚 Fuente: {item['source']}

📊 Datos relevantes:
{format_metrics(item.get('metrics', []))}

📋 Referencias:
{format_references(item.get('refs', []))}

💡 Contexto:
{item['content']}"""
        formatted_parts.append(section)
    
    return "\n---\n".join(formatted_parts)

def format_metrics(metrics):
    if not metrics:
        return "• No hay datos cuantitativos específicos"
    return "\n".join(f"• {metric}" for metric in metrics)

def format_references(refs):
    if not refs:
        return "• No hay referencias específicas"
    return "\n".join(f"• {ref}" for ref in refs)

def get_chat_response(prompt, vector_store, temperature=0.3):
    """Genera respuesta considerando el contexto municipal y el formato apropiado"""
    try:
        response_placeholder = st.empty()
        stream_handler = StreamHandler(response_placeholder)
        
        # 1. Detectar tipo de consulta
        query_type = detect_query_type(prompt)
        
        # 2. Obtener y formatear contexto
        context_items = get_municipal_context(vector_store, prompt)
        formatted_context = format_context_string(context_items)
        
        # 3. Construir prompt mejorado
        enhanced_prompt = f"""
Tipo de consulta: {query_type}

Contexto relevante:
{formatted_context}

Por favor proporciona una respuesta que:
1. Use la información del contexto proporcionado
2. Cite las fuentes específicas cuando sea posible
3. Destaque datos cuantitativos relevantes
4. Proporcione recomendaciones basadas en evidencia
"""
        
        # 4. Generar respuesta
        chat_model = ChatOpenAI(
            model="gpt-4",
            temperature=temperature,
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
    st.title("BogotAI", anchor=False)
    st.markdown("##### Asistente para la toma de decisiones de política pública")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.markdown("""
        ## Sistema de Gestión Pública Bogotá
        
        **Tipos de consultas:**
        - Planificación y diseño de Políticas Públicas 
        - Análisis de problemas sociales 
        - Evaluación de escenarios 
        - Identificación de patrones y tendencias 
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