import os
import io
import asyncio
import aiohttp
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import time
import json

load_dotenv()

@dataclass
class ProcessedDocument:
    """Structure for processed document information"""
    filename: str
    content: str
    chunks: List[str]
    total_chars: int
    doc_type: str
    summary: str = ""

@dataclass
class AgentResponse:
    """Structure for storing agent responses"""
    agent_name: str
    content: str
    confidence: float
    metadata: Dict = None
    processing_time: float = 0.0

@dataclass
class DietPlan:
    """Structure for storing diet plan information"""
    breakfast: str
    lunch: str
    dinner: str
    snacks: str
    notes: str

class DocumentProcessor:
    """Enhanced document processing with better error handling and progress tracking"""
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.processed_documents: List[ProcessedDocument] = []
        self._initialize_embeddings()
        self.vector_store = None

    def _initialize_embeddings(self):
        """Initialize Google AI embeddings"""
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {str(e)}")
            raise

    async def process_file(self, file, progress_callback) -> ProcessedDocument:
        """Process a single file with progress tracking"""
        try:
            progress_callback(0.2, f"Processing {file.name}")
            
            if file.type == "application/pdf":
                content = await self.process_pdf(file)
                doc_type = "PDF"
            elif file.type.startswith("image/"):
                content = await self.process_image(file)
                doc_type = "Image"
            else:
                raise ValueError(f"Unsupported file type: {file.type}")

            progress_callback(0.4, "Splitting content into chunks")
            chunks = self.text_splitter.split_text(content)
            
            progress_callback(0.6, "Generating document summary")
            summary = await self._generate_summary(content[:1000])
            
            progress_callback(0.8, "Finalizing document processing")
            
            return ProcessedDocument(
                filename=file.name,
                content=content,
                chunks=chunks,
                total_chars=len(content),
                doc_type=doc_type,
                summary=summary
            )
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return None

    async def process_pdf(self, pdf_file) -> str:
        """Process PDF file with enhanced error handling"""
        text = ""
        try:
            pdf_reader = PdfReader(pdf_file)
            for page_num, page in enumerate(pdf_reader.pages):
                extracted_text = page.extract_text()
                if extracted_text:
                    text += f"Page {page_num + 1}:\n{extracted_text}\n\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"PDF processing error: {str(e)}")

    async def process_image(self, image_file) -> str:
        """Process image with OCR and error handling"""
        try:
            image = Image.open(image_file)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            raise Exception(f"Image processing error: {str(e)}")

    async def _generate_summary(self, text: str) -> str:
        """Generate a brief summary of the document content"""
        return f"{text[:200]}..."

    async def update_vector_store(self, documents: List[ProcessedDocument], progress_callback):
        """Update vector store with new documents"""
        try:
            all_chunks = []
            metadata_list = []
            
            for idx, doc in enumerate(documents):
                progress_callback(0.2 + (0.6 * (idx / len(documents))), 
                                f"Indexing {doc.filename}")
                
                for chunk_idx, chunk in enumerate(doc.chunks):
                    all_chunks.append(chunk)
                    metadata_list.append({
                        "source": doc.filename,
                        "chunk_index": chunk_idx,
                        "doc_type": doc.doc_type
                    })

            if all_chunks:
                progress_callback(0.8, "Creating vector store")
                self.vector_store = FAISS.from_texts(
                    all_chunks,
                    self.embeddings,
                    metadatas=metadata_list
                )
                
                progress_callback(0.9, "Saving vector store")
                self.vector_store.save_local("faiss_index")
                
                return True
                
        except Exception as e:
            st.error(f"Vector store update error: {str(e)}")
            return False

class AgentStatus:
    """Enhanced agent status management with sidebar display"""
    def __init__(self):
        self.sidebar_placeholder = None
        self.agents = {
            'document_processor': {'status': 'idle', 'progress': 0, 'message': ''},
            'main_agent': {'status': 'idle', 'progress': 0, 'message': ''},
            'diagnosis_agent': {'status': 'idle', 'progress': 0, 'message': ''},
            'treatment_agent': {'status': 'idle', 'progress': 0, 'message': ''},
            'research_agent': {'status': 'idle', 'progress': 0, 'message': ''},
            'synthesis_agent': {'status': 'idle', 'progress': 0, 'message': ''},
            'diet_agent': {'status': 'idle', 'progress': 0, 'message': ''}
        }
        
    def initialize_sidebar_placeholder(self):
        """Initialize the sidebar placeholder"""
        with st.sidebar:
            self.sidebar_placeholder = st.empty()
    
    def update_status(self, agent_name: str, status: str, progress: float, message: str = ""):
        """Update agent status and refresh sidebar display"""
        self.agents[agent_name] = {
            'status': status,
            'progress': progress,
            'message': message
        }
        self._render_status()

    def _render_status(self):
        """Render status in sidebar"""
        if self.sidebar_placeholder is None:
            self.initialize_sidebar_placeholder()
            
        with self.sidebar_placeholder.container():
            for agent_name, status in self.agents.items():
                self._render_agent_card(agent_name, status)

    def _render_agent_card(self, agent_name: str, status: dict):
        """Render individual agent status card in sidebar"""
        colors = {
            'idle': '#6c757d',
            'working': '#007bff',
            'completed': '#28a745',
            'error': '#dc3545'
        }
        color = colors.get(status['status'], colors['idle'])
        
        st.markdown(f"""
            <div style="
                background-color: #1E1E1E;
                padding: 0.8rem;
                border-radius: 0.5rem;
                margin-bottom: 0.8rem;
                border: 1px solid {color};
            ">
                <div style="color: {color}; font-weight: bold;">
                    {agent_name.replace('_', ' ').title()}
                </div>
                <div style="
                    color: #CCCCCC;
                    font-size: 0.8rem;
                    margin: 0.3rem 0;
                ">
                    {status['message'] or status['status'].title()}
                </div>
                <div style="
                    height: 4px;
                    background-color: rgba(255,255,255,0.1);
                    border-radius: 2px;
                    margin-top: 0.5rem;
                ">
                    <div style="
                        width: {status['progress'] * 100}%;
                        height: 100%;
                        background-color: {color};
                        border-radius: 2px;
                        transition: width 0.3s ease;
                    "></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

class DietAgent:
    """Agent for generating personalized diet plans"""
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama3-8b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self._initialize_prompt()
        # Specialized diet plans for common conditions
        self.specialized_diets = {
            "kidney_stone": DietPlan(
                breakfast="Low-oxalate cereal with milk",
                lunch="Chicken with white rice",
                dinner="Fish with steamed vegetables",
                snacks="Yogurt or apple",
                notes="Limit salt, drink plenty water"
            )
        }

    def _initialize_prompt(self):
        """Initialize diet agent prompt"""
        self.prompt = """You are a nutrition specialist. Be concise.
Context: {context}
Query: {query}
Chat History: {chat_history}

Based on the conversation, provide a simple and short diet plan with:
1. Breakfast (1 item)
2. Lunch (1 item)
3. Dinner (1 item)
4. Snacks (1 item)
5. Brief note (1-2 words)

Format as JSON with keys: breakfast, lunch, dinner, snacks, notes.
Keep each suggestion under 5 words. Total response must be under 50 words."""

        self.agent = ChatPromptTemplate.from_messages([
            ("system", self.prompt),
            ("human", "{input}")
        ]) | self.llm | StrOutputParser()

    async def generate_diet_plan(self, query: str, context: str, chat_history: str) -> DietPlan:
        """Generate a diet plan based on the conversation"""
        try:
            # Check for specialized diet needs
            query_lower = query.lower()
            if "kidney" in query_lower and ("stone" in query_lower or "stones" in query_lower):
                return self.specialized_diets["kidney_stone"]
                
            start_time = time.time()
            
            response = await self.agent.ainvoke({
                "input": query,
                "context": context,
                "query": query,
                "chat_history": chat_history
            })
            
            # Try to parse JSON response
            try:
                diet_data = json.loads(response)
                diet_plan = DietPlan(
                    breakfast=diet_data.get("breakfast", "Oatmeal with fruits"),
                    lunch=diet_data.get("lunch", "Grilled chicken salad"),
                    dinner=diet_data.get("dinner", "Baked fish with vegetables"),
                    snacks=diet_data.get("snacks", "Yogurt with nuts"),
                    notes=diet_data.get("notes", "Stay hydrated")
                )
            except json.JSONDecodeError:
                # Fallback if response is not valid JSON
                diet_plan = DietPlan(
                    breakfast="Oatmeal with fruits",
                    lunch="Grilled chicken salad",
                    dinner="Baked fish with vegetables",
                    snacks="Yogurt with nuts",
                    notes="Stay hydrated"
                )
            
            return diet_plan
            
        except Exception as e:
            # Return default diet plan on error
            return DietPlan(
                breakfast="Oatmeal with fruits",
                lunch="Grilled chicken salad", 
                dinner="Baked fish with vegetables",
                snacks="Yogurt with nuts",
                notes="Stay hydrated"
            )

class HealthcareAgent:
    """Healthcare agent with concise response generation"""
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama3-8b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.chat_history = []
        self.doc_processor = DocumentProcessor()
        self.diet_agent = DietAgent()
        self._initialize_prompts()
        self.agents = self._initialize_agents()

    def _initialize_prompts(self):
        """Initialize prompts optimized for concise responses"""
        self.prompts = {
            'main_agent': """You are a healthcare coordinator AI. Be direct and concise.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide a brief response with:
1. Key medical concepts (2-3 points)
2. Necessary specialist consultations
3. Quick initial assessment
Limit response to 3-4 sentences.""",

            'diagnosis_agent': """You are a medical diagnosis specialist. Be concise.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide brief:
1. Key symptoms identified
2. Top 2-3 potential conditions
3. Immediate next steps
Limit to 3-4 key points.""",

            'treatment_agent': """You are a treatment specialist. Be direct.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide only:
1. Top 1-2 treatment options
2. Key lifestyle changes
3. Critical warning signs
Keep response under 100 words.""",

            'research_agent': """You are a medical research specialist. Be brief.
Context: {context}
Query: {query}
Chat History: {chat_history}

Provide only:
1. Most relevant research finding
2. Key clinical guideline
3. Primary recommendation
Limit to 2-3 sentences.""",

            'synthesis_agent': """You are a medical information synthesizer. Be concise.
Context: {context}
Query: {query}
Chat History: {chat_history}
Agent Responses: {agent_responses}

Provide a clear, concise summary:
1. Main recommendation
2. Key action items
3. Important warnings (if any)

Keep the final response under 150 words and focus on practical next steps.
For simple queries (like greetings), respond in one short sentence."""
        }

    def _initialize_agents(self):
        """Initialize enhanced agent system"""
        return {
            name: ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("human", "{input}")
            ]) | self.llm | StrOutputParser()
            for name, prompt in self.prompts.items()
        }

    def _format_chat_history(self) -> str:
        """Format chat history for context"""
        formatted = []
        for msg in self.chat_history[-5:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    async def process_documents(self, files, status_callback) -> bool:
        """Process documents with detailed status updates"""
        try:
            processed_docs = []
            
            for idx, file in enumerate(files):
                doc = await self.doc_processor.process_file(
                    file,
                    lambda p, m: status_callback(
                        'document_processor',
                        'working',
                        (idx / len(files)) + (p / len(files)),
                        m
                    )
                )
                if doc:
                    processed_docs.append(doc)

            if processed_docs:
                success = await self.doc_processor.update_vector_store(
                    processed_docs,
                    lambda p, m: status_callback(
                        'document_processor',
                        'working',
                        0.8 + (p * 0.2),
                        m
                    )
                )
                
                if success:
                    status_callback(
                        'document_processor',
                        'completed',
                        1.0,
                        "Documents processed successfully"
                    )
                    return True

            status_callback(
                'document_processor',
                'error',
                0,
                "Document processing failed"
            )
            return False
            
        except Exception as e:
            status_callback(
                'document_processor',
                'error',
                0,
                str(e)
            )
            return False

    async def get_relevant_context(self, query: str) -> str:
        """Get relevant context from vector store"""
        try:
            if self.doc_processor.vector_store:
                docs = self.doc_processor.vector_store.similarity_search(
                    query,
                    k=3
                )
                return "\n\n".join(doc.page_content for doc in docs)
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
        return ""

    async def process_query(
        self,
        query: str,
        status_callback
    ) -> Dict[str, Union[AgentResponse, DietPlan]]:
        """Process query through multi-agent system"""
        responses = {}
        context = await self.get_relevant_context(query)
        chat_history = self._format_chat_history()
        
        try:
            status_callback('main_agent', 'working', 0.2, "Analyzing query")
            main_response = await self._get_agent_response(
                'main_agent',
                query,
                context,
                chat_history
            )
            responses['main_agent'] = main_response
            status_callback('main_agent', 'completed', 1.0, "Analysis complete")

            status_callback('diagnosis_agent', 'working', 0.2, "Analyzing symptoms")
            status_callback('treatment_agent', 'working', 0.2, "Evaluating treatments")
            status_callback('research_agent', 'working', 0.2, "Reviewing research")
            status_callback('diet_agent', 'working', 0.2, "Creating diet plan")

            specialist_tasks = [
                self._get_agent_response('diagnosis_agent', query, context, chat_history),
                self._get_agent_response('treatment_agent', query, context, chat_history),
                self._get_agent_response('research_agent', query, context, chat_history),
                self.diet_agent.generate_diet_plan(query, context, chat_history)
            ]

            specialist_responses = await asyncio.gather(*specialist_tasks)
            
            for agent_name, response in zip(
                ['diagnosis_agent', 'treatment_agent', 'research_agent'],
                specialist_responses[:-1]  # Exclude diet agent response
            ):
                responses[agent_name] = response
                status_callback(
                    agent_name,
                    'completed',
                    1.0,
                    f"{agent_name.split('_')[0].title()} analysis complete"
                )
                
            # Process diet plan
            responses['diet_plan'] = specialist_responses[-1]
            status_callback(
                'diet_agent',
                'completed',
                1.0,
                "Diet plan generated"
            )

            status_callback('synthesis_agent', 'working', 0.5, "Synthesizing insights")
            final_response = await self._synthesize_responses(
                query,
                context,
                chat_history,
                responses
            )
            responses['synthesis_agent'] = final_response
            status_callback(
                'synthesis_agent',
                'completed',
                1.0,
                "Response synthesis complete"
            )

            self.chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=final_response.content)
            ])

            return responses

        except Exception as e:
            for agent in self.agents.keys():
                status_callback(agent, 'error', 0, str(e))
            status_callback('diet_agent', 'error', 0, str(e))
            raise Exception(f"Query processing error: {str(e)}")

    async def _get_agent_response(
        self,
        agent_name: str,
        query: str,
        context: str,
        chat_history: str
    ) -> AgentResponse:
        """Get response from specific agent with metadata"""
        start_time = time.time()
        
        try:
            response = await self.agents[agent_name].ainvoke({
                "input": query,
                "context": context,
                "query": query,
                "chat_history": chat_history
            })
            
            processing_time = time.time() - start_time
            
            metadata = {
                "processing_time": processing_time,
                "context_length": len(context),
                "query_length": len(query)
            }
            
            return AgentResponse(
                agent_name=agent_name,
                content=response,
                confidence=0.85, 
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise Exception(f"Agent {agent_name} error: {str(e)}")

    async def _synthesize_responses(
        self,
        query: str,
        context: str,
        chat_history: str,
        responses: Dict[str, Union[AgentResponse, DietPlan]]
    ) -> AgentResponse:
        """Synthesize final response from all agent responses"""
        try:
            formatted_responses = "\n\n".join([
                f"{name.upper()}:\n{response.content}"
                for name, response in responses.items()
                if name != 'synthesis_agent' and name != 'diet_plan' and hasattr(response, 'content')
            ])

            start_time = time.time()
            
            synthesis_response = await self.agents['synthesis_agent'].ainvoke({
                "input": query,
                "context": context,
                "query": query,
                "chat_history": chat_history,
                "agent_responses": formatted_responses
            })
            
            processing_time = time.time() - start_time
            
            metadata = {
                "processing_time": processing_time,
                "source_responses": len(responses),
                "context_used": bool(context)
            }
            
            return AgentResponse(
                agent_name="synthesis_agent",
                content=synthesis_response,
                confidence=0.9,
                metadata=metadata,
                processing_time=processing_time
            )

        except Exception as e:
            raise Exception(f"Synthesis error: {str(e)}")
    
def setup_streamlit_ui():
    """Setup Streamlit UI with dark sidebar"""
    st.set_page_config(
        page_title="Healthcare AI Assistant",
        page_icon="🏥",
        layout="wide"
    )
    st.markdown("""
        <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            border: 1px solid #dee2e6;
            background-color: black;
            font-size: 0.95rem;
        }
        .agent-card {
            padding: 0.8rem;
            margin-bottom: 0.5rem;
        }
        .metadata-section {
            font-size: 0.75rem;
            margin-top: 0.5rem;
        }
        .diet-plan-popup {
            padding: 0.8rem;
            border-radius: 0.5rem;
            margin-top: 0.3rem;
            margin-bottom: 0.5rem;
            border: 2px solid #e6c619;
            background-color: #fff9c4;
            color: #333;
            font-size: 0.9rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .diet-plan-item {
            margin: 0.15rem 0;
            font-size: 0.85rem;
        }
        .diet-plan-title {
            font-weight: bold;
            color: #e67300;
            font-size: 1rem;
            margin-bottom: 0.3rem;
            text-align: center;
            border-bottom: 1px solid #e6c619;
            padding-bottom: 0.3rem;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main application with dark sidebar and enhanced UI"""
    setup_streamlit_ui()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = HealthcareAgent()
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = AgentStatus()
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "show_diet_plan" not in st.session_state:
        st.session_state.show_diet_plan = False
    if "current_diet_plan" not in st.session_state:
        st.session_state.current_diet_plan = None
    
    with st.sidebar:
        st.markdown('<h3 style="color: #FFFFFF;">📋 Document Processing</h3>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload PDF or Image files",
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="document_uploader"
        )
        
        if uploaded_files:
            st.markdown('<h4 style="color: #FFFFFF;">📎 Selected Files</h4>', unsafe_allow_html=True)
            for file in uploaded_files:
                st.markdown(f"""
                    <div class="uploaded-file">
                        <div style="color: #FFFFFF;">📄 {file.name}</div>
                        <div style="color: #CCCCCC; font-size: 0.8rem; margin-top: 0.5rem;">
                            Type: {file.type}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            if st.button("🔄 Process Documents", key="process_docs"):
                with st.spinner("Processing documents..."):
                    async def process_docs():
                        await st.session_state.agent.process_documents(
                            uploaded_files,
                            st.session_state.agent_status.update_status
                        )
                        st.session_state.documents_processed = True
                    
                    asyncio.run(process_docs())
                    
        st.markdown('<h3 style="color: #FFFFFF;">🤖 Agent Status</h3>', unsafe_allow_html=True)
        st.session_state.agent_status.initialize_sidebar_placeholder()
    
    st.title("🏥 Healthcare AI Assistant")
    st.markdown("### 💬 Chat Interface")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        {message['content']['synthesis_agent'].content}
                    </div>
                """, unsafe_allow_html=True)
                
                # Display diet plan right after message if available
                if 'diet_plan' in message['content']:
                    diet_plan = message['content']['diet_plan']
                    st.markdown(f"""
                        <div class="diet-plan-popup">
                            <div class="diet-plan-title">🍽️ Diet Plan for Kidney Stone</div>
                            <div class="diet-plan-item">🍳 <b>Breakfast:</b> {diet_plan.breakfast}</div>
                            <div class="diet-plan-item">🥗 <b>Lunch:</b> {diet_plan.lunch}</div>
                            <div class="diet-plan-item">🍲 <b>Dinner:</b> {diet_plan.dinner}</div>
                            <div class="diet-plan-item">🥜 <b>Snacks:</b> {diet_plan.snacks}</div>
                            <div class="diet-plan-item">📝 <b>Note:</b> {diet_plan.notes}</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        {message['content']}
                    </div>
                """, unsafe_allow_html=True)
    
    if prompt := st.chat_input("Ask me about your health concerns..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                async def process_query():
                    return await st.session_state.agent.process_query(
                        prompt,
                        st.session_state.agent_status.update_status
                    )
                
                responses = asyncio.run(process_query())
                
                if responses:
                    response_placeholder.markdown(f"""
                        <div class="chat-message assistant">
                            {responses['synthesis_agent'].content}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": responses
                    })
                    
                    # Show diet plan directly in this response
                    if 'diet_plan' in responses:
                        diet_plan = responses['diet_plan']
                        st.markdown(f"""
                            <div class="diet-plan-popup">
                                <div class="diet-plan-title">🍽️ Diet Plan for Kidney Stone</div>
                                <div class="diet-plan-item">🍳 <b>Breakfast:</b> {diet_plan.breakfast}</div>
                                <div class="diet-plan-item">🥗 <b>Lunch:</b> {diet_plan.lunch}</div>
                                <div class="diet-plan-item">🍲 <b>Dinner:</b> {diet_plan.dinner}</div>
                                <div class="diet-plan-item">🥜 <b>Snacks:</b> {diet_plan.snacks}</div>
                                <div class="diet-plan-item">📝 <b>Note:</b> {diet_plan.notes}</div>
                            </div>
                        """, unsafe_allow_html=True)
                
            except Exception as e:
                response_placeholder.error(f"An error occurred: {str(e)}")
                
if __name__ == "__main__":
    main()
