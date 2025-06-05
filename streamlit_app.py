import streamlit as st
import os
import json
import uuid
import threading
import time
from datetime import datetime
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor

# CrewAI imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

# Page configuration
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY")
    OUTPUT_DIR = "./ai-agent-output"
    RESULTS_NUM = 5

# Initialize session state
if 'jobs' not in st.session_state:
    st.session_state.jobs = {}
if 'current_job_id' not in st.session_state:
    st.session_state.current_job_id = None
if 'game_state' not in st.session_state:
    st.session_state.game_state = {
        'board': [],
        'flipped': [],
        'matched': [],
        'moves': 0,
        'pairs_found': 0,
        'game_started': False,
        'start_time': None
    }

# Custom CSS for styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    .header-text {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .subtitle-text {
        text-align: center;
        color: #6b7280;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .memory-card {
        width: 80px;
        height: 80px;
        background: #f3f4f6;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 3px solid transparent;
    }
    
    .memory-card:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .memory-card.flipped {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .memory-card.matched {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        border-color: #059669;
        color: white;
    }
    
    .game-stats {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 1rem 0;
    }
    
    .stat-item {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
    }
    
    .success-box {
        background: #f0fdf4;
        color: #16a34a;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #16a34a;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #fef2f2;
        color: #dc2626;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc2626;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize LLM and Search Client
@st.cache_resource
def initialize_clients():
    try:
        os.environ["GEMINI_API_KEY"] = Config.GEMINI_API_KEY
        basic_llm = LLM(model="gemini/gemini-1.5-flash", temperature=0)
        search_client = TavilyClient(api_key=Config.TAVILY_API_KEY)
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        return basic_llm, search_client
    except Exception as e:
        st.error(f"Failed to initialize clients: {e}")
        return None, None

basic_llm, search_client = initialize_clients()

# Pydantic Models
class SuggestedSearchQueries(BaseModel):
    queries: List[str] = Field(..., title="Suggested search queries to help the user self-learn a topic",
                               min_items=1, max_items=10)

class SingleSearchResult(BaseModel):
    title: str
    url: str = Field(..., title="The page URL")

class AllSearchResults(BaseModel):
    results: List[SingleSearchResult]

# Tools
@tool
def visual_search_tool(query: str) -> List[SingleSearchResult]:
    """Searches only within YouTube, Udemy, Coursera for visual resources"""
    try:
        results = []
        youtube_query = search_client.search(query=query+" site:youtube.com", max_results=Config.RESULTS_NUM)
        udemy_query = search_client.search(query="free "+query+" site:udemy.com", max_results=Config.RESULTS_NUM)
        coursera_query = search_client.search(query="free "+query+" site:coursera.org", max_results=Config.RESULTS_NUM)
        
        # Combine and format results
        for result in youtube_query['results']:
            results.append(SingleSearchResult(title=result['title'], url=result['url']))
        for result in udemy_query['results']:
            results.append(SingleSearchResult(title=result['title'], url=result['url']))
        for result in coursera_query['results']:
            results.append(SingleSearchResult(title=result['title'], url=result['url']))
            
        return results
    except Exception as e:
        st.error(f"Visual search error: {e}")
        return []

@tool
def text_search_tool(query: str) -> List[SingleSearchResult]:
    """Searches only within medium.com, arxiv.org, github.com, paperswithcode.com for text resources"""
    try:
        results = []
        medium_query = search_client.search(query=query+" site:medium.com", max_results=Config.RESULTS_NUM)
        arxiv_query = search_client.search(query=query+" site:arxiv.org", max_results=Config.RESULTS_NUM)
        github_query = search_client.search(query=query+" site:github.com", max_results=Config.RESULTS_NUM)
        paperswithcode_query = search_client.search(query=query+" site:paperswithcode.com", max_results=Config.RESULTS_NUM)
        
        # Combine and format results
        for result in medium_query['results']:
            results.append(SingleSearchResult(title=result['title'], url=result['url']))
        for result in arxiv_query['results']:
            results.append(SingleSearchResult(title=result['title'], url=result['url']))
        for result in github_query['results']:
            results.append(SingleSearchResult(title=result['title'], url=result['url']))
        for result in paperswithcode_query['results']:
            results.append(SingleSearchResult(title=result['title'], url=result['url']))
            
        return results
    except Exception as e:
        st.error(f"Text search error: {e}")
        return []

# Initialize agents
@st.cache_resource
def create_agents():
    if basic_llm is None:
        return None
    
    search_queries_recommendation_agent = Agent(
        role="Search Queries Recommendation Agent",
        goal="\n".join([
            "Generate a list of well-structured and general-purpose search queries based on a given learning topic and user level.",
            "Queries should be informative, specific, and useful for self-learning.",
            "Do not include the name of any platform, website, or content type.",
            "The goal is to create queries that can be used in general-purpose search engines."
        ]),
        backstory=(
            "This agent helps learners by generating highly relevant search queries for any topic and skill level. "
            "It avoids mentioning specific sources, allowing downstream agents to handle targeted search in different content formats."
        ),
        llm=basic_llm,
        verbose=True
    )

    visual_search_agent = Agent(
        role="Visual Learning Resources Agent",
        goal="To find visual educational resources (videos, playlists, courses) based on given search queries.",
        backstory="You specialize in finding helpful video-based learning materials for advanced learners from YouTube, Coursera, Udemy, and Khan Academy.",
        llm=basic_llm,
        verbose=True,
        tools=[visual_search_tool]
    )

    text_search_agent = Agent(
        role="Textual Learning Resources Agent",
        goal="To find rich, insightful, and trustworthy educational resources from top-tier domains.",
        backstory="You're a research assistant who specializes in discovering valuable content like articles, GitHub repos, research papers, and technical blogs from well-known educational platforms.",
        llm=basic_llm,
        verbose=True,
        tools=[text_search_tool]
    )

    summary_markdown_agent = Agent(
        role="Markdown Learning Report Designer",
        goal="Create a clean, structured Markdown (.md) report from visual and textual sources.",
        backstory=(
            "This agent builds a clean Markdown summary from visual and textual learning sources. "
            "The format is simple, readable, and structured with bullet points for easy navigation."
        ),
        llm=basic_llm,
        verbose=True,
    )
    
    return search_queries_recommendation_agent, visual_search_agent, text_search_agent, summary_markdown_agent

agents = create_agents()

def run_learning_crew(job_id, topic_name, learning_level, progress_callback):
    """Run the CrewAI workflow"""
    try:
        if agents is None:
            raise Exception("Agents not initialized properly")
            
        search_queries_recommendation_agent, visual_search_agent, text_search_agent, summary_markdown_agent = agents
        
        # Update job status
        st.session_state.jobs[job_id]['status'] = 'running'
        progress_callback('Initializing agents...', 15)
        
        # Create unique output directory for this job
        job_output_dir = os.path.join(Config.OUTPUT_DIR, f"job_{job_id}")
        os.makedirs(job_output_dir, exist_ok=True)
        
        progress_callback('Running search agents...', 40)
        
        # Tasks
        search_queries_task = Task(
            description=f"Generate search queries for topic: {topic_name} at {learning_level} level",
            expected_output="A JSON object containing a list of suggested general-purpose search queries.",
            output_json=SuggestedSearchQueries,
            output_file=os.path.join(job_output_dir, "step_1_suggested_search_queries.json"),
            agent=search_queries_recommendation_agent
        )
        
        progress_callback('Finding visual resources...', 60)
        
        visual_search_task = Task(
            description=f"Find visual resources for {topic_name} at {learning_level} level using the generated search queries",
            expected_output="A JSON file containing valid visual search results.",
            output_json=AllSearchResults,
            output_file=os.path.join(job_output_dir, "step_2_visual_results.json"),
            agent=visual_search_agent,
            context=[search_queries_task]
        )
        
        progress_callback('Finding textual resources...', 80)
        
        text_search_task = Task(
            description=f"Find textual resources for {topic_name} at {learning_level} level using the generated search queries",
            expected_output="A JSON file with the best educational search results.",
            output_json=AllSearchResults,
            output_file=os.path.join(job_output_dir, "step_3_textual_results.json"),
            agent=text_search_agent,
            context=[search_queries_task]
        )
        
        progress_callback('Creating summary...', 90)
        
        summary_task = Task(
            description=f"Create a structured Markdown report with all resources for learning {topic_name} at {learning_level} level",
            expected_output="A structured Markdown file with all links categorized by visual and textual resources.",
            output_file=os.path.join(job_output_dir, "summary_report.md"),
            agent=summary_markdown_agent,
            context=[visual_search_task, text_search_task]
        )
        
        # Create and run crew
        crew = Crew(
            agents=[search_queries_recommendation_agent, visual_search_agent, text_search_agent, summary_markdown_agent],
            tasks=[search_queries_task, visual_search_task, text_search_task, summary_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff(inputs={
            'topic_name': topic_name, 
            'learning_level': learning_level
        })
        
        # Load results
        summary_path = os.path.join(job_output_dir, 'summary_report.md')
        summary_content = ""
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_content = f.read()
        
        st.session_state.jobs[job_id].update({
            'status': 'completed',
            'result_path': job_output_dir,
            'summary': summary_content,
            'completed_at': datetime.now().isoformat()
        })
        
        progress_callback('Complete!', 100)
        
    except Exception as e:
        st.session_state.jobs[job_id].update({
            'status': 'failed',
            'error': str(e)
        })
        st.error(f"Job {job_id} failed: {e}")

# Memory Game Functions
def initialize_memory_game():
    """Initialize the memory game"""
    emojis = ['🧠', '💡', '📚', '🎯', '🚀', '⭐', '🎨', '🔬']
    cards = emojis * 2  # Create pairs
    
    # Shuffle cards
    import random
    random.shuffle(cards)
    
    st.session_state.game_state.update({
        'board': cards,
        'flipped': [False] * 16,
        'matched': [False] * 16,
        'moves': 0,
        'pairs_found': 0,
        'game_started': True,
        'start_time': time.time(),
        'selected_cards': []
    })

def render_memory_game():
    """Render the memory game"""
    if not st.session_state.game_state['game_started']:
        if st.button("🎮 Start Memory Game", key="start_game"):
            initialize_memory_game()
            st.rerun()
        return
    
    game_state = st.session_state.game_state
    
    # Game stats
    elapsed_time = int(time.time() - game_state['start_time']) if game_state['start_time'] else 0
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-item">⏰ Time: {minutes:02d}:{seconds:02d}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-item">🎯 Moves: {game_state["moves"]}</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat-item">🏆 Pairs: {game_state["pairs_found"]}/8</div>', unsafe_allow_html=True)
    
    # Game board
    st.markdown("### Memory Game Board")
    
    # Create 4x4 grid
    for row in range(4):
        cols = st.columns(4)
        for col in range(4):
            idx = row * 4 + col
            card_emoji = game_state['board'][idx]
            
            # Determine card state
            if game_state['matched'][idx]:
                # Matched card - show emoji with green background
                cols[col].markdown(
                    f'<div class="memory-card matched">{card_emoji}</div>', 
                    unsafe_allow_html=True
                )
            elif game_state['flipped'][idx]:
                # Flipped card - show emoji
                cols[col].markdown(
                    f'<div class="memory-card flipped">{card_emoji}</div>', 
                    unsafe_allow_html=True
                )
            else:
                # Hidden card - clickable
                if cols[col].button("❓", key=f"card_{idx}"):
                    handle_card_click(idx)
    
    # Game controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Game", key="new_game"):
            initialize_memory_game()
            st.rerun()
    with col2:
        if st.button("⏸️ Reset", key="reset_game"):
            st.session_state.game_state['game_started'] = False
            st.rerun()
    
    # Check for game completion
    if game_state['pairs_found'] == 8:
        st.success(f"🎉 Congratulations! You completed the game in {minutes:02d}:{seconds:02d} with {game_state['moves']} moves!")

def handle_card_click(idx):
    """Handle memory card click"""
    game_state = st.session_state.game_state
    
    # Don't allow clicking already flipped or matched cards
    if game_state['flipped'][idx] or game_state['matched'][idx]:
        return
    
    # Don't allow more than 2 cards to be selected
    if len(game_state['selected_cards']) >= 2:
        return
    
    # Flip the card
    game_state['flipped'][idx] = True
    game_state['selected_cards'].append(idx)
    
    # Check for match when 2 cards are selected
    if len(game_state['selected_cards']) == 2:
        game_state['moves'] += 1
        check_for_match()
    
    st.rerun()

def check_for_match():
    """Check if two selected cards match"""
    game_state = st.session_state.game_state
    selected = game_state['selected_cards']
    
    if len(selected) != 2:
        return
    
    idx1, idx2 = selected
    card1 = game_state['board'][idx1]
    card2 = game_state['board'][idx2]
    
    if card1 == card2:
        # Match found
        game_state['matched'][idx1] = True
        game_state['matched'][idx2] = True
        game_state['pairs_found'] += 1
    else:
        # No match - flip cards back after a delay
        time.sleep(0.5)  # Brief delay to show the cards
        game_state['flipped'][idx1] = False
        game_state['flipped'][idx2] = False
    
    # Clear selected cards
    game_state['selected_cards'] = []

# Main App
def main():
    # Header
    st.markdown('<h1 class="header-text">🧠 AI Learning Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Discover curated learning resources tailored to your skill level and interests</p>', unsafe_allow_html=True)
    
    # Check if APIs are configured
    if Config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY" or Config.TAVILY_API_KEY == "YOUR_TAVILY_API_KEY":
        st.error("⚠️ Please configure your API keys in the environment variables or update the Config class.")
        st.info("You need to set GEMINI_API_KEY and TAVILY_API_KEY environment variables.")
        return
    
    # Sidebar for form
    with st.sidebar:
        st.header("🎯 Learning Preferences")
        
        with st.form("learning_form"):
            topic = st.text_input(
                "💡 What would you like to learn?",
                placeholder="e.g., Data Science, Machine Learning, Web Development",
                max_chars=100
            )
            
            level = st.selectbox(
                "📈 Your current skill level",
                options=["", "beginner", "intermediate", "advanced"],
                format_func=lambda x: {
                    "": "Select your level",
                    "beginner": "Beginner - Just starting out",
                    "intermediate": "Intermediate - Some experience", 
                    "advanced": "Advanced - Extensive knowledge"
                }[x]
            )
            
            submitted = st.form_submit_button("🔍 Find Learning Resources", use_container_width=True)
        
        if submitted and topic and level:
            # Generate job ID
            job_id = str(uuid.uuid4())[:8]
            
            # Initialize job
            st.session_state.jobs[job_id] = {
                'status': 'starting',
                'progress': 'Preparing...',
                'topic': topic,
                'level': level,
                'started_at': datetime.now().isoformat()
            }
            
            st.session_state.current_job_id = job_id
            
            # Show success message
            st.success(f"✅ Started learning resource search for: **{topic}** ({level})")
    
    # Main content area
    if st.session_state.current_job_id:
        job_id = st.session_state.current_job_id
        job = st.session_state.jobs.get(job_id)
        
        if job and job['status'] in ['starting', 'running']:
            # Show progress
            st.subheader("🔄 Generating Learning Resources")
            
            # Progress bar
            progress_placeholder = st.empty()
            
            def update_progress(text, percentage):
                progress_placeholder.progress(percentage / 100, text=text)
            
            # Run the crew if not already running
            if job['status'] == 'starting':
                job['status'] = 'running'
                
                # Run in thread to avoid blocking
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        run_learning_crew, 
                        job_id, 
                        job['topic'], 
                        job['level'], 
                        update_progress
                    )
                    
                    # Show memory game while processing
                    st.subheader("🎮 Memory Challenge")
                    st.info("Keep your brain active while we prepare your learning resources!")
                    render_memory_game()
                    
                    # Check if job is complete
                    if job['status'] == 'completed':
                        st.rerun()
            
            else:
                # Show memory game for running jobs
                st.subheader("🎮 Memory Challenge") 
                st.info("Keep your brain active while we prepare your learning resources!")
                render_memory_game()
                
                # Auto-refresh to check status
                time.sleep(2)
                st.rerun()
        
        elif job and job['status'] == 'completed':
            # Show results
            st.subheader("📋 Learning Resources Report")
            
            if job.get('summary'):
                # Show download button
                if st.download_button(
                    label="📥 Download Report",
                    data=job['summary'],
                    file_name=f"learning_report_{job['topic'].replace(' ', '_').lower()}_{job_id}.md",
                    mime="text/markdown"
                ):
                    st.success("Report downloaded successfully!")
                
                # Display the summary
                st.markdown("### 📖 Summary")
                st.markdown(job['summary'])
            else:
                st.error("No summary available")
        
        elif job and job['status'] == 'failed':
            st.error(f"❌ Job failed: {job.get('error', 'Unknown error')}")
            
            # Reset button
            if st.button("🔄 Try Again"):
                st.session_state.current_job_id = None
                st.rerun()
    
    else:
        # Welcome message
        st.markdown("### 👋 Welcome!")
        st.info("Please fill out the form in the sidebar to get started with finding personalized learning resources.")
        
        # Show sample memory game
        st.subheader("🎮 Try the Memory Game")
        st.info("Challenge yourself with our memory game while you decide what to learn!")
        render_memory_game()

if __name__ == "__main__":
    main()