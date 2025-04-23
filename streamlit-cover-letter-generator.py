import streamlit as st
import requests
import json

def query_llama(prompt, model="llama3.2", temperature=0.7):
    """
    Send a query to Llama 3.2 running in Ollama and return the response.
    
    Args:
        prompt (str): The prompt to send to the model
        model (str): The model name to use (default: "llama3.2")
        temperature (float): Controls randomness in responses (0.0 to 1.0)
        
    Returns:
        str: The model's response
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response received")
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure the Ollama service is running."
    except requests.exceptions.HTTPError as e:
        return f"HTTP Error: {e}"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except json.JSONDecodeError:
        return "Error: Could not parse the response from Ollama."

def generate_prompt(job_title, hiring_company, current_company, years_exp, current_roles, job_requirements):
    """
    Generate a formatted prompt for the Llama model to create a cover letter.
    """
    prompt = f"""You are an expert of writing cover letter given a job description.
For the given job description, you will match the user's provided job roles and generate a suitable cover letter content that will closely matched with the job description provided by the hiring company. The generated cover letter should fit regular size one page including the Headers and Salutation part.

The job title open for hiring is "{job_title}"
The name of the hiring company name is "{hiring_company}".
The applicant's current work company is "{current_company}".
The applicant have worked in this role "{years_exp}" years.

Here is job description from the hiring company, 
"{job_requirements}"

Current roles.
Here is the roles description from the applicant, "{current_roles}"

Please generate a professional cover letter.
"""
    return prompt

# Set up the Streamlit app
st.set_page_config(page_title="Cover Letter Generator", layout="wide")

st.title("Cover Letter Generator with local LLM")
st.write("""
This application helps you generate a tailored cover letter based on your experience and job requirements.
Fill in the details below, and let Llama 3.2 create a personalized cover letter for you.
""")

# Check if Ollama is available
@st.cache_data(ttl=300)
def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            return True, available_models
        return False, []
    except:
        return False, []

ollama_available, available_models = check_ollama_status()

if not ollama_available:
    st.error("⚠️ Cannot connect to Ollama. Please make sure Ollama is running on localhost:11434.")
    st.info("Installation instructions: https://ollama.com/download")
    st.stop()

# Input fields
with st.form("cover_letter_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        job_title = st.text_input("Job Title", placeholder="e.g., AI Engineer")
        hiring_company = st.text_input("Hiring Company Name", placeholder="e.g., AlphaAI Corp.")
        current_company = st.text_input("Your Current Company", placeholder="e.g., LeopardAI Solutions")
    
    with col2:
        years_exp = st.text_input("Years of Experience", placeholder="e.g., 7")
        model_selection = st.selectbox("Select Model", ["llama3.2"] + [m for m in available_models if "llama" in m.lower()])
        temperature = st.slider("Creativity Level", min_value=0.1, max_value=1.0, value=0.7, step=0.1,
                               help="Lower values make the output more deterministic, higher values make it more creative")
    
    current_roles = st.text_area("Current Job Roles", height=150, 
                               placeholder="Describe your current responsibilities and achievements...")
    
    job_requirements = st.text_area("Job Requirements/Description", height=200,
                                  placeholder="Paste the job description here...")
    
    submitted = st.form_submit_button("Generate Cover Letter")

# Generate and display the cover letter
if submitted:
    if not job_title or not hiring_company or not current_company or not years_exp or not current_roles or not job_requirements:
        st.error("Please fill in all fields.")
    else:
        with st.spinner("Generating your cover letter... This may take a moment."):
            # Create the prompt
            prompt = generate_prompt(job_title, hiring_company, current_company, years_exp, current_roles, job_requirements)
            
            # Debug - Show the prompt if needed
            with st.expander("Show Generated Prompt"):
                st.text(prompt)
            
            # Query the model
            cover_letter = query_llama(prompt, model=model_selection, temperature=temperature)
            
            # Display the results
            st.subheader("Your Generated Cover Letter")
            st.write(cover_letter)
            
            # Add a download button
            st.download_button(
                label="Download Cover Letter",
                data=cover_letter,
                file_name="cover_letter.txt",
                mime="text/plain"
            )
            
            st.success("Cover letter generated successfully! Review and edit as needed before using.")

# Add some helpful tips
with st.expander("Tips for Better Results"):
    st.markdown("""
    ### Tips for getting better cover letters:
    
    1. **Be specific** in describing your current roles and achievements
    2. **Include keywords** from the job description in your current roles section
    3. **Quantify achievements** where possible (e.g., "Increased efficiency by 30%")
    4. **Check that the job requirements** are pasted completely
    5. **Adjust the creativity level** (temperature) if results are too generic or too creative
    6. **Always review and edit** the generated cover letter before using it
    """)

# Footer
st.markdown("---")
st.caption("This application uses Ollama to run Llama 3.2 locally on your machine. No data is sent to external servers.")
