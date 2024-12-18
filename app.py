import os
import streamlit as st
import pandas as pd
from google.generativeai import GenerativeModel
import google.generativeai as genai
import json
import re

class JobAdAnalyzer:
    def __init__(self, api_key):
        """
        Initialize the job ad analyzer with the Gemini API key
        
        Args:
            api_key (str): Google Gemini API key
        """
        genai.configure(api_key=api_key)
        self.model = GenerativeModel('gemini-pro')

    def analyze_job_ad(self, job_ad_text):
        """
        Analyze a single job ad and determine if it needs rewriting
        
        Args:
            job_ad_text (str): Text of the job advertisement
        
        Returns:
            dict: Analysis results including needs_rewrite flag and reasoning
        """
        prompt = f"""Evaluate this job advertisement and determine if it needs to be rewritten.
        
        Criteria for evaluation:
        1. Clarity of job responsibilities
        2. Specificity of required skills
        3. Professional language and tone
        4. Completeness of job description
        5. Potential bias or exclusionary language

        Job Advertisement:
        {job_ad_text}

        Provide a JSON response with the following structure:
        {{
            "needs_rewrite": true/false,
            "reasoning": "Brief explanation of why the job ad needs or does not need rewriting"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)

            # Clean the response in case it contains non json elements
            response_text = response.text.strip()
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)            

            # Attempt to parse the response
            analysis = json.loads(response_text)
            return analysis
        
        except Exception as e:
            return {
                "needs_rewrite": True,
                "reasoning": f"Analysis error: {str(e)}"
            }

def main():
    st.title("Job Advertisement Analyzer")
    
    # Retrieve API Key from environment
    api_key = os.environ.get('GEMINI_API_KEY')
    
    if not api_key:
        st.error("Gemini API Key not found. Please set the GEMINI_API_KEY environment variable.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validate columns
            required_columns = ['reference_id', 'job_ad_text']
            if not all(col in df.columns for col in required_columns):
                st.error(f"File must contain columns: {required_columns}")
                return
            
            # Initialize analyzer
            analyzer = JobAdAnalyzer(api_key)
            
            # Analyze job ads
            results = []
            progress_bar = st.progress(0)
            for i, row in df.iterrows():
                analysis = analyzer.analyze_job_ad(row['job_ad_text'])
                results.append({
                    'reference_id': row['reference_id'],
                    'needs_rewrite': analysis['needs_rewrite'],
                    'reasoning': analysis['reasoning']
                })
                progress_bar.progress((i + 1) / len(df))
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            # Separate job ads
            rewrite_ads = results_df[results_df['needs_rewrite']]['reference_id'].tolist()
            ok_ads = results_df[~results_df['needs_rewrite']]['reference_id'].tolist()
            
            # Display results
            st.subheader("Analysis Results")
            st.write(f"Total Job Ads: {len(results_df)}")
            st.write(f"Job Ads Needing Rewrite: {len(rewrite_ads)}")
            st.write(f"Job Ads OK to Post: {len(ok_ads)}")
            
            # Detailed results
            st.subheader("Detailed Analysis")
            for index, row in results_df.iterrows():
                st.write(f"{row['reference_id']}, {row['needs_rewrite']}, Reasoning: {row['reasoning']}")
            
            # Download options
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Full Results",
                data=csv,
                file_name='job_ad_analysis_results.csv',
                mime='text/csv'
            )
            
            # Separate download buttons for each category
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download Job Ads to Rewrite",
                    data='\n'.join(rewrite_ads),
                    file_name='job_ads_to_rewrite.txt'
                )
            
            with col2:
                st.download_button(
                    label="Download OK Job Ads",
                    data='\n'.join(ok_ads),
                    file_name='ok_job_ads.txt'
                )
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
