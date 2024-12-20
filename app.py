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
        #Model Config
        model_config = {
            "temperature": 0,
        }
        
        genai.configure(api_key=api_key)
        self.model = GenerativeModel('gemini-1.5-flash', generation_config=model_config)

    def analyze_job_ad(self, job_ad_text):
        """
        Analyze a single job ad and determine if it needs rewriting
        
        Args:
            job_ad_text (str): Text of the job advertisement
        
        Returns:
            dict: Analysis results including needs_rewrite flag and reasoning
        """
        prompt = f"""You are an expert at writing convincing, attractive job ads to recruit candidates to companies. It is imperative to the company's success that you evaluate the quality of these job ads correctly. Evaluate this job advertisement and determine if it needs to be rewritten.
        
        Criteria for evaluation:
        1. Clarity of job responsibilities
        2. Specificity of required skills
        3. Professional language and tone
        4. Completeness of job description

        Job Advertisement:
        {job_ad_text}

        Respond with a JSON object that follows this exact format, with no additional text:
        {{
            "needs_rewrite": true/false, "reasoning": "Brief concise explanation of why the job ad needs or does not need rewriting and which criteria the job ad does well or is lacking in, and what the recommended changes are. The response should include \n\n1. Clarity of job responsibilities: \n\n2. Specificity of required skills: \n\n3. Professional language and tone: \n\n4. Completeness of Job Description: "
        }}

        Here is an example response:
        {{ "needs_rewrite": true, "reasoning": "This job advertisement needs significant rewriting. While it lists a salary range, which is positive, it lacks clarity in several key areas. \n\n1. Clarity of job responsibilities: The job description is vague. Phrases like "株式報酬制度等の制度設計に関するアドバイス" are not easily understood by non-Japanese speakers and even Japanese speakers may need more detail. Specific tasks and responsibilities should be listed. \n\n2. Specificity of required skills: The required skills are too general. Instead of "株式報酬に関する業務経験や知見のある方", the ad should specify the number of years of experience, types of equity compensation plans worked with, and specific software or tools used. Similarly, the legal knowledge requirements need more detail. \n\n3. Professional language and tone: While professional in tone, the ad is almost entirely in Japanese except for a few English phrases, creating inconsistency. The entire ad should be in either Japanese or English, depending on the target audience. \n\n4. Completeness of job description: The ad is missing crucial information such as company culture, benefits beyond work-from-home options, and the application process. It also lacks a compelling reason for candidates to apply. \n\nRecommended changes include providing detailed job responsibilities, quantifiable skills requirements, consistent language, a more comprehensive benefits package description, and a strong call to action." }}

        It is essential that the response has the correct JSON formatting. If there are quotes in the reasonsing, remember to use escape characters, \, before the quotes as that is needed in JSON formatting. Do not use any special characters such as bullet points in your JSON response.
        """
        
        try:
            response = self.model.generate_content(prompt)

            #Handling for multi-part responses
            if hasattr(response, 'parts'):
                response_text = ' '.join(part.text for part in response.parts)
            elif hasattr(response, 'candidates') and response.candidates:
                response_text = ' '.join(part.text for part in response.candidates[0].content.parts)
            else:
                raise ValueError("Unexpected response format from Gemini API")

            #debug
            st.write("Extracted Response Text:", response_text)
            
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
                st.write(f"{row['reference_id']}")
                st.write(f"Rewrite: {row['needs_rewrite']}")
                st.write(f"Reasoning: {row['reasoning']}")
            
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
