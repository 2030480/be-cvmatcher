import json
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from typing import Dict, List
from app.models import AnalysisResult, Strength, Weakness

class CVAnalyzer:
    """Service for analyzing CV against job description using GitHub Models GPT-4"""
    
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.model_endpoint = os.getenv("GITHUB_MODEL_ENDPOINT", "https://models.github.ai/inference")
        self.model_id = os.getenv("GITHUB_MODEL_ID", "openai/gpt-5")
        # Optional comma-separated candidates: e.g., "openai/gpt-4o-mini,openai/gpt-4o"
        candidates_from_env = os.getenv("GITHUB_MODEL_CANDIDATES", "").strip()
        self.model_candidates = [m.strip() for m in candidates_from_env.split(",") if m.strip()] or [
            self.model_id,
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
        ]
        
        if not self.github_token:
            raise ValueError("GITHUB_TOKEN environment variable is required")
    
    async def analyze(self, cv_text: str, job_description: str) -> AnalysisResult:
        """
        Analyze CV against job description and return structured results
        """
        try:
            # Create the analysis prompt
            prompt = self._create_analysis_prompt(cv_text, job_description)
            
            # Call GitHub Models via Azure AI Inference client
            response_text = await self._call_github_models(prompt)
            
            # Parse the response into structured data
            analysis_result = self._parse_analysis_response(response_text)
            
            return analysis_result
            
        except Exception as e:
            raise Exception(f"Error during CV analysis: {str(e)}")
    
    def _create_analysis_prompt(self, cv_text: str, job_description: str) -> str:
        """Create a comprehensive prompt for CV analysis"""
        
        prompt = f"""
You are an expert HR analyst specializing in CV evaluation. Analyze the following CV against the job description and provide a detailed assessment.

JOB DESCRIPTION:
{job_description}

CV CONTENT:
{cv_text}

Please analyze the CV and provide a response in the following JSON format:

{{
    "match_percentage": <integer between 0-100>,
    "strengths": [
        {{
            "title": "<strength title>",
            "description": "<detailed explanation of why this is a strength>"
        }}
    ],
    "weaknesses": [
        {{
            "title": "<weakness title>",
            "description": "<explanation of the weakness>",
            "suggestion": "<specific actionable suggestion for improvement>"
        }}
    ],
    "summary": "<overall assessment summary in 2-3 sentences>"
}}

ANALYSIS CRITERIA:
1. Technical skills alignment (35% weight)
2. Experience level and relevance (25% weight)
3. Education and certifications (20% weight)
4. Soft skills and cultural fit (15% weight)
5. Additional qualifications and achievements (5% weight)

REQUIREMENTS:
- Provide exactly 4 strengths
- Provide exactly 5 weaknesses
- Match percentage should be realistic and justified
- Strengths should highlight the best matching aspects
- Weaknesses should focus on gaps that matter most for this role
- Suggestions should be specific and actionable
- Consider both explicit matches and transferable skills
- Account for experience level expectations vs. actual experience

Return ONLY the JSON response, no additional text.
"""
        return prompt
    
    async def _call_github_models(self, prompt: str) -> str:
        """Call GitHub Models via Azure AI Inference client with model fallbacks"""
        last_error: Exception | None = None
        for candidate_model in self.model_candidates:
            try:
                client = ChatCompletionsClient(
                    endpoint=self.model_endpoint,
                    credential=AzureKeyCredential(self.github_token),
                )

                messages = [
                    SystemMessage("You are an expert HR analyst. Provide responses in valid JSON format only."),
                    UserMessage(prompt),
                ]

                response = client.complete(
                    messages=messages,
                    model=candidate_model,
                )

                if not response or not response.choices or not response.choices[0].message:
                    raise Exception("No response returned from GitHub Models")

                # Success: cache the working model_id for subsequent calls
                self.model_id = candidate_model
                return response.choices[0].message.content
            except Exception as e:
                # If unavailable model, try next; otherwise re-raise
                error_text = str(e).lower()
                if "unavailable model" in error_text or "unavailable_model" in error_text or "unknown model" in error_text:
                    last_error = e
                    continue
                # Other errors should not be swallowed
                raise Exception(f"Error calling GitHub Models API: {str(e)}")
        # If all candidates failed
        raise Exception(
            "Error calling GitHub Models API: none of the candidate models are available. "
            f"Tried: {', '.join(self.model_candidates)}. Last error: {last_error}"
        )
    
    def _parse_analysis_response(self, response: str) -> AnalysisResult:
        """Parse the JSON response from GitHub Models into structured data"""
        
        try:
            # Clean the response (remove any markdown formatting)
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            data = json.loads(response)
            
            # Validate required fields
            required_fields = ["match_percentage", "strengths", "weaknesses", "summary"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate match percentage
            match_percentage = data["match_percentage"]
            if not isinstance(match_percentage, int) or match_percentage < 0 or match_percentage > 100:
                raise ValueError("Match percentage must be an integer between 0 and 100")
            
            # Parse strengths
            strengths = []
            for strength_data in data["strengths"]:
                if "title" not in strength_data or "description" not in strength_data:
                    raise ValueError("Strength missing title or description")
                strengths.append(Strength(
                    title=strength_data["title"],
                    description=strength_data["description"]
                ))
            
            # Parse weaknesses
            weaknesses = []
            for weakness_data in data["weaknesses"]:
                required_weakness_fields = ["title", "description", "suggestion"]
                for field in required_weakness_fields:
                    if field not in weakness_data:
                        raise ValueError(f"Weakness missing {field}")
                weaknesses.append(Weakness(
                    title=weakness_data["title"],
                    description=weakness_data["description"],
                    suggestion=weakness_data["suggestion"]
                ))
            
            # Validate counts
            if len(strengths) != 4:
                print(f"Warning: Expected 4 strengths, got {len(strengths)}")
            if len(weaknesses) != 5:
                print(f"Warning: Expected 5 weaknesses, got {len(weaknesses)}")
            
            return AnalysisResult(
                match_percentage=match_percentage,
                strengths=strengths,
                weaknesses=weaknesses,
                summary=data["summary"]
            )
            
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response from GitHub Models: {str(e)}")
        except Exception as e:
            raise Exception(f"Error parsing analysis response: {str(e)}")
    
    def _fallback_analysis(self, cv_text: str, job_description: str) -> AnalysisResult:
        """Fallback analysis if GitHub Models fails"""
        
        # Basic keyword matching fallback
        cv_words = set(cv_text.lower().split())
        job_words = set(job_description.lower().split())
        common_words = cv_words.intersection(job_words)
        
        match_percentage = min(100, len(common_words) * 2)
        
        return AnalysisResult(
            match_percentage=match_percentage,
            strengths=[
                Strength(title="Basic Match Found", description="Some keywords match between CV and job description"),
                Strength(title="Document Processed", description="CV was successfully processed and analyzed"),
                Strength(title="Format Compatible", description="CV format is supported by the system"),
                Strength(title="Content Available", description="CV contains readable text content")
            ],
            weaknesses=[
                Weakness(title="Detailed Analysis Unavailable", description="Full AI analysis could not be completed", suggestion="Try again later or check your connection"),
                Weakness(title="Limited Matching", description="Only basic keyword matching was performed", suggestion="Ensure your CV contains relevant keywords"),
                Weakness(title="No Skill Analysis", description="Technical skills could not be properly analyzed", suggestion="List your technical skills clearly"),
                Weakness(title="No Experience Evaluation", description="Work experience could not be evaluated", suggestion="Include detailed work history"),
                Weakness(title="No Education Assessment", description="Educational background was not assessed", suggestion="Include education and certifications")
            ],
            summary="Basic analysis completed with limited functionality. Full AI analysis was not available."
        ) 